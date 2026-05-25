#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型预测与 Montel 预测对比脚本（含真实电价）

功能：将训练输出的预测结果、Montel 预测与真实电价进行时间对齐对比。

发布时间点（瑞典时间）：
  - 00:00 发布 → begin 是次日 00:00（target_start_days=1）
  - 12:00 发布 → begin 是后日 00:00（target_start_days=2）

使用方法：
    python compare_predictions.py \
        --pred_csv outputs/xxx/predictions_business.csv \
        --montel_prefix_csv sourceData/SE2_Price_Spot_EUR_MWh_H_Forecast/forecast_issued_00_prefix_latest.csv \
        --montel_postfix_csv sourceData/SE2_Price_Spot_EUR_MWh_H_Forecast/forecast_issued_12_postfix_latest.csv \
        --actual_csv sourceData/SE2_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv \
        --out_dir outputs/comparison
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ========== 指标计算函数 ==========
def mae(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs(y[mask] - yhat[mask]))) if mask.any() else np.nan


def rmse(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(np.mean((y[mask] - yhat[mask]) ** 2))) if mask.any() else np.nan


def bias(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(yhat[mask] - y[mask])) if mask.any() else np.nan


def wape(y, yhat, eps: float = 1e-8) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not mask.any():
        return np.nan
    denom = float(np.sum(np.abs(y[mask])))
    return float(np.sum(np.abs(y[mask] - yhat[mask])) / max(denom, eps))


# ========== 时区工具函数 ==========
LOCAL_TZ = "Europe/Stockholm"


def to_utc_ts(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def localize_naive_local(ts_naive: pd.Timestamp, tz: str) -> pd.Timestamp:
    return pd.Timestamp(ts_naive).tz_localize(tz)


def is_dst(ts_utc: pd.Timestamp) -> bool:
    """判断某个 UTC 时间点是否处于夏令时（欧洲中部夏令时）"""
    local = ts_utc.tz_convert(LOCAL_TZ)
    offset_hours = local.utcoffset().total_seconds() / 3600.0
    return offset_hours > 1.0


def get_utc_offset(ts_utc: pd.Timestamp) -> float:
    """获取某个 UTC 时间点对应的本地时区偏移（小时）"""
    local = ts_utc.tz_convert(LOCAL_TZ)
    return local.utcoffset().total_seconds() / 3600.0


# ========== 业务口径配置 ==========
def business_horizon_config(issued_hour_local: int, target_start_days: int, target_hours: int) -> dict:
    """
    计算业务口径下的预测窗口配置。

    参数：
        issued_hour_local: 本地发布时间小时（0 或 12）
        target_start_days: 目标开始天数（1=次日0点，2=后日0点）
        target_hours: 目标预测小时数

    返回：
        包含 first_target_horizon, last_target_horizon, model_h_required 的字典
    """
    if issued_hour_local not in [0, 12]:
        raise ValueError("issued_hour_local 必须是 0 或 12")

    first_target_horizon = target_start_days * 24 - issued_hour_local + 1

    if first_target_horizon < 1:
        raise ValueError("目标开始时间不能早于 cutoff")

    last_target_horizon = first_target_horizon + target_hours - 1

    return {
        "issued_hour_local": issued_hour_local,
        "target_start_days": target_start_days,
        "target_hours": target_hours,
        "first_target_horizon": first_target_horizon,
        "last_target_horizon": last_target_horizon,
        "model_h_required": last_target_horizon,
    }


# ========== 真实电价数据加载 ==========
def load_actual_price(actual_csv: str, date_col: str = None, target_col: str = None,
                     csv_sep: str = ",", csv_encoding: str = "utf-8") -> pd.DataFrame:
    """加载真实电价数据（UTC 时间）"""
    p = Path(actual_csv)
    if not p.exists():
        raise FileNotFoundError(f"真实电价文件不存在：{p}")

    try:
        raw = pd.read_csv(p, sep=csv_sep, encoding=csv_encoding)
    except UnicodeDecodeError:
        raw = pd.read_csv(p, sep=csv_sep, encoding="utf-8-sig")

    if date_col is None:
        date_candidates = [c for c in raw.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_candidates:
            date_col = date_candidates[0]
        else:
            raise ValueError(f"找不到日期列，请指定 --date_col")

    if target_col is None:
        target_candidates = [c for c in raw.columns if c.lower() not in [date_col.lower(), 'date', 'time']]
        if target_candidates:
            target_col = target_candidates[0]
        else:
            raise ValueError(f"找不到价格列，请指定 --target_col")

    df = pd.DataFrame({
        "ds_utc": pd.to_datetime(raw[date_col], utc=True, errors="coerce"),
        "y": pd.to_numeric(raw[target_col], errors="coerce"),
    })
    df = df.dropna(subset=["ds_utc"])
    df = df.drop_duplicates(subset=["ds_utc"], keep="last")
    df = df.sort_values("ds_utc")

    return df


# ========== Montel 数据加载 ==========
H_COL_RE = re.compile(r"^h(\d+)$", re.IGNORECASE)


def load_montel_long(montel_csv: str, target_hours: int, csv_sep: str = ",", csv_encoding: str = "utf-8") -> pd.DataFrame:
    """
    读取 Montel 预测文件（宽表转长表）

    输入格式：issued_utc, begin_utc, h001, h002, ...
    - issued_utc：发布时间（UTC）
    - begin_utc：目标窗口开始时间（UTC）
    - h001, h002, ...：从 begin_utc 开始的第 1, 2, ... 小时预测值

    输出格式：issued_utc, begin_utc, ds_utc, target_horizon, montel
    """
    p = Path(montel_csv)
    if not p.exists():
        raise FileNotFoundError(f"Montel 预测文件不存在：{p}")

    try:
        raw = pd.read_csv(p, sep=csv_sep, encoding=csv_encoding)
    except UnicodeDecodeError:
        raw = pd.read_csv(p, sep=csv_sep, encoding="utf-8-sig")

    raw = raw.dropna(axis=1, how="all")

    h_cols = []
    for c in raw.columns:
        m = H_COL_RE.fullmatch(str(c).strip())
        if m:
            h_cols.append((c, int(m.group(1))))

    if not h_cols:
        raise ValueError("Montel 文件中没有找到 h001, h002, ... 这类预测列。")

    h_cols = sorted(h_cols, key=lambda x: x[1])
    selected = [(c, h) for c, h in h_cols if 1 <= h <= target_hours]
    h_cols = [c for c, _ in selected]

    df = raw[["issued_utc", "begin_utc"] + h_cols].copy()
    df["issued_utc"] = pd.to_datetime(df["issued_utc"], utc=True, errors="coerce")
    df["begin_utc"] = pd.to_datetime(df["begin_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["issued_utc", "begin_utc"])

    long_df = df.melt(
        id_vars=["issued_utc", "begin_utc"],
        value_vars=h_cols,
        var_name="horizon_col",
        value_name="montel",
    )
    long_df["target_horizon"] = long_df["horizon_col"].str.extract(r"(\d+)").astype(int)
    long_df["montel"] = pd.to_numeric(long_df["montel"], errors="coerce")
    long_df["ds_utc"] = long_df["begin_utc"] + pd.to_timedelta(long_df["target_horizon"] - 1, unit="h")

    return long_df[["issued_utc", "begin_utc", "ds_utc", "target_horizon", "montel"]]


# ========== 数据对齐函数 ==========
def align_predictions_with_actual(pred_df: pd.DataFrame, montel_df: pd.DataFrame,
                                   actual_df: pd.DataFrame, insured_time: int) -> pd.DataFrame:
    """
    将模型预测、Montel 预测与真实电价按时间对齐

    参数：
        pred_df: 模型预测结果
        montel_df: Montel 预测结果
        actual_df: 真实电价
        insured_time: 发布时间点（0 或 12）

    对齐策略：
    1. 以 Montel 数据为主表进行对齐（只保留 Montel 中有数据的时间点）
    2. 通过 ds_utc, issued_utc, begin_utc 对齐
    3. Montel 的 target_horizon 是相对 horizon（从 begin_utc 开始）
    """
    # 标准化 Montel 时间格式
    montel = montel_df.copy()
    if montel.empty:
        return pd.DataFrame()
    
    montel["ds_utc"] = pd.to_datetime(montel["ds_utc"], utc=True)
    montel["issued_utc"] = pd.to_datetime(montel["issued_utc"], utc=True)
    montel["begin_utc"] = pd.to_datetime(montel["begin_utc"], utc=True)

    # 标准化模型预测时间格式
    pred = pred_df.copy()
    pred["ds_utc"] = pd.to_datetime(pred["ds_utc"], utc=True)
    pred["issued_utc"] = pd.to_datetime(pred["issued_utc"], utc=True)
    pred["begin_utc"] = pd.to_datetime(pred["begin_utc"], utc=True)

    # 标准化真实电价时间格式
    actual = actual_df.copy()
    actual["ds_utc"] = pd.to_datetime(actual["ds_utc"], utc=True)

    # 以 Montel 为主表，左连接模型预测（只保留 Montel 中有数据的时间点）
    out = montel.merge(
        pred,
        on=["ds_utc", "issued_utc", "begin_utc"],
        how="left"
    )

    # 合并真实电价
    out = out.merge(actual[["ds_utc", "y"]], on="ds_utc", how="left", suffixes=("", "_actual"))

    # 用真实电价覆盖模型预测中的 y
    if "y_actual" in out.columns:
        out["y"] = out["y_actual"].combine_first(out["y"])
        out = out.drop(columns=["y_actual"])

    return out


# ========== 指标汇总函数 ==========
def summarize_metrics(df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """汇总计算各模型的指标（与真实电价比较）"""
    rows = []
    
    # 计算所有模型都有数据的掩码（公平比较）
    all_models_mask = np.ones(len(df), dtype=bool)
    for model in model_cols:
        if model in df.columns:
            all_models_mask &= np.isfinite(df[model].values)
    
    y = np.asarray(df["y"], dtype=float)
    # 只使用所有模型都有数据的行
    y = y[all_models_mask]

    for model in model_cols:
        if model not in df.columns:
            continue
        yhat = np.asarray(df[model], dtype=float)
        # 只使用所有模型都有数据的行
        yhat = yhat[all_models_mask]
        valid_mask = np.isfinite(y) & np.isfinite(yhat)

        if valid_mask.any():
            rows.append({
                "model": model,
                "count": int(valid_mask.sum()),
                "mae": mae(y[valid_mask], yhat[valid_mask]),
                "rmse": rmse(y[valid_mask], yhat[valid_mask]),
                "bias": bias(y[valid_mask], yhat[valid_mask]),
                "wape": wape(y[valid_mask], yhat[valid_mask]),
            })
        else:
            rows.append({
                "model": model,
                "count": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "bias": np.nan,
                "wape": np.nan,
            })

    return pd.DataFrame(rows)


def summarize_by_forecast_day(df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """按预测天数汇总指标"""
    rows = []

    for forecast_day, part in df.groupby("forecast_day", dropna=False):
        y = np.asarray(part["y"], dtype=float)

        for model in model_cols:
            if model not in part.columns:
                continue
            yhat = np.asarray(part[model], dtype=float)
            valid_mask = np.isfinite(y) & np.isfinite(yhat)

            if valid_mask.any():
                rows.append({
                    "forecast_day": forecast_day,
                    "model": model,
                    "count": int(valid_mask.sum()),
                    "mae": mae(y[valid_mask], yhat[valid_mask]),
                    "rmse": rmse(y[valid_mask], yhat[valid_mask]),
                    "bias": bias(y[valid_mask], yhat[valid_mask]),
                    "wape": wape(y[valid_mask], yhat[valid_mask]),
                })

    return pd.DataFrame(rows)


def summarize_by_target_date(df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """按目标日期汇总指标"""
    rows = []
    df = df.copy()
    df["target_date"] = df["ds_utc"].dt.date.astype(str)

    for target_date, part in df.groupby("target_date", dropna=False):
        y = np.asarray(part["y"], dtype=float)

        for model in model_cols:
            if model not in part.columns:
                continue
            yhat = np.asarray(part[model], dtype=float)
            valid_mask = np.isfinite(y) & np.isfinite(yhat)

            if valid_mask.any():
                rows.append({
                    "target_date": target_date,
                    "model": model,
                    "count": int(valid_mask.sum()),
                    "mae": mae(y[valid_mask], yhat[valid_mask]),
                    "rmse": rmse(y[valid_mask], yhat[valid_mask]),
                    "bias": bias(y[valid_mask], yhat[valid_mask]),
                    "wape": wape(y[valid_mask], yhat[valid_mask]),
                })

    return pd.DataFrame(rows)


def summarize_by_issued_hour(df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """按发布时间点（00:00 或 12:00）汇总指标"""
    rows = []
    df = df.copy()

    if "issued_hour_local" not in df.columns:
        if "issued_utc" in df.columns:
            df["issued_hour_local"] = pd.to_datetime(df["issued_utc"], utc=True).dt.tz_convert(LOCAL_TZ).dt.hour
        else:
            return pd.DataFrame(rows)

    for issued_hour, part in df.groupby("issued_hour_local", dropna=False):
        y = np.asarray(part["y"], dtype=float)

        # 跳过 NaN 的发布时间
        if pd.isna(issued_hour):
            continue

        for model in model_cols:
            if model not in part.columns:
                continue
            yhat = np.asarray(part[model], dtype=float)
            valid_mask = np.isfinite(y) & np.isfinite(yhat)

            if valid_mask.any():
                issued_label = f"{int(issued_hour):02d}:00"
                rows.append({
                    "issued_hour_local": issued_hour,
                    "issued_label": issued_label,
                    "model": model,
                    "count": int(valid_mask.sum()),
                    "mae": mae(y[valid_mask], yhat[valid_mask]),
                    "rmse": rmse(y[valid_mask], yhat[valid_mask]),
                    "bias": bias(y[valid_mask], yhat[valid_mask]),
                    "wape": wape(y[valid_mask], yhat[valid_mask]),
                })

    return pd.DataFrame(rows)


# ========== 绘图函数 ==========
def plot_comparison(pred_df: pd.DataFrame, model_cols: list, out_path: Path, insured_time: int = 0):
    """按业务窗口绘制预测对比图（每个窗口单独绘制）"""
    pred_df = pred_df.copy()

    if "issued_hour_local" not in pred_df.columns:
        if "issued_utc" in pred_df.columns:
            pred_df["issued_hour_local"] = pd.to_datetime(pred_df["issued_utc"], utc=True).dt.tz_convert(LOCAL_TZ).dt.hour
        else:
            pred_df["issued_hour_local"] = 0

    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    montel_label = "Montel"

    # 按业务窗口分组（每个 issued_utc + begin_utc 组合是一个业务窗口）
    pred_df["window_key"] = pred_df["issued_utc"].astype(str) + "_" + pred_df["begin_utc"].astype(str)
    windows = pred_df.groupby("window_key")

    # 只绘制前5个窗口作为示例
    plot_count = 0
    max_plots = 5

    for window_key, window_data in windows:
        if plot_count >= max_plots:
            break

        # 跳过数据太少的窗口
        if len(window_data) < 24:
            continue

        window_data = window_data.sort_values("ds_utc")
        issued_utc = window_data["issued_utc"].iloc[0]
        begin_utc = window_data["begin_utc"].iloc[0]
        issued_hour_local = window_data["issued_hour_local"].iloc[0]

        fig, ax = plt.subplots(figsize=(16, 8))

        # 绘制真实电价
        valid_mask = window_data["y"].notna()
        if valid_mask.any():
            ax.plot(window_data.loc[valid_mask, "ds_utc"], window_data.loc[valid_mask, "y"],
                    marker=".", linewidth=2.0, label="Actual Price", color="#1f77b4")

        # 绘制模型预测
        for i, model in enumerate(model_cols):
            if model not in window_data.columns:
                continue
            mask = window_data[model].notna()
            if mask.any():
                ax.plot(window_data.loc[mask, "ds_utc"], window_data.loc[mask, model],
                        marker=".", linewidth=1.5, label=model, color=colors[i % len(colors)])

        # 绘制 Montel 预测
        if "montel" in window_data.columns:
            montel_part = window_data.dropna(subset=["montel"])
            if not montel_part.empty:
                ax.plot(montel_part["ds_utc"], montel_part["montel"],
                        marker=".", linewidth=1.5, label=montel_label, color="#e377c2", linestyle="--")

        # 设置标题和标签
        title = f"Forecast Window: Issued={issued_utc}, Begin={begin_utc}, Local Hour={issued_hour_local}:00"
        ax.set_title(title)
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # 保存每个窗口的图
        window_out_path = out_path.parent / f"comparison_window_{plot_count}.png"
        plt.tight_layout()
        plt.savefig(window_out_path, dpi=200, bbox_inches="tight")
        plt.close()

        plot_count += 1

    print(f"已生成 {plot_count} 个业务窗口的对比图")


# ========== 主函数 ==========
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("  模型预测与 Montel 预测对比分析")
    print("  (与真实电价比较)")
    print("========================================")

    # 1. 加载模型预测结果
    print(f"\n1. 加载模型预测文件: {args.pred_csv}")
    if not Path(args.pred_csv).exists():
        raise FileNotFoundError(f"模型预测文件不存在: {args.pred_csv}")
    pred_df = pd.read_csv(args.pred_csv)

    target_hours = int(pred_df["target_horizon"].max()) if "target_horizon" in pred_df.columns else 120
    print(f"   推断 target_hours: {target_hours}")

    # 根据 insured_time 确定使用的 Montel 文件
    # 业务规则：0点发布用 prefix 文件，12点发布用 postfix 文件
    if args.insured_time == 0:
        montel_csv = args.montel_prefix_csv
        montel_label = "00点发布"
    else:
        montel_csv = args.montel_postfix_csv
        montel_label = "12点发布"

    # 加载 Montel 预测
    print(f"\n2. 加载 Montel 预测文件 ({montel_label}): {montel_csv}")
    if montel_csv is None or not Path(montel_csv).exists():
        print(f"   警告: Montel 文件未提供或不存在，将跳过 Montel 对比")
        montel_df = pd.DataFrame(columns=["ds_utc", "issued_utc", "begin_utc", "target_horizon", "montel"])
    else:
        montel_df = load_montel_long(
            montel_csv=montel_csv,
            target_hours=target_hours,
            csv_sep=args.csv_sep,
            csv_encoding=args.csv_encoding,
        )
    print(f"   Montel 记录数: {len(montel_df)}")

    # 4. 加载真实电价
    print(f"\n4. 加载真实电价文件: {args.actual_csv}")
    actual_df = load_actual_price(
        actual_csv=args.actual_csv,
        date_col=args.date_col,
        target_col=args.target_col,
        csv_sep=args.csv_sep,
        csv_encoding=args.csv_encoding,
    )
    print(f"   真实电价记录数: {len(actual_df)}")

    exog_configs = json.loads(args.exog_configs)
    if exog_configs:
        from exogenous import ExogenousLoader
        exog_loader = ExogenousLoader(
            exog_configs,
            csv_sep=args.csv_sep,
            csv_encoding=args.csv_encoding,
        )
        exog_feats = exog_loader.load_features(actual_df["ds_utc"])
        for col in exog_loader.feat_cols:
            actual_df[col] = exog_feats[col].values
        print(f"   外部外生变量: {exog_loader.feat_cols}")

    # 5. 对齐数据
    print("\n5. 对齐预测数据与真实电价...")
    aligned_df = align_predictions_with_actual(pred_df, montel_df, actual_df, args.insured_time)

    # 6. 统计匹配情况
    montel_matched = int(aligned_df["montel"].notna().sum()) if "montel" in aligned_df.columns else 0
    actual_matched = int(aligned_df["y"].notna().sum())
    total = int(len(aligned_df))

    print(f"   总预测点数: {total}")
    print(f"   Montel 匹配点数: {montel_matched} ({montel_matched/total*100:.1f}%)")
    print(f"   真实电价匹配点数: {actual_matched} ({actual_matched/total*100:.1f}%)")

    if actual_matched == 0:
        print("警告：真实电价没有匹配到任何点，请检查时间对齐是否正确。")

    # 7. 识别模型列（包括 Montel）
    id_cols = {"unique_id", "ds", "ds_utc", "y", "horizon", "target_horizon",
               "target_horizon_x", "target_horizon_y",
               "forecast_day", "hour_in_forecast_day", "issued_date_local",
               "target_date_utc", "issued_local", "issued_utc", "cutoff_utc",
               "begin_local", "begin_utc", "target_end_utc",
               "issued_hour_local",
               "first_target_horizon", "last_target_horizon", "split"}

    model_cols = [c for c in aligned_df.columns
                  if c not in id_cols
                  and not c.startswith("feat_")
                  and pd.api.types.is_numeric_dtype(aligned_df[c])]

    # 确保 montel 列也参与评估（如果存在）
    eval_cols = list(model_cols)
    if "montel" in aligned_df.columns and "montel" not in eval_cols:
        eval_cols.append("montel")

    print(f"\n   识别到模型列: {eval_cols}")

    # 8. 计算指标
    has_montel = montel_matched > 0

    print("\n6. 计算指标（与真实电价比较）...")
    metrics_summary = summarize_metrics(aligned_df, eval_cols)
    metrics_by_day = summarize_by_forecast_day(aligned_df, eval_cols)
    metrics_by_date = summarize_by_target_date(aligned_df, eval_cols)
    metrics_by_issued = summarize_by_issued_hour(aligned_df, eval_cols)

    # 9. 保存结果
    aligned_df.to_csv(out_dir / "aligned_predictions.csv", index=False)
    metrics_summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    metrics_by_day.to_csv(out_dir / "metrics_by_forecast_day.csv", index=False)
    metrics_by_date.to_csv(out_dir / "metrics_by_target_date.csv", index=False)
    metrics_by_issued.to_csv(out_dir / "metrics_by_issued_hour.csv", index=False)

    # 10. 生成可视化
    print("\n7. 生成对比图表（按业务窗口）...")
    plot_comparison(aligned_df, model_cols, out_dir / "comparison_plot.png", args.insured_time)

    # 11. 打印结果
    print("\n========== 对比结果汇总（与真实电价比较）==========")
    print(metrics_summary.to_string(index=False))

    print("\n========== 按发布时间点对比 ==========")
    if not metrics_by_issued.empty:
        print(metrics_by_issued.to_string(index=False))
    else:
        print("无可用数据")

    print("\n========== 按预测天数对比 ==========")
    print(metrics_by_day.to_string(index=False))

    print("\n========== 输出文件 ==========")
    print(f"对齐后的预测数据: {out_dir / 'aligned_predictions.csv'}")
    print(f"指标汇总: {out_dir / 'metrics_summary.csv'}")
    print(f"按发布时间指标: {out_dir / 'metrics_by_issued_hour.csv'}")
    print(f"按天数指标: {out_dir / 'metrics_by_forecast_day.csv'}")
    print(f"按日期指标: {out_dir / 'metrics_by_target_date.csv'}")
    print(f"业务窗口对比图表: comparison_window_0.png ~ comparison_window_4.png")

    # _save_commits_preds_exog(aligned_df, model_cols, out_dir, args.pred_csv)


def _save_commits_preds_exog(aligned_df, model_cols, out_dir, pred_csv):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    commits_dir = Path("commits") / timestamp
    commits_dir.mkdir(parents=True, exist_ok=True)

    _plot_commits_comparison(aligned_df, model_cols, commits_dir)
    _extract_predictions_csv(pred_csv, commits_dir)

    print(f"\n========== Commits Deliverable ==========")
    for f in sorted(commits_dir.iterdir()):
        print(f"  {f.name}")
    print(f"Dir: {commits_dir.resolve()}")


def _extract_predictions_csv(pred_csv, commits_dir):
    df = pd.read_csv(pred_csv)

    c_col_name = df.columns[2]
    c_first = _safe_filename(df.iloc[0, 2])
    q_max = int(df.iloc[:, 16].max())
    out_name = f"{c_col_name}_{c_first}_{q_max}day.csv"

    out = pd.DataFrame({
        "unique_id": df.iloc[:, 0],
        "ds": df.iloc[:, 11],
        "ours": df.iloc[:, 20],
    })
    out.to_csv(commits_dir / out_name, index=False)
    print(f"\n  提取提交文件: {out_name}")


def _safe_filename(s: str) -> str:
    s = str(s)
    if "+" in s:
        s = s[: s.index("+")]
    return s.replace(":", "").replace(" ", "_")


def _plot_commits_comparison(df, model_cols, commits_dir):
    df = df.copy()
    if "issued_utc" not in df.columns:
        return

    df["ds_utc"] = pd.to_datetime(df["ds_utc"], utc=True)
    df["issued_utc"] = pd.to_datetime(df["issued_utc"], utc=True)

    common_issued = sorted(df["issued_utc"].dropna().unique())
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for issued in common_issued:
        part = df[df["issued_utc"] == issued].sort_values("ds_utc")
        if part.empty:
            continue

        issued_str = pd.Timestamp(issued).strftime("%Y-%m-%d %H:%M")
        safe = issued_str.replace(":", "").replace(" ", "_")

        fig, ax = plt.subplots(figsize=(16, 7))

        if "montel" in part.columns:
            m = part.dropna(subset=["montel"])
            if not m.empty:
                ax.plot(m["ds_utc"], m["montel"], marker=".", linewidth=1.5,
                        label="Montel", color="#e377c2", alpha=0.8)

        for i, col in enumerate(model_cols):
            if col not in part.columns:
                continue
            ax.plot(part["ds_utc"], part[col], marker=".", linewidth=1.5,
                    label=col, color=colors[i % len(colors)], alpha=0.8)

        ax.set_title(f"Forecast Comparison  |  Issued: {issued_str} UTC")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        all_times = part["ds_utc"].sort_values()
        if len(all_times) >= 3:
            ax.set_xticks([all_times.iloc[0], all_times.iloc[len(all_times)//2], all_times.iloc[-1]])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

        plt.tight_layout()
        plt.savefig(commits_dir / f"forecast_comparison_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  生成对比图: {commits_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="模型预测与 Montel 预测对比（与真实电价比较）")

    parser.add_argument("--pred_csv", type=str, required=True,
                        help="训练输出的预测结果文件（predictions_business.csv）")
    parser.add_argument("--montel_prefix_csv", type=str, default=None,
                        help="Montel 预测文件（00点发布，瑞典时间）")
    parser.add_argument("--montel_postfix_csv", type=str, default=None,
                        help="Montel 预测文件（12点发布，瑞典时间）")
    parser.add_argument("--actual_csv", type=str, required=True,
                        help="真实电价数据文件路径")
    parser.add_argument("--insured_time", type=int, choices=[0, 12], required=True,
                        help="发布时间点（只能是 0 或 12，表示 00:00 或 12:00 瑞典时间发布）")

    parser.add_argument("--date_col", type=str, default=None, help="日期列名（自动检测）")
    parser.add_argument("--target_col", type=str, default=None, help="价格列名（自动检测）")

    parser.add_argument("--out_dir", type=str, default="./outputs/comparison",
                        help="输出目录")

    parser.add_argument("--csv_sep", type=str, default=",", help="CSV 分隔符")
    parser.add_argument("--csv_encoding", type=str, default="utf-8", help="CSV 编码")

    parser.add_argument(
        "--exog_configs", type=str, default="[]",
        help="外部外生变量配置，JSON 格式"
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
