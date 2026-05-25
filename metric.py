# -*- coding: utf-8 -*-
"""
指标计算模块

该模块负责：
1. 定义常用的回归指标（MAE、RMSE、Bias、WAPE）
2. 业务窗口筛选（按发布时间点筛选预测）
3. 多维度指标汇总

发布时间点（瑞典时间）：
  - 00:00 发布 → begin 是次日 00:00（target_start_days=1）
  - 12:00 发布 → begin 是后日 00:00（target_start_days=2）
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ========== 时区工具函数 ==========
LOCAL_TZ = "Europe/Stockholm"


def is_dst(ts_utc: pd.Timestamp) -> bool:
    """判断某个 UTC 时间点是否处于夏令时（欧洲中部夏令时）"""
    local = ts_utc.tz_convert(LOCAL_TZ)
    offset_hours = local.utcoffset().total_seconds() / 3600.0
    return offset_hours > 1.0


def get_utc_offset(ts_utc: pd.Timestamp) -> float:
    """获取某个 UTC 时间点对应的本地时区偏移（小时）"""
    local = ts_utc.tz_convert(LOCAL_TZ)
    return local.utcoffset().total_seconds() / 3600.0


# 标识列集合（非模型预测列）
ID_COLS = {
    "unique_id",       # 时间序列标识
    "ds",              # 预测时间点（无时区）
    "cutoff",          # 预测截断点（无时区）
    "ds_utc",          # 预测时间点（UTC）
    "cutoff_utc",      # 预测截断点（UTC）
    "issued_utc",      # 发布时间（UTC）
    "issued_local",    # 发布时间（本地时间）
    "begin_utc",       # 目标窗口开始时间（UTC）
    "begin_local",     # 目标窗口开始时间（本地时间）
    "target_end_utc",  # 目标窗口结束时间（UTC）
    "target_end_local",# 目标窗口结束时间（本地时间）
    "y",               # 真实值
    "split",           # 数据集划分（train/val/test）
    "horizon",         # 预测步长（从 cutoff 开始）
    "target_horizon",  # 目标步长（从 begin 开始）
    "forecast_day",    # 预测天数（第1天、第2天...）
    "hour_in_forecast_day",  # 预测天内小时位置
    "issued_date",     # 发布日期（字符串）
    "target_date",     # 目标日期（字符串）
    "issued_hour_local",  # 发布时间小时（本地）
}


# ========== 指标计算函数 ==========
def mae(y, yhat) -> float:
    """计算平均绝对误差（Mean Absolute Error）"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs(y[mask] - yhat[mask]))) if mask.any() else np.nan


def rmse(y, yhat) -> float:
    """计算均方根误差（Root Mean Squared Error）"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(np.mean((y[mask] - yhat[mask]) ** 2))) if mask.any() else np.nan


def bias(y, yhat) -> float:
    """计算偏差（Bias）"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(yhat[mask] - y[mask])) if mask.any() else np.nan


def wape(y, yhat, eps: float = 1e-8) -> float:
    """计算加权绝对百分比误差（Weighted Absolute Percentage Error）"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not mask.any():
        return np.nan
    denom = float(np.sum(np.abs(y[mask])))
    return float(np.sum(np.abs(y[mask] - yhat[mask])) / max(denom, eps))


def infer_model_cols(df: pd.DataFrame) -> List[str]:
    """从 DataFrame 中推断模型预测列"""
    return [
        c for c in df.columns
        if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


# ========== 业务口径配置函数 ==========
def business_horizon_config(
    issued_hour_local: int,
    target_start_days: int,
    target_hours: int,
) -> dict:
    """
    计算单个发布时间点的业务预测窗口配置。

    参数：
        issued_hour_local: 本地发布时间小时（0 或 12）
        target_start_days: 目标开始天数（1=次日0点，2=后日0点）
        target_hours: 目标预测小时数

    返回：
        包含 first_target_horizon, last_target_horizon, model_h_required 的字典
    """
    if issued_hour_local not in [0, 12]:
        raise ValueError("issued_hour_local 必须是 0 或 12")

    # cutoff = issued_local - 1h
    # first_target_horizon = 从 cutoff 到第一个目标时刻的小时数
    # 例如：issued_local=00:00, target_start_days=1
    #       cutoff = 前一天 23:00, begin = 次日 00:00
    #       first_target_horizon = 24 - 0 + 1 = 25
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


def business_horizon_configs(
    issued_hours_local: List[int],
    target_start_days_list: List[int],
    target_hours: int,
) -> Dict[int, dict]:
    """
    计算多个发布时间点的业务预测窗口配置。

    业务规则：
    - 00:00 瑞典时间发布 → begin 是次日 00:00（target_start_days=1）
    - 12:00 瑞典时间发布 → begin 是后日 00:00（target_start_days=2）

    参数：
        issued_hours_local: 本地发布时间小时列表（如 [0, 12]）
        target_start_days_list: 对应每个发布时间的目标开始天数（如 [1, 2]）
        target_hours: 目标预测小时数

    返回：
        字典，key 是 issued_hour_local，value 是对应的配置
    """
    if len(issued_hours_local) != len(target_start_days_list):
        raise ValueError("issued_hours_local 和 target_start_days_list 长度必须一致")

    configs = {}
    for hour, days in zip(issued_hours_local, target_start_days_list):
        configs[hour] = business_horizon_config(hour, days, target_hours)

    return configs


# ========== 筛选业务窗口函数 ==========
def _to_utc_series(s: pd.Series) -> pd.Series:
    """将时间列转换为 UTC 时区"""
    return pd.to_datetime(s, utc=True)


def filter_business_window(
    cv_df: pd.DataFrame,
    *,
    split: str,
    issued_tz: str = LOCAL_TZ,
    issued_hours_local: List[int] = None,
    target_start_days_dict: Dict[int, int] = None,
    target_hours: int = 120,
    split_start_utc=None,
    split_end_utc=None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """
    从 NeuralForecast cross_validation 结果中筛选业务窗口

    支持多个发布时间点：
    - 00:00 瑞典时间发布 → begin 是次日 00:00（target_start_days=1）
    - 12:00 瑞典时间发布 → begin 是后日 00:00（target_start_days=2）

    参数：
        cv_df: cross_validation 原始结果
        split: 数据集划分名称（val/test）
        issued_tz: 发布时区
        issued_hours_local: 本地发布小时列表（如 [0, 12]）
        target_start_days_dict: 发布时间到目标开始天数的映射（如 {0: 1, 12: 2}）
        target_hours: 目标小时数
        split_start_utc: 划分开始时间
        split_end_utc: 划分结束时间
        require_complete: 是否要求完整的 target_hours 预测

    返回：
        筛选后的业务窗口 DataFrame
    """
    if issued_hours_local is None:
        issued_hours_local = [0, 12]
    if target_start_days_dict is None:
        target_start_days_dict = {0: 1, 12: 2}

    if cv_df.empty:
        return cv_df.copy()

    out = cv_df.copy()

    # 将无时区时间转换为 UTC
    out["ds_utc"] = _to_utc_series(out["ds"])
    out["cutoff_utc"] = _to_utc_series(out["cutoff"])

    # 计算发布时间（cutoff + 1小时）
    out["issued_utc"] = out["cutoff_utc"] + pd.Timedelta(hours=1)

    # 转换为本地时间
    out["issued_local"] = out["issued_utc"].dt.tz_convert(issued_tz)

    # 筛选指定发布时间点的预测
    out = out[out["issued_local"].dt.hour.isin(issued_hours_local)].copy()
    if out.empty:
        return out

    # 为每个发布时间点计算目标窗口
    begin_list = []
    for idx, row in out.iterrows():
        issued_local = row["issued_local"]
        issued_hour = issued_local.hour
        target_start_days = target_start_days_dict.get(issued_hour, 1)
        # begin_local = issued_local 次日 00:00 + target_start_days 天
        begin_local = issued_local.normalize() + pd.DateOffset(days=target_start_days)
        begin_list.append(begin_local)

    out["begin_local"] = pd.Series(begin_list, index=out.index)
    out["begin_utc"] = out["begin_local"].dt.tz_convert("UTC")

    # 计算目标窗口结束时间
    out["target_end_utc"] = out["begin_utc"] + pd.Timedelta(hours=target_hours - 1)
    out["target_end_local"] = out["target_end_utc"].dt.tz_convert(issued_tz)

    # 筛选目标窗口内的预测
    in_target = (out["ds_utc"] >= out["begin_utc"]) & (out["ds_utc"] <= out["target_end_utc"])
    out = out[in_target].copy()
    if out.empty:
        return out

    # 按 split 时间范围过滤
    if split_start_utc is not None:
        start = pd.Timestamp(split_start_utc)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        out = out[out["ds_utc"] >= start].copy()

    if split_end_utc is not None:
        end = pd.Timestamp(split_end_utc)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        out = out[out["ds_utc"] <= end].copy()

    if out.empty:
        return out

    # 计算预测步长
    out["horizon"] = ((out["ds_utc"] - out["cutoff_utc"]).dt.total_seconds() // 3600).astype(int)
    out["target_horizon"] = ((out["ds_utc"] - out["begin_utc"]).dt.total_seconds() // 3600 + 1).astype(int)

    # 获取每个发布时间对应的 first_target_horizon
    out["first_target_horizon"] = out["issued_local"].dt.hour.map(
        lambda h: business_horizon_config(
            h, target_start_days_dict.get(h, 1), target_hours
        )["first_target_horizon"]
    )

    # 调整 target_horizon：加上 first_target_horizon - 1
    out["target_horizon"] = out["target_horizon"] + out["first_target_horizon"] - 1

    # 确保目标步长在有效范围内
    out["last_target_horizon"] = out["issued_local"].dt.hour.map(
        lambda h: business_horizon_config(
            h, target_start_days_dict.get(h, 1), target_hours
        )["last_target_horizon"]
    )
    valid_mask = (out["target_horizon"] >= 1) & (out["target_horizon"] <= out["last_target_horizon"])
    out = out[valid_mask].copy()

    # 如果要求完整窗口，过滤掉不完整的发布时间
    if require_complete:
        n_by_issued = out.groupby("issued_utc")["target_horizon"].nunique()
        complete_issued = n_by_issued[n_by_issued == target_hours].index
        out = out[out["issued_utc"].isin(complete_issued)].copy()

    if out.empty:
        return out

    # 添加额外标识列
    out["split"] = split
    out["issued_hour_local"] = out["issued_local"].dt.hour
    out["forecast_day"] = ((out["target_horizon"] - out["first_target_horizon"]) // 24 + 1).astype(int)
    out["hour_in_forecast_day"] = ((out["target_horizon"] - out["first_target_horizon"]) % 24).astype(int)
    out["issued_date"] = out["issued_local"].dt.date.astype(str)
    out["target_date"] = out["ds_utc"].dt.tz_convert(issued_tz).dt.date.astype(str)

    # 重新排列列顺序
    id_cols = [
        "unique_id", "split", "issued_utc", "issued_local", "issued_hour_local",
        "cutoff_utc", "begin_utc", "begin_local", "target_end_utc", "target_end_local",
        "ds_utc", "ds", "horizon", "target_horizon", "first_target_horizon", "last_target_horizon",
        "forecast_day", "hour_in_forecast_day", "issued_date", "target_date", "y",
    ]
    existing_id_cols = [c for c in id_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in existing_id_cols]

    return out[existing_id_cols + other_cols].reset_index(drop=True)


# ========== 向后兼容函数 ==========
def business_horizon_config_local(
    issued_hour_local: int = 11,
    target_start_days: int = 1,
    target_hours: int = 120,
) -> dict:
    """向后兼容的单一发布时间点配置函数"""
    return business_horizon_config(issued_hour_local, target_start_days, target_hours)


def filter_business_window_local(
    cv_df: pd.DataFrame,
    *,
    split: str,
    issued_tz: str = LOCAL_TZ,
    issued_hour_local: int = 11,
    target_start_days: int = 1,
    target_hours: int = 120,
    split_start_utc=None,
    split_end_utc=None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """向后兼容的单一发布时间点筛选函数"""
    return filter_business_window(
        cv_df,
        split=split,
        issued_tz=issued_tz,
        issued_hours_local=[issued_hour_local],
        target_start_days_dict={issued_hour_local: target_start_days},
        target_hours=target_hours,
        split_start_utc=split_start_utc,
        split_end_utc=split_end_utc,
        require_complete=require_complete,
    )


# ========== 汇总函数 ==========
def _summarize_group(df: pd.DataFrame, model_cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    """通用分组汇总函数"""
    rows = []
    if df.empty:
        return pd.DataFrame(rows)

    group_obj = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    for key, part in group_obj:
        if not isinstance(key, tuple):
            key = (key,)
        prefix = dict(zip(group_cols, key))

        for model in model_cols:
            rows.append({
                **prefix,
                "model": model,
                "count": int(np.isfinite(part["y"]).sum()),
                "mae": mae(part["y"], part[model]),
                "rmse": rmse(part["y"], part[model]),
                "bias": bias(part["y"], part[model]),
                "wape": wape(part["y"], part[model]),
            })

    return pd.DataFrame(rows)


def summarize_overall(df: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    """按数据集划分（split）汇总指标"""
    return _summarize_group(df, model_cols, ["split"])


def summarize_by_issued_day(df: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    """按发布日期汇总指标"""
    return _summarize_group(df, model_cols, ["split", "issued_date"])


def summarize_by_target_date(df: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    """按目标日期汇总指标"""
    return _summarize_group(df, model_cols, ["split", "target_date"])


def summarize_by_forecast_day(df: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    """按预测天数汇总指标（第1天、第2天...）"""
    return _summarize_group(df, model_cols, ["split", "forecast_day"])


def summarize_by_horizon(df: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    """按预测步长汇总指标"""
    return _summarize_group(df, model_cols, ["split", "target_horizon"])


def summarize_by_issued_hour(df: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    """按发布时间点（00:00 或 12:00）汇总指标"""
    return _summarize_group(df, model_cols, ["split", "issued_hour_local"])
