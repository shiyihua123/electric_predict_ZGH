# -*- coding: utf-8 -*-
"""
电价预测脚本 — 基于训练好的模型预测单个业务窗口

该脚本负责：
1. 加载已训练的 NeuralForecast 模型
2. 从 CSV 数据文件读取最新数据并生成时间特征
3. 执行预测
4. 按业务口径筛选预测窗口
5. 输出 CSV 结果

发布时间点（瑞典时间）：
  - 00:00 发布 → begin 是次日 00:00（target_start_days=1）
  - 12:00 发布 → begin 是后日 00:00（target_start_days=2）

使用方法：
    python predict.py \
        --data_path ./sourceData/.../data.csv \
        --model_dir ./outputs/nhits_tcn_patchtst/neuralforecast_bundle \
        --insured_time 0 \
        --target_hours 120 \
        --out_csv ./predictions_business.csv
"""

import argparse
import json
from pathlib import Path
import warnings

import pandas as pd

warnings.filterwarnings("ignore", ".*isinstance\\(treespec, LeafSpec\\).*")

from neuralforecast import NeuralForecast

from dataset import PriceDatasetBuilder
from metric import (
    business_horizon_config_local,
    filter_business_window_local,
    infer_model_cols,
)


def build_parser():
    parser = argparse.ArgumentParser(description="电价预测脚本 — 基于已训练模型预测单个业务窗口")

    parser.add_argument(
        "--data_path", type=str, required=True,
        help="CSV 数据文件路径"
    )
    parser.add_argument("--date_col", type=str, default="date", help="时间列名")
    parser.add_argument("--target_col", type=str, default="price", help="目标价格列名")
    parser.add_argument("--csv_sep", type=str, default=",", help="CSV 分隔符")
    parser.add_argument("--csv_encoding", type=str, default="utf-8", help="CSV 编码")
    parser.add_argument("--freq", type=str, default="h", help="时间频率")
    parser.add_argument("--unique_id", type=str, default="SE2", help="时间序列唯一标识")
    parser.add_argument(
        "--missing_strategy", type=str, default="interpolate",
        choices=["interpolate", "ffill", "raise"],
        help="缺失值处理策略"
    )

    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="已训练模型的 neuralforecast_bundle 目录路径"
    )

    parser.add_argument(
        "--issued_tz", type=str, default="Europe/Stockholm",
        help="发布时间所在时区"
    )
    parser.add_argument(
        "--issued_date", type=str, default=None,
        help="发布日期（瑞典时区），格式 YYYY-MM-DD，不指定则自动取数据中最新的匹配日期"
    )
    parser.add_argument(
        "--insured_time", type=int, default=0, choices=[0, 12],
        help="发布时间点（0 表示 00:00 发布，12 表示 12:00 发布）"
    )
    parser.add_argument(
        "--target_hours", type=int, default=120,
        help="目标预测小时数（默认 120 = 5 天）"
    )

    parser.add_argument(
        "--out_csv", type=str, default="./predictions_business.csv",
        help="输出 CSV 文件路径"
    )

    parser.add_argument(
        "--exog_configs", type=str, default="[]",
        help="外部外生变量配置，JSON 格式，如 '[{\"csv_path\":\"...\",\"name\":\"SE3\"}]'"
    )

    return parser


def find_cutoff_for_business_window(df, issued_tz, insured_time, issued_date=None):
    if issued_date is not None:
        target_local = pd.Timestamp(f"{issued_date} {insured_time:02d}:00:00").tz_localize(issued_tz)
        target_utc = target_local.tz_convert("UTC")
        cutoff_utc = target_utc - pd.Timedelta(hours=1)
        cutoff_naive = cutoff_utc.tz_localize(None)

        if cutoff_naive in df["ds"].values:
            return cutoff_naive

        raise ValueError(
            f"数据中不存在 cutoff ({cutoff_naive})，"
            f"发布时间: {issued_date} {insured_time:02d}:00 {issued_tz}"
        )

    ds_sorted = df["ds"].sort_values(ascending=False)

    for ds_naive in ds_sorted:
        cutoff_utc = pd.Timestamp(ds_naive).tz_localize("UTC")
        issued_utc = cutoff_utc + pd.Timedelta(hours=1)
        issued_local = issued_utc.tz_convert(issued_tz)
        if issued_local.hour == insured_time:
            return ds_naive

    raise ValueError(
        f"数据中找不到合适的 cutoff，使得 ({issued_tz} 时间发布的小时数 = {insured_time}。"
    )


def build_futr_df(df, cutoff_naive, builder, model_h):
    future_ds_naive = pd.date_range(
        start=pd.Timestamp(cutoff_naive) + pd.Timedelta(hours=1),
        periods=model_h,
        freq=builder.freq,
    )

    future_ds_utc = future_ds_naive.tz_localize("UTC")
    future_feat = pd.DataFrame({"ds_utc": future_ds_utc})
    future_feat = builder._add_time_features(future_feat)

    futr_exog_cols = [c for c in future_feat.columns if c.startswith("feat_")]

    futr_df = pd.DataFrame({
        "unique_id": builder.unique_id,
        "ds": future_ds_naive,
    })
    for col in futr_exog_cols:
        futr_df[col] = future_feat[col].values

    return futr_df


def main():
    args = build_parser().parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    exog_configs = json.loads(args.exog_configs)

    builder = PriceDatasetBuilder(
        excel_path=args.data_path,
        date_col=args.date_col,
        target_col=args.target_col,
        freq=args.freq,
        unique_id=args.unique_id,
        missing_strategy=args.missing_strategy,
        csv_sep=args.csv_sep,
        csv_encoding=args.csv_encoding,
        train_start="2000-01-01 00:00:00+00:00",
        test_end="2099-12-31 23:00:00+00:00",
        exog_configs=exog_configs,
    )

    data = builder.load(use_exog=True)
    full_df = data.full_df.copy()
    print(f"数据加载完成，共 {len(full_df)} 行")

    exog_loader = builder.get_exog_loader()
    if exog_loader is not None:
        print(f"外部外生变量: {exog_loader.feat_cols}")

    nf = NeuralForecast.load(path=str(model_dir))
    print(f"模型加载完成，共 {len(nf.models)} 个模型")

    model_h = None
    for model in nf.models:
        model_h = model.h
        break
    if model_h is None:
        raise ValueError("无法获取模型的 horizon")

    target_start_days = 1 if args.insured_time == 0 else 2
    config = business_horizon_config_local(
        args.insured_time, target_start_days, args.target_hours
    )
    print(f"发布时间点: {args.insured_time:02d}:00 {args.issued_tz}")
    if args.issued_date:
        print(f"发布日期: {args.issued_date}")
    print(f"目标开始天数: {target_start_days}")
    print(f"目标预测小时数: {args.target_hours}")
    print(f"模型 horizon: {model_h}")

    cutoff_naive = find_cutoff_for_business_window(
        full_df, args.issued_tz, args.insured_time, args.issued_date
    )
    print(f"业务窗口 cutoff: {cutoff_naive} (本地发布小时={args.insured_time})")

    input_df = full_df[full_df["ds"] <= cutoff_naive].copy()

    futr_df = build_futr_df(full_df, cutoff_naive, builder, model_h)

    preds = nf.predict(df=input_df, futr_df=futr_df)

    preds["cutoff"] = cutoff_naive

    business_preds = filter_business_window_local(
        preds,
        split="predict",
        issued_tz=args.issued_tz,
        issued_hour_local=args.insured_time,
        target_start_days=target_start_days,
        target_hours=args.target_hours,
        require_complete=True,
    )

    model_cols = infer_model_cols(business_preds)

    if len(model_cols) >= 2:
        business_preds["Ensemble"] = business_preds[model_cols].mean(axis=1)
        model_cols.append("Ensemble")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    business_preds.to_csv(out_path, index=False)

    print(f"预测完成！")
    print(f"业务窗口行数: {len(business_preds)}")
    print(f"模型预测列: {model_cols}")
    print(f"结果已保存到: {out_path}")

    return business_preds


if __name__ == "__main__":
    main()
