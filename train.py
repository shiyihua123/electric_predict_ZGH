# -*- coding: utf-8 -*-
"""
主训练脚本

该脚本负责：
1. 解析命令行参数
2. 配置 PyTorch 运行环境
3. 加载和预处理数据
4. 构建并训练 NeuralForecast 模型
5. 执行交叉验证
6. 计算和保存指标

发布时间点（瑞典时间）：
  - 00:00 发布 → begin 是次日 00:00（target_start_days=1）
  - 12:00 发布 → begin 是后日 00:00（target_start_days=2）
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore", ".*isinstance\\(treespec, LeafSpec\\).*")

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, HuberLoss
from neuralforecast.models import NHITS, PatchTST, TCN, NBEATSx, TiDE

from dataset import PriceDatasetBuilder
from metric import (
    business_horizon_config,
    business_horizon_configs,
    filter_business_window,
    infer_model_cols,
    summarize_by_forecast_day,
    summarize_by_horizon,
    summarize_by_issued_day,
    summarize_by_issued_hour,
    summarize_by_target_date,
    summarize_overall,
)

def set_seed(seed: int) -> None:
    """设置随机种子，确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(args) -> None:
    """配置 PyTorch 运行环境"""
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if not torch.cuda.is_available():
        return

    if args.matmul_precision != "highest":
        torch.set_float32_matmul_precision(args.matmul_precision)

    if not args.warn_cuda_grad_stream_mismatch:
        fn = getattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch", None)
        if callable(fn):
            fn(False)


def parse_models(models: str) -> List[str]:
    """解析模型列表字符串"""
    return [m.strip().upper() for m in models.split(",") if m.strip()]


def _parse_devices(devices: str | None):
    """解析设备参数"""
    if devices is None or str(devices).lower() == "auto":
        return None
    s = str(devices).strip()
    if s.isdigit():
        return int(s)
    return s


def _trainer_kwargs(args) -> dict:
    """构建 PyTorch Lightning Trainer 参数"""
    kwargs = {}
    devices = _parse_devices(args.devices)
    if devices is not None:
        kwargs["devices"] = devices

    if args.accelerator != "auto":
        kwargs["accelerator"] = args.accelerator
    elif torch.cuda.is_available():
        kwargs["accelerator"] = "gpu"
    else:
        kwargs["accelerator"] = "cpu"

    if args.strategy != "auto":
        kwargs["strategy"] = args.strategy

    return kwargs


def build_models(args, futr_exog_cols: List[str], hist_exog_cols: List[str], model_h: int):
    """构建 NeuralForecast 模型列表"""
    input_size = args.input_days * 24

    if args.loss_type == "huber":
        train_loss = HuberLoss(delta=args.huber_delta)
        val_loss = HuberLoss(delta=args.huber_delta)
    else:
        train_loss = MAE()
        val_loss = MAE()

    common = dict(
        h=model_h,
        input_size=input_size,
        loss=train_loss,
        valid_loss=val_loss,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        val_check_steps=args.val_check_steps,
        early_stop_patience_steps=args.early_stop_patience_steps,
        batch_size=args.batch_size,
        windows_batch_size=args.windows_batch_size,
        scaler_type=args.nf_scaler_type,
        random_seed=args.seed,
        **_trainer_kwargs(args),
    )

    common_with_exog = common.copy()
    if args.use_exog and futr_exog_cols:
        common_with_exog["futr_exog_list"] = futr_exog_cols
    if args.use_exog and hist_exog_cols:
        common_with_exog["hist_exog_list"] = hist_exog_cols

    wanted = parse_models(args.models)
    models = []

    if "NHITS" in wanted:
        models.append(NHITS(
            **common_with_exog,
            n_blocks=[1, 1, 1, 1],
            n_pool_kernel_size=[8, 4, 2, 1],
            dropout_prob_theta=0.2,
        ))

    if "TCN" in wanted:
        models.append(TCN(**common_with_exog))

    if "NBEATSX" in wanted:
        models.append(NBEATSx(**common_with_exog))
    
    if "PATCHTST" in wanted:
        models.append(PatchTST(**common))

    if "TIDE" in wanted:
        models.append(TiDE(
            **common_with_exog,
            dropout=0.5,
            num_encoder_layers=2,
            num_decoder_layers=2,
            decoder_output_dim=128,
            temporal_width=8,
        ))

    if not models:
        raise ValueError("没有可训练模型。可选：NHITS,TCN,NBEATSX,PatchTST,TiDE")

    return models


def save_json(obj, path: Path) -> None:
    """保存对象为 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """转换为无时区标记的 UTC 时间"""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def count_rows_between(df: pd.DataFrame, start_utc, end_utc) -> int:
    """计算时间范围内的行数"""
    start = to_naive_utc(pd.Timestamp(start_utc))
    end = to_naive_utc(pd.Timestamp(end_utc))
    return int(((df["ds"] >= start) & (df["ds"] <= end)).sum())


def truncate_df_until(df: pd.DataFrame, end_utc) -> pd.DataFrame:
    """截断 DataFrame 到指定时间"""
    end = to_naive_utc(pd.Timestamp(end_utc))
    return df[df["ds"] <= end].copy()


def adjusted_test_size(raw_size: int, h: int, step_size: int, split_name: str) -> int:
    """调整测试集大小以匹配模型 horizon"""
    if step_size <= 0:
        raise ValueError("eval_step_size 必须为正整数。")
    if raw_size < h:
        raise ValueError(
            f"{split_name} 可评测小时数 {raw_size} 小于模型 horizon {h}。"
            "请扩大该 split 的时间范围，或减小 target_hours / target_start_days。"
        )
    return h + ((raw_size - h) // step_size) * step_size


def run_cv_for_split(
    *,
    split_name: str,
    df: pd.DataFrame,
    eval_start_utc: pd.Timestamp,
    eval_end_utc: pd.Timestamp,
    val_size: int,
    args,
    futr_exog_cols: List[str],
    hist_exog_cols: List[str],
    model_h: int,
    out_dir: Path,
    save_model: bool = False,
) -> Tuple[pd.DataFrame, NeuralForecast]:
    """
    对一个数据集划分执行交叉验证

    参数：
        split_name: 划分名称（val/test）
        df: 完整数据集
        eval_start_utc: 评估开始时间
        eval_end_utc: 评估结束时间
        val_size: 验证集大小（用于早停）
        args: 命令行参数
        futr_exog_cols: 外生变量列
        model_h: 模型 horizon
        out_dir: 输出目录
        save_model: 是否保存模型

    返回：
        (业务窗口预测结果, NeuralForecast 对象)
    """
    df_cv = truncate_df_until(df, eval_end_utc)

    raw_size = count_rows_between(df_cv, eval_start_utc, eval_end_utc)

    test_size = adjusted_test_size(raw_size, model_h, args.eval_step_size, split_name)

    print(f"\n========== {split_name.upper()} CV ==========")
    print(f"评估开始时间       : {eval_start_utc}")
    print(f"评估结束时间       : {eval_end_utc}")
    print(f"原始评估小时数     : {raw_size}")
    print(f"模型预测步长       : {model_h}")
    print(f"滚动步长           : {args.eval_step_size}")
    print(f"调整后测试集大小   : {test_size}")
    print(f"舍弃的开头小时数   : {raw_size - test_size}")
    print(f"训练用验证集大小   : {val_size}")

    nf = NeuralForecast(
        models=build_models(args, futr_exog_cols, hist_exog_cols, model_h),
        freq=args.freq,
    )

    cv_raw = nf.cross_validation(
        df=df_cv,
        val_size=val_size,
        test_size=test_size,
        step_size=args.eval_step_size,
        n_windows=None,
    )

    cv_raw = cv_raw.reset_index(drop=True)
    # cv_raw.to_csv(out_dir / f"cv_predictions_raw_{split_name}.csv", index=False)

    # 构建发布时间到目标开始天数的映射
    target_start_days_dict = dict(zip(args.issued_hours_local, args.target_start_days_list))

    # 筛选业务窗口（支持多个发布时间点）
    cv_business = filter_business_window(
        cv_raw,
        split=split_name,
        issued_tz=args.issued_tz,
        issued_hours_local=args.issued_hours_local,
        target_start_days_dict=target_start_days_dict,
        target_hours=args.target_hours,
        split_start_utc=eval_start_utc,
        split_end_utc=eval_end_utc,
        require_complete=True,
    )

    if cv_business.empty:
        msg = (
            f"{split_name} 业务窗口筛选后为空。\n"
            f"当前业务口径：\n"
        )
        for hour, days in target_start_days_dict.items():
            msg += f"  - {args.issued_tz} {hour:02d}:00 发布 → 目标开始天数={days}\n"
        msg += (
            f"目标小时数: {args.target_hours}\n"
            "建议：1) 保持 --eval_step_size 1；2) 确认 split_end 覆盖完整目标小时数。"
        )
        raise ValueError(msg)

    print(f"业务窗口行数       : {len(cv_business)}")
    print(f"业务窗口发布天数   : {cv_business['issued_utc'].nunique()}")
    print(f"发布时间点分布     : {cv_business['issued_hour_local'].value_counts().to_dict()}")

    if save_model:
        model_bundle_dir = out_dir / "neuralforecast_bundle"
        nf.save(str(model_bundle_dir), overwrite=True)
        print(f"NeuralForecast 模型已保存到: {model_bundle_dir}")

    return cv_business, nf


def main(args) -> None:
    """主函数：执行完整的训练和评估流程"""
    if args.target_hours is None:
        args.target_hours = args.pred_days * 24

    set_seed(args.seed)

    configure_torch_runtime(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 根据 insured_time 自动推算发布时间和目标开始天数
    # 业务规则：0点发布预测次日0点开始，12点发布预测后天0点开始
    issued_hours_local = [args.insured_time]
    target_start_days_list = [1 if args.insured_time == 0 else 2]
    args.issued_hours_local = issued_hours_local
    args.target_start_days_list = target_start_days_list

    # 计算业务预测窗口配置
    configs = business_horizon_configs(
        issued_hours_local=issued_hours_local,
        target_start_days_list=target_start_days_list,
        target_hours=args.target_hours,
    )

    # 使用最大的 model_h_required 作为模型 horizon
    model_h = max(cfg["model_h_required"] for cfg in configs.values())

    print("========== 业务口径配置 ==========")
    for hour, cfg in configs.items():
        print(f"发布时间 {hour:02d}:00 瑞典时间:")
        print(f"  目标开始天数 : {cfg['target_start_days']}")
        print(f"  目标小时数   : {cfg['target_hours']}")
        print(f"  first_target_horizon: {cfg['first_target_horizon']}")
        print(f"  last_target_horizon: {cfg['last_target_horizon']}")
        print(f"  model_h_required: {cfg['model_h_required']}")
    print(f"模型 horizon : {model_h}")

    builder = PriceDatasetBuilder(
        excel_path=args.excel_path,
        date_col=args.date_col,
        target_col=args.target_col,
        sheet_name=args.sheet_name,
        unique_id=args.unique_id,
        local_tz=args.local_tz,
        freq=args.freq,
        missing_strategy=args.missing_strategy,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
        csv_sep=args.csv_sep,
        csv_encoding=args.csv_encoding,
        exog_configs=json.loads(args.exog_configs),
    )

    data = builder.load(use_exog=args.use_exog)

    bounds = builder.split_boundaries_utc()

    print("\n========== 数据摘要 ==========")
    print(f"完整数据行数 : {len(data.full_df)}")
    print(f"训练集行数   : {len(data.train_df)}")
    print(f"验证集行数   : {len(data.val_df)}")
    print(f"测试集行数   : {len(data.test_df)}")
    print(f"输入时间步数 : {args.input_days * 24}")
    print(f"发布时区     : {args.issued_tz}")
    print(f"发布时间列表 : {args.issued_hours_local}")
    print(f"目标开始天数 : {args.target_start_days_list}")
    print(f"目标小时数   : {args.target_hours}")
    print(f"模型 horizon : {model_h}")
    print(f"使用外生变量 : {args.use_exog}")
    print(f"未来外生变量列 : {data.futr_exog_cols}")
    print(f"历史外生变量列 : {data.hist_exog_cols}")
    print(f"GPU 索引     : {args.gpu_ids or os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    print(f"加速器类型   : {args.accelerator}")
    print(f"设备数量     : {args.devices}")

    data.full_df.to_csv(out_dir / "prepared_full_df.csv", index=False)
    data.train_df.to_csv(out_dir / "prepared_train_df.csv", index=False)
    data.val_df.to_csv(out_dir / "prepared_val_df.csv", index=False)
    data.test_df.to_csv(out_dir / "prepared_test_df.csv", index=False)

    all_business = []

    if not args.skip_val_cv:
        internal_val_size = min(args.internal_val_days * 24, max(24, data.train_size // 5))
        max_internal = max(24, data.train_size - args.input_days * 24 - model_h - 1)
        internal_val_size = min(internal_val_size, max_internal)

        val_business, _ = run_cv_for_split(
            split_name="val",
            df=data.full_df,
            eval_start_utc=bounds["val_start"],
            eval_end_utc=bounds["val_end"],
            val_size=internal_val_size,
            args=args,
            futr_exog_cols=data.futr_exog_cols,
            hist_exog_cols=data.hist_exog_cols,
            model_h=model_h,
            out_dir=out_dir,
            save_model=False,
        )
        all_business.append(val_business)

    test_business, _ = run_cv_for_split(
        split_name="test",
        df=data.full_df,
        eval_start_utc=bounds["test_start"],
        eval_end_utc=bounds["test_end"],
        val_size=data.val_size,
        args=args,
        futr_exog_cols=data.futr_exog_cols,
        hist_exog_cols=data.hist_exog_cols,
        model_h=model_h,
        out_dir=out_dir,
        save_model=True,
    )
    all_business.append(test_business)

    pred_df = pd.concat(all_business, ignore_index=True)

    model_cols_raw = infer_model_cols(pred_df)
    if len(model_cols_raw) >= 2:
        pred_df["Ensemble"] = pred_df[model_cols_raw].mean(axis=1)

    pred_df.to_csv(out_dir / "predictions_business.csv", index=False)

    model_cols = infer_model_cols(pred_df)

    metrics_summary = summarize_overall(pred_df, model_cols)
    metrics_by_issued_day = summarize_by_issued_day(pred_df, model_cols)
    metrics_by_target_date = summarize_by_target_date(pred_df, model_cols)
    metrics_by_forecast_day = summarize_by_forecast_day(pred_df, model_cols)
    metrics_by_horizon = summarize_by_horizon(pred_df, model_cols)
    metrics_by_issued_hour = summarize_by_issued_hour(pred_df, model_cols)

    metrics_summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    metrics_by_issued_day.to_csv(out_dir / "metrics_by_issued_day.csv", index=False)
    metrics_by_target_date.to_csv(out_dir / "metrics_by_target_date.csv", index=False)
    metrics_by_forecast_day.to_csv(out_dir / "metrics_by_forecast_day.csv", index=False)
    metrics_by_horizon.to_csv(out_dir / "metrics_by_horizon.csv", index=False)
    metrics_by_issued_hour.to_csv(out_dir / "metrics_by_issued_hour.csv", index=False)

    config = vars(args).copy()
    config["business_configs"] = {k: v for k, v in configs.items()}
    config["model_cols"] = model_cols
    save_json(config, out_dir / "config.json")

    print("\n========== 指标摘要 ==========")
    print(metrics_summary)
    print("\n========== 按发布时间点指标 ==========")
    print(metrics_by_issued_hour)
    print("\n========== 按预测天数指标 ==========")
    print(metrics_by_forecast_day)
    print("\n========== 输出文件 ==========")
    print(out_dir / "predictions_business.csv")
    print(out_dir / "metrics_summary.csv")
    print(out_dir / "metrics_by_issued_hour.csv")
    print(out_dir / "metrics_by_issued_day.csv")
    print(out_dir / "metrics_by_target_date.csv")
    print(out_dir / "metrics_by_forecast_day.csv")
    print(out_dir / "metrics_by_horizon.csv")
    print(f"\n完成！结果已保存到: {out_dir}")


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="电价预测模型训练脚本")

    # ========== 数据参数 ==========
    parser.add_argument(
        "--excel_path",
        type=str,
        default="sourceData/SE2_Price_Spot_EUR_MWh_NordPool_15min_Actual/actual_min_to_H_true_latest.csv",
        help="CSV / Excel 数据文件路径"
    )
    parser.add_argument("--date_col", type=str, default="date", help="时间列名")
    parser.add_argument("--target_col", type=str, default="price", help="目标列名（价格）")
    parser.add_argument("--sheet_name", type=str, default="0", help="Excel 工作表名/索引")
    parser.add_argument("--unique_id", type=str, default="SE2", help="时间序列标识")
    parser.add_argument("--local_tz", type=str, default="Europe/Stockholm", help="本地时区")
    parser.add_argument("--freq", type=str, default="h", help="时间频率")
    parser.add_argument("--csv_sep", type=str, default=",", help="CSV 分隔符")
    parser.add_argument("--csv_encoding", type=str, default="utf-8", help="CSV 编码")
    parser.add_argument(
        "--missing_strategy",
        type=str,
        default="interpolate",
        choices=["interpolate", "ffill", "raise"],
        help="缺失值处理策略"
    )

    # ========== 时间切分参数 ==========
    parser.add_argument("--train_start", type=str, default="2015-01-01 00:00:00+00:00", help="训练集开始时间")
    parser.add_argument("--train_end", type=str, default="2023-12-31 23:00:00+00:00", help="训练集结束时间")
    parser.add_argument("--val_start", type=str, default="2024-01-01 00:00:00+00:00", help="验证集开始时间")
    parser.add_argument("--val_end", type=str, default="2024-12-31 23:00:00+00:00", help="验证集结束时间")
    parser.add_argument("--test_start", type=str, default="2025-01-01 00:00:00+00:00", help="测试集开始时间")
    parser.add_argument("--test_end", type=str, default="2026-05-18 23:00:00+00:00", help="测试集结束时间")

    # ========== 业务预测任务参数（支持多个发布时间点） ==========
    parser.add_argument(
        "--input_days",
        type=int,
        default=28,
        help="模型输入历史天数（建议 28 或 56）"
    )
    parser.add_argument("--issued_tz", type=str, default="Europe/Stockholm", help="发布时区")
    parser.add_argument(
        "--insured_time",
        type=int,
        choices=[0, 12],
        default=0,
        help="发布时间点（只能是 0 或 12，表示 00:00 或 12:00 瑞典时间发布）"
    )
    parser.add_argument(
        "--pred_days",
        type=int,
        default=5,
        help="当 --target_hours 未指定时，target_hours=pred_days*24"
    )
    parser.add_argument(
        "--target_hours",
        type=int,
        default=None,
        help="固定评价小时数（默认 pred_days*24，即 5 天=120 小时）"
    )

    # ========== 模型参数 ==========
    parser.add_argument(
        "--models",
        type=str,
        default="NHITS,TCN,PatchTST",
        help="要训练的模型（逗号分隔，可选：NHITS,TCN,PatchTST）"
    )
    parser.add_argument(
        "--use_exog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用时间特征作为未来外生变量（PatchTST 会自动不使用）"
    )

    # ========== NeuralForecast 训练参数 ==========
    parser.add_argument("--max_steps", type=int, default=2000, help="最大训练步数")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--windows_batch_size", type=int, default=1024, help="窗口批次大小")
    parser.add_argument("--val_check_steps", type=int, default=100, help="验证间隔步数")
    parser.add_argument("--early_stop_patience_steps", type=int, default=10, help="早停耐心值")
    parser.add_argument(
        "--internal_val_days",
        type=int,
        default=90,
        help="做验证集 CV 时，从训练集末尾切出的内部早停验证天数"
    )
    parser.add_argument(
        "--nf_scaler_type",
        type=str,
        default="robust",
        choices=["identity", "standard", "robust", "minmax"],
        help="数据缩放类型"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mae",
        choices=["mae", "huber"],
        help="损失函数类型：mae（默认）、huber（对尖峰不敏感，缓解低估）"
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=10.0,
        help="Huber loss 的 delta 参数（仅 loss_type=huber 时生效）"
    )

    # ========== 评估参数 ==========
    parser.add_argument(
        "--eval_step_size",
        type=int,
        default=1,
        help="CV 滚动预测步长（默认 1）"
    )
    parser.add_argument(
        "--skip_val_cv",
        action="store_true",
        help="跳过验证集 CV，只跑测试集"
    )

    # ========== GPU / 系统参数 ==========
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out_dir", type=str, default="./outputs/nhits_tcn_patchtst", help="输出目录")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="可见 GPU 索引（如 '0,1'），也可在命令前用 CUDA_VISIBLE_DEVICES=0,1"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="加速器类型"
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="传给 PyTorch Lightning 的 devices 参数（默认 1，避免自动使用全部 GPU）"
    )
    parser.add_argument("--strategy", type=str, default="auto", help="Lightning 分布式策略")
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="medium",
        choices=["highest", "high", "medium"],
        help="矩阵乘法精度"
    )
    parser.add_argument(
        "--warn_cuda_grad_stream_mismatch",
        action="store_true",
        help="是否显示 CUDA 梯度流警告"
    )

    parser.add_argument(
        "--exog_configs", type=str, default="[]",
        help="外部外生变量配置，JSON 格式，如 '[{\"csv_path\":\"...\",\"name\":\"SE3\"}]'"
    )

    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
