#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot forecasted electricity prices: Montel forecast vs model predictions.

Usage:
    python plot_forecast.py --montel_csv <path> --pred_csv <path>
    python plot_forecast.py --montel_csv <path> --pred_csv <path> --model_name ours
"""

import argparse
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

H_COL_RE = re.compile(r"^h(\d+)$", re.IGNORECASE)


def load_montel_long(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    raw = pd.read_csv(p, encoding="utf-8-sig")
    raw = raw.dropna(axis=1, how="all")

    h_cols = []
    for c in raw.columns:
        m = H_COL_RE.fullmatch(str(c).strip())
        if m:
            h_cols.append((c, int(m.group(1))))

    if not h_cols:
        raise ValueError("No h0001, h0002, ... columns found in Montel file")

    h_cols_sorted = sorted(h_cols, key=lambda x: x[1])
    h_col_names = [c for c, _ in h_cols_sorted]

    df = raw[["issued_utc", "begin_utc"] + h_col_names].copy()
    df["issued_utc"] = pd.to_datetime(df["issued_utc"], utc=True)
    df["begin_utc"] = pd.to_datetime(df["begin_utc"], utc=True)

    long = df.melt(
        id_vars=["issued_utc", "begin_utc"],
        value_vars=h_col_names,
        var_name="h_col",
        value_name="montel",
    )
    long["target_horizon"] = long["h_col"].str.extract(r"(\d+)").astype(int)
    long["ds_utc"] = long["begin_utc"] + pd.to_timedelta(long["target_horizon"] - 1, unit="h")
    long["montel"] = pd.to_numeric(long["montel"], errors="coerce")

    return long[["issued_utc", "begin_utc", "ds_utc", "target_horizon", "montel"]]


def build_parser():
    parser = argparse.ArgumentParser(description="Plot forecast comparison: Montel vs model")
    parser.add_argument(
        "--montel_csv", type=str, required=True,
        help="Montel forecast CSV file path",
    )
    parser.add_argument(
        "--pred_csv", type=str, required=True,
        help="Model predictions CSV file path",
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Custom label for model predictions in legend (default: original model name)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./outputs/commits",
        help="Output directory for saved plots",
    )
    parser.add_argument(
        "--max_plots", type=int, default=0,
        help="Max number of issued_utc to plot (0 = all, default: 0). "
             "取 pred 中最新的 N 个发布时间点进行绘图",
    )
    return parser


def main():
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Montel forecast ...")
    montel = load_montel_long(args.montel_csv)
    print(f"  Montel: {len(montel)} rows, {montel['issued_utc'].nunique()} issued_utc")

    print("Loading model predictions ...")
    pred = pd.read_csv(args.pred_csv)
    pred["ds_utc"] = pd.to_datetime(pred["ds_utc"], utc=True)
    pred["issued_utc"] = pd.to_datetime(pred["issued_utc"], utc=True)
    print(f"  Model: {len(pred)} rows, {pred['issued_utc'].nunique()} issued_utc")

    model_cols = [c for c in pred.columns if c not in {
        "unique_id", "split", "issued_utc", "issued_local", "issued_hour_local",
        "cutoff_utc", "begin_utc", "begin_local", "target_end_utc", "target_end_local",
        "ds_utc", "ds", "horizon", "target_horizon", "first_target_horizon",
        "last_target_horizon", "forecast_day", "hour_in_forecast_day",
        "issued_date", "target_date", "cutoff",
    } and not c.startswith("feat_")]
    print(f"  Model columns: {model_cols}")

    if args.model_name:
        n_models = len(model_cols)
        if n_models == 1:
            model_labels = [args.model_name]
        else:
            model_labels = [f"{args.model_name}-{col}" for col in model_cols]
        print(f"  Model label: {model_labels}")
    else:
        model_labels = list(model_cols)

    pred_issued = sorted(pred["issued_utc"].unique())

    if len(pred_issued) == 0:
        print("No pred issued_utc found!")
        return

    # 取 pred 中最新的 N 个发布时间点（默认全部，可通过参数控制）
    plot_issued = pred_issued[-args.max_plots:] if args.max_plots else pred_issued
    print(f"  Plot issued_utc count: {len(plot_issued)}")

    montel_issued_set = set(montel["issued_utc"]) if not montel.empty else set()

    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, issued in enumerate(plot_issued):
        issued_str = pd.Timestamp(issued).strftime("%Y-%m-%d %H:%M")

        p_part = pred[pred["issued_utc"] == issued].sort_values("ds_utc")
        if p_part.empty:
            continue

        has_montel = issued in montel_issued_set
        m_part = montel[montel["issued_utc"] == issued].sort_values("ds_utc") if has_montel else pd.DataFrame()

        fig, ax = plt.subplots(figsize=(16, 7))

        if not m_part.empty:
            ax.plot(
                m_part["ds_utc"], m_part["montel"],
                marker=".", linewidth=1.5, label="Montel", color="#e377c2", alpha=0.8,
            )

        for i, col in enumerate(model_cols):
            if col not in p_part.columns:
                continue
            ax.plot(
                p_part["ds_utc"], p_part[col],
                marker=".", linewidth=1.5, label=model_labels[i],
                color=colors[i % len(colors)], alpha=0.8,
            )

        ax.set_title(f"Forecast Comparison  |  Issued: {issued_str} UTC")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        all_data_parts = [p_part["ds_utc"]]
        if not m_part.empty:
            all_data_parts.append(m_part["ds_utc"])
        all_times = pd.concat(all_data_parts).sort_values()
        if len(all_times) >= 3:
            t0 = all_times.iloc[0]
            t1 = all_times.iloc[len(all_times) // 2]
            t2 = all_times.iloc[-1]
            ax.set_xticks([t0, t1, t2])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

        plt.tight_layout()

        safe_issued = issued_str.replace(":", "").replace(" ", "_")
        out_path = out_dir / f"forecast_comparison_{safe_issued}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path.name}")

    print(f"\nDone. {len(plot_issued)} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
