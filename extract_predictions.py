#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract columns A(unique_id), L(ds), U(PatchTST) from predictions CSV,
rename to unique_id, ds, ours, and save with a derived filename.

Usage:
    python extract_predictions.py
    python extract_predictions.py --input <path> --out_dir <dir>
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent


def build_parser():
    today_str = date.today().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Extract and rename prediction columns")
    parser.add_argument(
        "--input", type=str,
        default=str(_BASE_DIR / "outputs" / "predict_results" / f"predict_{today_str}_issued_00_1056h.csv"),
        help="Input predictions CSV file",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=str(_BASE_DIR / "outputs" / "commits"),
        help="Output directory",
    )
    return parser


def _safe(s: str) -> str:
    s = str(s)
    if "+" in s:
        s = s[: s.index("+")]
    return s.replace(":", "_").replace(" ", "_")


def main():
    args = build_parser().parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    c_idx = 2
    q_idx = 16

    c_col_name = df.columns[c_idx]
    c_first = _safe(df.iloc[0, c_idx])
    q_max = int(df.iloc[:, q_idx].max())
    out_name = f"{c_col_name}_{c_first}_{q_max}day.csv"
    out_path = out_dir / out_name

    out = pd.DataFrame({
        "unique_id": df.iloc[:, 0],
        "ds": df.iloc[:, 11],
        "ours": df.iloc[:, 20],
    })

    out.to_csv(out_path, index=False)

    print(f"Input : {input_path}")
    print(f"Output: {out_path}")
    print(f"Rows  : {len(out)}")


if __name__ == "__main__":
    main()
