# -*- coding: utf-8 -*-
"""
外生变量加载与对齐模块

职责：
1. 从外部 CSV 文件加载电价等外生变量
2. 将外生变量对齐到基准时间索引（如 SE2 真实电价）
3. 时间不对齐时给出明确的报错信息
4. 支持生成未来外生变量（用最近已知值填充）
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


def load_exog_csv(
    csv_path: str,
    name: str,
    date_col: str = "date",
    value_col: str = None,
    csv_sep: str = ",",
    csv_encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    加载单个外生变量 CSV 文件

    参数：
        csv_path: CSV 文件路径
        name: 变量名称，输出的列名为 feat_{name}
        date_col: 时间列名
        value_col: 值列名，不指定则自动检测
        csv_sep: CSV 分隔符
        csv_encoding: CSV 编码

    返回：
        包含 ds_utc (UTC) 和 feat_{name} 的 DataFrame
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"外生变量文件不存在: {csv_path}")

    try:
        raw = pd.read_csv(p, sep=csv_sep, encoding=csv_encoding)
    except UnicodeDecodeError:
        raw = pd.read_csv(p, sep=csv_sep, encoding="utf-8-sig")

    raw = raw.dropna(axis=1, how="all")

    if date_col not in raw.columns:
        raise ValueError(
            f"外生变量文件 {csv_path} 中找不到时间列 '{date_col}'，"
            f"当前列名：{list(raw.columns)}"
        )

    if value_col is None:
        candidates = [c for c in raw.columns if c != date_col]
        if not candidates:
            raise ValueError(
                f"外生变量文件 {csv_path} 中没有找到值列，请指定 value_col"
            )
        value_col = candidates[0]

    if value_col not in raw.columns:
        raise ValueError(
            f"外生变量文件 {csv_path} 中找不到值列 '{value_col}'，"
            f"当前列名：{list(raw.columns)}"
        )

    feat_col = f"feat_{name}"

    df = pd.DataFrame({
        "ds_utc": pd.to_datetime(raw[date_col], utc=True, errors="coerce"),
        feat_col: pd.to_numeric(raw[value_col], errors="coerce"),
    })

    df = df.dropna(subset=["ds_utc"])
    df = df.drop_duplicates(subset=["ds_utc"], keep="last")
    df = df.sort_values("ds_utc").reset_index(drop=True)

    return df


def align_exog_to_base(
    exog_df: pd.DataFrame,
    base_index: pd.DatetimeIndex,
    exog_name: str,
    missing_strategy: str = "interpolate",
) -> pd.DataFrame:
    """
    将外生变量对齐到基准时间索引

    参数：
        exog_df: 外生变量 DataFrame，包含 ds_utc 和 feat_{name} 列
        base_index: 基准时间索引（如 SE2 真实电价的 ds_utc）
        exog_name: 外生变量名称（用于报错信息）
        missing_strategy: 缺失值处理策略 (interpolate / ffill / raise)

    返回：
        对齐后的 DataFrame，索引为 base_index，包含 feat_{name} 列

    异常：
        ValueError: 时间未对齐时抛出，指明是哪个外生变量、哪些时间点不对齐
    """
    feat_col = f"feat_{exog_name}"

    aligned = pd.DataFrame({"ds_utc": base_index})
    aligned = aligned.merge(exog_df, on="ds_utc", how="left")

    missing_mask = aligned[feat_col].isna()

    if missing_mask.any() and missing_strategy == "raise":
        missing_times = aligned.loc[missing_mask, "ds_utc"].tolist()
        n_missing = missing_mask.sum()
        raise ValueError(
            f"\n外生变量 '{exog_name}' 与基准数据时间未对齐。\n"
            f"缺失时间点数: {n_missing} / {len(base_index)}\n"
            f"前 10 个缺失时间点:\n  " +
            "\n  ".join(str(t) for t in missing_times[:10]) +
            (f"\n  ... 还有 {n_missing - 10} 个" if n_missing > 10 else "") +
            f"\n请检查文件中的时间列是否与基准数据的时间列完全一致。"
        )

    if missing_mask.any():
        if missing_strategy == "interpolate":
            aligned = aligned.set_index("ds_utc")
            aligned[feat_col] = aligned[feat_col].interpolate(method="time").ffill().bfill()
            aligned = aligned.reset_index()
        elif missing_strategy == "ffill":
            aligned[feat_col] = aligned[feat_col].ffill().bfill()

    return aligned


class ExogenousLoader:
    """
    外生变量加载器

    从外部 CSV 文件加载多个外生变量，与基准数据时间对齐，
    并生成未来外生变量（用于预测）。

    用法：
        loader = ExogenousLoader([
            {
                "csv_path": "sourceData/.../SE3_price.csv",
                "name": "SE3",
            },
        ])
        # 对齐历史数据
        feats_df = loader.load_features(base_ds_utc, freq="h")
        # 生成未来外生变量
        future_feats = loader.build_future_features(future_ds_utc)
    """

    def __init__(
        self,
        exog_configs: List[dict],
        csv_sep: str = ",",
        csv_encoding: str = "utf-8",
    ):
        self.exog_configs = exog_configs
        self.csv_sep = csv_sep
        self.csv_encoding = csv_encoding

        self._exog_dfs = {}
        self._exog_names = []

        for cfg in exog_configs:
            name = cfg["name"]
            self._exog_names.append(name)
            self._exog_dfs[name] = load_exog_csv(
                csv_path=cfg["csv_path"],
                name=name,
                date_col=cfg.get("date_col", "date"),
                value_col=cfg.get("value_col", None),
                csv_sep=self.csv_sep,
                csv_encoding=self.csv_encoding,
            )

    @property
    def feat_cols(self) -> List[str]:
        """返回所有外生变量的列名列表"""
        return [f"feat_{name}" for name in self._exog_names]

    def find_common_range(
        self,
        base_ds_utc: pd.DatetimeIndex,
    ) -> pd.DatetimeIndex:
        """
        取所有外生变量与基准数据的共同时间覆盖范围

        返回裁剪后的 DatetimeIndex，仅保留所有变量都有数据的时间段
        """
        valid = np.ones(len(base_ds_utc), dtype=bool)

        for name in self._exog_names:
            exog_df = self._exog_dfs[name]
            if exog_df.empty:
                continue
            first_ts = exog_df["ds_utc"].iloc[0]
            last_ts = exog_df["ds_utc"].iloc[-1]
            valid &= (base_ds_utc >= first_ts) & (base_ds_utc <= last_ts)

        if valid.all():
            return base_ds_utc

        trimmed = base_ds_utc[valid]
        if len(trimmed) == 0:
            names = ", ".join(self._exog_names)
            raise ValueError(
                f"所有外生变量 ({names}) 与基准数据没有共同时间范围。\n"
                f"基准数据范围: {base_ds_utc[0]} ~ {base_ds_utc[-1]}"
            )

        n_removed = len(base_ds_utc) - len(trimmed)
        print(
            f"[外生变量对齐] 裁剪了 {n_removed} 个时间点，"
            f"剩余 {len(trimmed)} 行"
        )
        return trimmed

    def load_features(
        self,
        base_ds_utc: pd.DatetimeIndex,
        missing_strategy: str = "interpolate",
    ) -> pd.DataFrame:
        """
        加载所有外生变量，对齐到基准时间索引

        参数：
            base_ds_utc: 基准 UTC 时间索引
            missing_strategy: 缺失值处理策略

        返回：
            DataFrame，索引与 base_ds_utc 对齐，列为 feat_{name}
        """
        result = pd.DataFrame({"ds_utc": base_ds_utc})

        for name in self._exog_names:
            aligned = align_exog_to_base(
                self._exog_dfs[name],
                base_ds_utc,
                exog_name=name,
                missing_strategy=missing_strategy,
            )
            result[f"feat_{name}"] = aligned[f"feat_{name}"].values

        return result.drop(columns=["ds_utc"])

    def build_future_features(
        self,
        future_ds_utc: pd.DatetimeIndex,
        last_known_ds_utc: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        生成未来外生变量

        对于没有未来值的外生变量，使用最近已知值填充。

        参数：
            future_ds_utc: 未来 UTC 时间索引
            last_known_ds_utc: 基准数据的最后已知时间点

        返回：
            DataFrame，包含 feat_{name} 列
        """
        result = pd.DataFrame(index=future_ds_utc)

        for name in self._exog_names:
            feat_col = f"feat_{name}"
            exog_df = self._exog_dfs[name]

            result[feat_col] = np.nan

            exog_known = exog_df[exog_df["ds_utc"] <= future_ds_utc.max()]
            if exog_known.empty:
                last_val = exog_df[feat_col].iloc[-1] if not exog_df.empty else np.nan
                result[feat_col] = last_val
                continue

            for idx in future_ds_utc:
                past_vals = exog_df[exog_df["ds_utc"] <= idx][feat_col]
                if not past_vals.empty:
                    result.loc[idx, feat_col] = past_vals.iloc[-1]
                elif not exog_df.empty:
                    result.loc[idx, feat_col] = exog_df[feat_col].iloc[-1]

        return result
