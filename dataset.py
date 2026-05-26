# -*- coding: utf-8 -*-
"""
数据集构建器模块

该模块负责：
1. 加载 CSV/Excel 数据文件
2. 处理缺失值
3. 生成时间特征（作为外生变量）
4. 按时间切分训练/验证/测试集
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from exogenous import ExogenousLoader


@dataclass
class SplitData:
    """
    切分后的数据集容器，符合 NeuralForecast 要求的长表格式
    
    属性：
        full_df: 完整数据集 (unique_id, ds, y[, feat_*])
        train_df: 训练集
        val_df: 验证集
        test_df: 测试集
        train_size: 训练集样本数
        val_size: 验证集样本数
        test_size: 测试集样本数
        futr_exog_cols: 未来外生变量列名列表（时间特征，如小时/星期）
        hist_exog_cols: 历史外生变量列名列表（如 SE3 真实电价，只有历史值）
    """
    full_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_size: int
    val_size: int
    test_size: int
    futr_exog_cols: List[str]
    hist_exog_cols: List[str]


class PriceDatasetBuilder:
    """
    电价预测数据集构建器
    
    设计原则：
    1. 支持 CSV 和 Excel 格式输入
    2. 只需要时间列和价格列
    3. 输出符合 NeuralForecast 的长表格式
    4. 自动生成时间特征作为外生变量
    5. 按连续时间序列切分数据集
    """

    def __init__(
        self,
        excel_path: str,           # 数据文件路径（CSV/Excel）
        date_col: str = "date",    # 时间列名
        target_col: Optional[str] = None,  # 目标列名（价格）
        sheet_name: Union[int, str] = 0,   # Excel 工作表名/索引
        unique_id: str = "SE2",    # 时间序列唯一标识
        local_tz: str = "Europe/Stockholm",  # 本地时区（用于特征生成）
        freq: str = "h",           # 时间频率（小时）
        missing_strategy: str = "interpolate",  # 缺失值处理策略
        train_start: str = "2015-01-01 00:00:00+00:00",  # 训练集开始时间
        train_end: str = "2023-12-31 23:00:00+00:00",    # 训练集结束时间
        val_start: str = "2024-01-01 00:00:00+00:00",    # 验证集开始时间
        val_end: str = "2024-12-31 23:00:00+00:00",      # 验证集结束时间
        test_start: str = "2025-01-01 00:00:00+00:00",   # 测试集开始时间
        test_end: str = "2026-05-01 23:00:00+00:00",     # 测试集结束时间
        csv_sep: str = ",",        # CSV 分隔符
        csv_encoding: str = "utf-8",  # CSV 编码
        exog_configs: Optional[List[dict]] = None,  # 外部外生变量配置列表
    ):
        # 保存文件路径和列配置
        self.data_path = excel_path
        self.date_col = date_col
        self.target_col = target_col
        self.sheet_name = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
        self.unique_id = unique_id
        self.local_tz = local_tz
        self.freq = freq
        self.missing_strategy = missing_strategy
        self.csv_sep = csv_sep
        self.csv_encoding = csv_encoding
        self.exog_configs = exog_configs or []

        # 将时间字符串转换为 UTC 时间戳
        self.train_start = self._to_utc(train_start)
        self.train_end = self._to_utc(train_end)
        self.val_start = self._to_utc(val_start)
        self.val_end = self._to_utc(val_end)
        self.test_start = self._to_utc(test_start)
        self.test_end = self._to_utc(test_end)

        # 验证时间切分的合理性
        self._validate_split_order()

    @staticmethod
    def _to_utc(x: str | pd.Timestamp) -> pd.Timestamp:
        """将时间字符串或 Timestamp 转换为 UTC 时间戳"""
        ts = pd.Timestamp(x)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")  # 无时区信息则视为 UTC
        else:
            ts = ts.tz_convert("UTC")   # 有时区则转换为 UTC
        return ts

    @staticmethod
    def _to_naive_utc(x: pd.Timestamp) -> pd.Timestamp:
        """将带时区的 UTC 时间转换为无时区标记的 UTC 时间"""
        return x.tz_convert("UTC").tz_localize(None)

    def _validate_split_order(self) -> None:
        """验证时间切分顺序是否正确"""
        if not (self.train_start <= self.train_end < self.val_start <= self.val_end < self.test_start <= self.test_end):
            raise ValueError(
                "时间切分需要满足：train_start <= train_end < val_start <= val_end < test_start <= test_end"
            )

    def _read_file(self) -> pd.DataFrame:
        """根据文件类型读取数据"""
        path = Path(self.data_path)
        
        # 检查文件是否存在
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")

        # 根据文件后缀选择读取方式
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                return pd.read_csv(path, sep=self.csv_sep, encoding=self.csv_encoding)
            except UnicodeDecodeError:
                # 处理 UTF-8 BOM 等编码问题
                return pd.read_csv(path, sep=self.csv_sep, encoding="utf-8-sig")
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path, sheet_name=self.sheet_name)
        
        raise ValueError(f"暂不支持的文件格式：{suffix}。请使用 .csv / .xlsx / .xls")

    def load(self, use_exog: bool = True) -> SplitData:
        """
        加载并处理数据集
        
        参数：
            use_exog: 是否使用外生变量（时间特征）
        
        返回：
            SplitData 对象，包含切分后的数据集
        """
        # 读取原始数据并删除全空列
        raw_df = self._read_file().dropna(axis=1, how="all")

        # 验证时间列是否存在
        if self.date_col not in raw_df.columns:
            raise ValueError(f"找不到时间列 '{self.date_col}'，当前列名：{list(raw_df.columns)}")

        # 如果未指定目标列，则自动选择非时间列
        if self.target_col is None:
            candidates = [c for c in raw_df.columns if c != self.date_col]
            if not candidates:
                raise ValueError("没有找到价格列，请传入 target_col 参数。")
            self.target_col = candidates[0]

        # 验证目标列是否存在
        if self.target_col not in raw_df.columns:
            raise ValueError(f"找不到价格列 '{self.target_col}'，当前列名：{list(raw_df.columns)}")

        # 创建基础 DataFrame，包含时间和价格
        df = pd.DataFrame()
        df["ds_utc"] = pd.to_datetime(raw_df[self.date_col], utc=True, errors="coerce")  # 转换为 UTC 时间
        df["y"] = pd.to_numeric(raw_df[self.target_col], errors="coerce")               # 转换为数值

        # 删除无效时间戳和重复值，按时间排序
        df = df.dropna(subset=["ds_utc"])
        df = df.drop_duplicates(subset=["ds_utc"], keep="last")
        df = df.sort_values("ds_utc")

        # 按小时频率重采样，确保时间序列连续
        df = df.set_index("ds_utc").asfreq(self.freq)
        df.index.name = "ds_utc"

        # 处理缺失值
        if df["y"].isna().any():
            missing_count = int(df["y"].isna().sum())
            if self.missing_strategy == "raise":
                raise ValueError(f"存在 {missing_count} 个缺失小时，请先处理缺失值。")
            elif self.missing_strategy == "ffill":
                df["y"] = df["y"].ffill().bfill()  # 前向填充 + 后向填充
            elif self.missing_strategy == "interpolate":
                df["y"] = df["y"].interpolate(method="time").ffill().bfill()  # 时间插值
            else:
                raise ValueError("missing_strategy 只能是 raise / ffill / interpolate")

        # 重置索引并添加时间特征
        df = df.reset_index()
        df = self._add_time_features(df)

        futr_exog_cols = [c for c in df.columns if c.startswith("feat_")]

        # 加载外部外生变量（如 SE3 电价）—— 这些是历史外生变量，只有过去的值
        df = self._load_exog_features(df)

        hist_exog_cols = [c for c in df.columns if c.startswith("feat_") and c not in futr_exog_cols]

        # 转换为无时区标记的 UTC 时间（NeuralForecast 要求）
        df["ds"] = df["ds_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["unique_id"] = self.unique_id

        # 过滤到目标时间范围
        mask = (df["ds_utc"] >= self.train_start) & (df["ds_utc"] <= self.test_end)
        df = df.loc[mask].copy().reset_index(drop=True)

        if df.empty:
            raise ValueError("过滤 train_start ~ test_end 后数据为空，请检查时间范围。")

        # 确定输出列（外生变量以 feat_ 开头）
        all_feat_cols = [c for c in df.columns if c.startswith("feat_")]
        cols = ["unique_id", "ds", "y"]
        if use_exog:
            cols += all_feat_cols

        # 切分数据集
        train_raw = self._slice_raw(df, self.train_start, self.train_end)
        val_raw = self._slice_raw(df, self.val_start, self.val_end)
        test_raw = self._slice_raw(df, self.test_start, self.test_end)

        # 验证切分结果
        if len(train_raw) == 0:
            raise ValueError("训练集为空，请检查 train_start / train_end。")
        if len(val_raw) == 0:
            raise ValueError("验证集为空，请检查 val_start / val_end。")
        if len(test_raw) == 0:
            raise ValueError("测试集为空，请检查 test_start / test_end。")

        # 返回切分后的数据集
        return SplitData(
            full_df=df[cols].copy(),
            train_df=train_raw[cols].copy(),
            val_df=val_raw[cols].copy(),
            test_df=test_raw[cols].copy(),
            train_size=len(train_raw),
            val_size=len(val_raw),
            test_size=len(test_raw),
            futr_exog_cols=futr_exog_cols if use_exog else [],
            hist_exog_cols=hist_exog_cols if use_exog else [],
        )

    def _slice_raw(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """按时间范围切片数据"""
        return df[(df["ds_utc"] >= start) & (df["ds_utc"] <= end)].copy()

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征（作为未来外生变量）

        生成的特征：
        - 小时周期 sin/cos
        - 星期周期 sin/cos
        - 一周内小时周期 sin/cos
        - 月份周期 sin/cos
        - 年内日周期 sin/cos
        - 周末/工作时段/峰谷时段
        - 季节/采暖季
        - UTC 偏移量
        - 是否夏令时
        """
        df = df.copy()

        # 将 UTC 时间转换为本地时间
        local_time = df["ds_utc"].dt.tz_convert(self.local_tz)

        # 基础时间字段
        hour = local_time.dt.hour.astype(float)
        dow = local_time.dt.dayofweek.astype(float)
        month = local_time.dt.month.astype(float)
        doy = local_time.dt.dayofyear.astype(float)

        # 小时周期：24小时
        df["feat_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["feat_hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # 星期周期：7天
        df["feat_dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["feat_dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # 一周内第几个小时：168小时周期
        hour_of_week = dow * 24 + hour
        df["feat_how_sin"] = np.sin(2 * np.pi * hour_of_week / 168)
        df["feat_how_cos"] = np.cos(2 * np.pi * hour_of_week / 168)

        # 月份周期
        df["feat_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        df["feat_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

        # 年内日周期
        df["feat_doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
        df["feat_doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25)

        # 周末
        df["feat_is_weekend"] = (dow >= 5).astype(float)

        # 工作日
        df["feat_is_workday"] = (dow < 5).astype(float)

        # 峰谷时段
        df["feat_is_night"] = ((hour >= 0) & (hour <= 5)).astype(float)

        df["feat_is_morning_peak"] = (
            (hour >= 7) & (hour <= 10) & (dow < 5)
        ).astype(float)

        df["feat_is_evening_peak"] = (
            (hour >= 17) & (hour <= 20) & (dow < 5)
        ).astype(float)

        df["feat_is_business_hour"] = (
            (hour >= 8) & (hour <= 18) & (dow < 5)
        ).astype(float)

        df["feat_is_offpeak"] = (
            (hour <= 6) | (hour >= 22)
        ).astype(float)

        # 季节特征
        df["feat_is_winter"] = local_time.dt.month.isin([12, 1, 2]).astype(float)
        df["feat_is_summer"] = local_time.dt.month.isin([6, 7, 8]).astype(float)

        df["feat_is_heating_season"] = (
            local_time.dt.month.isin([10, 11, 12, 1, 2, 3])
        ).astype(float)

        # UTC 偏移量
        utc_offset = local_time.map(
            lambda x: x.utcoffset().total_seconds() / 3600.0
        )
        df["feat_utc_offset"] = utc_offset.astype(float)

        # 是否夏令时
        df["feat_is_dst"] = local_time.map(
            lambda x: float(x.dst().total_seconds() != 0)
        )

        return df

    def _load_exog_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.exog_configs:
            return df

        loader = self.get_exog_loader()

        common_ds_utc = loader.find_common_range(df["ds_utc"])
        t_min, t_max = common_ds_utc.min(), common_ds_utc.max()
        df = df[(df["ds_utc"] >= t_min) & (df["ds_utc"] <= t_max)].reset_index(drop=True)

        exog_feats = loader.load_features(
            df["ds_utc"],
            missing_strategy=self.missing_strategy,
        )

        for col in loader.feat_cols:
            df[col] = exog_feats[col].values

        return df

    def get_exog_loader(self) -> Optional[ExogenousLoader]:
        if not self.exog_configs:
            return None

        return ExogenousLoader(
            self.exog_configs,
            csv_sep=self.csv_sep,
            csv_encoding=self.csv_encoding,
        )

    def split_boundaries_naive(self) -> dict:
        """返回无时区标记的 UTC 边界（用于与 NeuralForecast 对齐）"""
        return {
            "train_start": self._to_naive_utc(self.train_start),
            "train_end": self._to_naive_utc(self.train_end),
            "val_start": self._to_naive_utc(self.val_start),
            "val_end": self._to_naive_utc(self.val_end),
            "test_start": self._to_naive_utc(self.test_start),
            "test_end": self._to_naive_utc(self.test_end),
        }

    def split_boundaries_utc(self) -> dict:
        """返回带时区的 UTC 边界"""
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }
