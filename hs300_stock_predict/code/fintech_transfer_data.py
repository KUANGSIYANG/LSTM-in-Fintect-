from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CANONICAL_COLUMNS = {
    "date": ["date", "Date", "日期"],
    "open": ["open", "Open", "开盘价"],
    "high": ["high", "High", "最高价"],
    "close": ["close", "Close", "收盘价"],
    "low": ["low", "Low", "最低价"],
    "volume": ["volume", "Volume", "成交量"],
    "p_change": ["p_change", "pct_chg", "change", "涨跌幅"],
}


@dataclass
class SeriesData:
    segments: list[np.ndarray]
    feature_cols: list[str]
    target_col: str
    target_idx: int
    mean: np.ndarray
    std: np.ndarray


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    column_set = {str(c).strip(): c for c in columns}
    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]
    return None


def _canonicalize_stock_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    date_col = _find_column(df.columns, CANONICAL_COLUMNS["date"])
    rename_map = {}
    for canonical, candidates in CANONICAL_COLUMNS.items():
        found = _find_column(df.columns, candidates)
        if found is not None:
            rename_map[found] = canonical

    df = df.rename(columns=rename_map)
    date_col = "date" if date_col in rename_map else date_col
    feature_cols = ["open", "high", "close", "low", "volume", "p_change"]
    if all(col in df.columns for col in feature_cols):
        group_cols = [col for col in ["ticker", "stock", "code", "ts_code", "symbol"] if col in df.columns]
        leading_cols = ([date_col] if date_col is not None else []) + feature_cols + group_cols
        extra_cols = [col for col in df.columns if col not in set(leading_cols)]
        keep_cols = leading_cols + extra_cols
        out = df[keep_cols].copy()
        if date_col is not None and date_col != "date":
            out = out.rename(columns={date_col: "date"})
        return out, "date" if date_col is not None else None

    return df.copy(), date_col


def _numeric_feature_columns(df: pd.DataFrame, date_col: str | None) -> list[str]:
    excluded = {date_col, "date", "ticker", "stock", "code", "ts_code", "symbol"}
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            df[col] = converted
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns were found in the CSV file.")
    return cols


def _split_segments(df: pd.DataFrame, date_col: str | None, feature_cols: list[str]) -> list[np.ndarray]:
    values = df[feature_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    values = values.ffill().bfill().fillna(0.0)

    for group_col in ["ticker", "stock", "code", "ts_code", "symbol"]:
        if group_col in df.columns:
            segments = []
            row_ids = np.arange(len(df))
            for _, group_idx in pd.Series(row_ids).groupby(df[group_col], sort=False):
                idx = group_idx.to_numpy()
                if len(idx):
                    segments.append(values.iloc[idx].to_numpy(dtype=np.float32))
            return segments

    if date_col is None or date_col not in df.columns:
        return [values.to_numpy(dtype=np.float32)]

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.notna().all():
        reset_positions = np.flatnonzero(dates.diff().dt.total_seconds().fillna(0).to_numpy() < 0)
    else:
        reset_positions = np.array([], dtype=np.int64)

    starts = [0] + reset_positions.tolist()
    ends = reset_positions.tolist() + [len(df)]
    segments = []
    arr = values.to_numpy(dtype=np.float32)
    for start, end in zip(starts, ends):
        if end > start:
            segments.append(arr[start:end])
    return segments


def load_series_csv(
    csv_path: str | Path,
    target_col: str = "p_change",
    feature_cols: list[str] | None = None,
    max_features: int = 0,
    fit_stats: bool = True,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> SeriesData:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df, date_col = _canonicalize_stock_frame(df)

    if feature_cols is None:
        feature_cols = _numeric_feature_columns(df, date_col)
    else:
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required feature columns: {missing}")
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_cols = [
        col for col in feature_cols
        if not (str(col).startswith("future_") and col != target_col)
    ]

    if target_col not in feature_cols:
        target_col = feature_cols[-1]

    if max_features > 0 and len(feature_cols) > max_features:
        if target_col in feature_cols[:max_features]:
            feature_cols = feature_cols[:max_features]
        else:
            feature_cols = feature_cols[: max_features - 1] + [target_col]

    segments = _split_segments(df, date_col, feature_cols)
    stacked = np.concatenate(segments, axis=0)

    if fit_stats:
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
    elif mean is None or std is None:
        raise ValueError("mean/std must be supplied when fit_stats=False")

    std = np.where(std < 1e-6, 1.0, std)
    return SeriesData(
        segments=segments,
        feature_cols=list(feature_cols),
        target_col=target_col,
        target_idx=list(feature_cols).index(target_col),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


class ForecastWindowDataset(Dataset):
    def __init__(self, series: SeriesData, seq_len: int, pred_len: int):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for segment in series.segments:
            if len(segment) < seq_len + pred_len:
                continue
            normalized = (segment - series.mean) / series.std
            for start in range(0, len(segment) - seq_len - pred_len + 1):
                x = normalized[start : start + seq_len]
                y = normalized[start + seq_len : start + seq_len + pred_len]
                raw_target = segment[
                    start + seq_len : start + seq_len + pred_len,
                    series.target_idx,
                ]
                self.samples.append((x.astype(np.float32), y.astype(np.float32), raw_target.astype(np.float32)))

        if not self.samples:
            raise ValueError(
                f"No windows created. Need segment length >= seq_len + pred_len "
                f"({seq_len + pred_len})."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y, raw_target = self.samples[index]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(raw_target)


def split_dataset(dataset: Dataset, train_ratio: float, seed: int):
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)


def pchange_to_class(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    bins = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return np.digitize(values, bins, right=True)
