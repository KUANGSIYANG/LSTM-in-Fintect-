from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "Adilbai/stock-dataset"
REPO_FILE = "data/train-00000-of-00001.parquet"


BASE_COLUMNS = ["date", "open", "high", "close", "low", "volume", "p_change", "ticker"]
TARGET_COLUMN = "future_return_1d_pct"
LABEL_COLUMNS = ["future_up_1d", "future_category_1d"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: str(col).strip().lower() for col in df.columns}
    df = df.rename(columns=rename)

    aliases = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Ticker": "ticker",
    }
    for old, new in aliases.items():
        old_lower = old.lower()
        if old_lower in df.columns:
            df = df.rename(columns={old_lower: new})

    df.columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]
    return df


def scale_future_return(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    # Most finance datasets store returns either as decimals (0.01) or percent (1.0).
    # Convert decimal-like returns to percent for consistency with the HS300 pipeline.
    finite = values[np.isfinite(values)]
    if len(finite) and finite.abs().quantile(0.95) < 1.0:
        values = values * 100.0
    return values


def choose_feature_columns(df: pd.DataFrame, max_features: int) -> list[str]:
    forbidden = {
        "date",
        "ticker",
        "future_return_1d",
        "future_return_1d_pct",
        "future_up_1d",
        "future_category_1d",
        "future_return_5d",
        "future_up_5d",
        "future_category_5d",
        "future_return_10d",
        "future_up_10d",
        "future_category_10d",
        "future_return_20d",
        "future_up_20d",
        "future_category_20d",
    }

    preferred = [
        "open",
        "high",
        "close",
        "low",
        "volume",
        "p_change",
        "open_close_ratio",
        "volume_sma",
        "volume_ratio",
        "close_lag_1",
        "close_lag_2",
        "close_lag_3",
        "close_lag_5",
        "close_lag_10",
        "volume_lag_1",
        "volume_lag_2",
        "volume_lag_3",
        "volume_lag_5",
        "volume_lag_10",
    ]

    numeric_cols = []
    for col in df.columns:
        if col in forbidden:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            df[col] = converted
            numeric_cols.append(col)

    ordered = [col for col in preferred if col in numeric_cols]
    ordered += [col for col in numeric_cols if col not in ordered]

    if max_features > 0:
        ordered = ordered[:max_features]

    if TARGET_COLUMN not in ordered:
        ordered.append(TARGET_COLUMN)
    return ordered


def prepare_dataframe(df: pd.DataFrame, max_features: int, max_tickers: int) -> pd.DataFrame:
    df = normalize_columns(df)

    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("Expected columns Date/date and Ticker/ticker in the Hugging Face dataset.")
    if "future_return_1d" not in df.columns:
        raise ValueError("Expected Future_Return_1d/future_return_1d in the Hugging Face dataset.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "p_change" not in df.columns:
        df["p_change"] = df.groupby("ticker")["close"].pct_change() * 100.0

    df[TARGET_COLUMN] = scale_future_return(df["future_return_1d"])
    if "future_up_1d" in df.columns:
        df["future_up_1d"] = pd.to_numeric(df["future_up_1d"], errors="coerce")
    else:
        df["future_up_1d"] = (df[TARGET_COLUMN] > 0).astype(int)

    if "future_category_1d" in df.columns:
        df["future_category_1d"] = pd.to_numeric(df["future_category_1d"], errors="coerce")

    if max_tickers > 0:
        tickers = df["ticker"].drop_duplicates().head(max_tickers)
        df = df[df["ticker"].isin(tickers)]

    feature_cols = choose_feature_columns(df, max_features=max_features)
    keep = ["date", "ticker"] + feature_cols + [col for col in LABEL_COLUMNS if col in df.columns]
    keep = list(dict.fromkeys(keep))

    out = df[keep].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def split_by_date(df: pd.DataFrame, train_end: str, test_start: str):
    dates = pd.to_datetime(df["date"], errors="coerce")
    train = df[dates <= pd.Timestamp(train_end)].copy()
    test = df[dates >= pd.Timestamp(test_start)].copy()
    return train, test


def main():
    parser = argparse.ArgumentParser(description="Prepare the Hugging Face S&P500 stock dataset for P-sLSTM experiments.")
    parser.add_argument("--cache_dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/sp500"))
    parser.add_argument("--max_features", type=int, default=32)
    parser.add_argument("--max_tickers", type=int, default=0, help="0 means all tickers")
    parser.add_argument("--train_end", default="2021-12-31")
    parser.add_argument("--test_start", default="2022-01-01")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=REPO_FILE,
        repo_type="dataset",
        cache_dir=str(args.cache_dir),
    )
    print(f"downloaded: {parquet_path}")

    raw = pd.read_parquet(parquet_path)
    prepared = prepare_dataframe(raw, max_features=args.max_features, max_tickers=args.max_tickers)
    train, test = split_by_date(prepared, train_end=args.train_end, test_start=args.test_start)

    full_path = args.out_dir / "sp500_full.csv"
    train_path = args.out_dir / "sp500_train.csv"
    test_path = args.out_dir / "sp500_test.csv"
    metadata_path = args.out_dir / "metadata.json"

    prepared.to_csv(full_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    metadata = {
        "repo_id": REPO_ID,
        "repo_file": REPO_FILE,
        "rows_full": int(len(prepared)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "tickers": int(prepared["ticker"].nunique()),
        "features": [col for col in prepared.columns if col not in {"date", "ticker"}],
        "target_col": TARGET_COLUMN,
        "train_end": args.train_end,
        "test_start": args.test_start,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {full_path}")
    print(f"saved: {train_path}")
    print(f"saved: {test_path}")
    print(f"rows full/train/test: {len(prepared)}/{len(train)}/{len(test)}")
    print(f"tickers: {prepared['ticker'].nunique()}")
    print(f"target: {TARGET_COLUMN}")


if __name__ == "__main__":
    main()
