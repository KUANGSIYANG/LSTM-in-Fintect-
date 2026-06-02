from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "COST",
    "NFLX", "AMD", "ADBE", "PEP", "CSCO", "TMUS", "INTU", "QCOM", "TXN", "AMAT",
    "INTC", "AMGN", "HON", "BKNG", "VRTX", "ISRG", "LRCX", "REGN", "ADI", "PANW",
    "MU", "ADP", "MDLZ", "KLAC", "SBUX", "GILD", "SNPS", "CDNS", "MELI", "MAR",
    "ORLY", "CSX", "PYPL", "ABNB", "CRWD", "NXPI", "WDAY", "MRVL", "MNST", "ROP",
    "PCAR", "CHTR", "FTNT", "KDP", "DXCM", "AEP", "PAYX", "ROST", "KHC", "ODFL",
]


def normalize_yahoo_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
            "open": df["open"],
            "high": df["high"],
            "close": df["close"],
            "low": df["low"],
            "volume": df["volume"],
        }
    )
    out["p_change"] = out["close"].pct_change() * 100.0
    out["ticker"] = ticker
    out = out.dropna().replace([float("inf"), float("-inf")], pd.NA).dropna()
    return out[["date", "open", "high", "close", "low", "volume", "p_change", "ticker"]]


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return normalize_yahoo_frame(raw, ticker)


def download_ticker_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    symbol = ticker.lower()
    if not symbol.endswith(".us"):
        symbol = f"{symbol}.us"
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    if "No data" in response.text or len(response.text.splitlines()) <= 1:
        return pd.DataFrame()
    raw = pd.read_csv(pd.io.common.StringIO(response.text))
    raw = raw.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    out = raw[["date", "open", "high", "close", "low", "volume"]].copy()
    out["p_change"] = out["close"].pct_change() * 100.0
    out["ticker"] = ticker.upper().replace(".US", "")
    out = out.dropna().replace([float("inf"), float("-inf")], pd.NA).dropna()
    return out[["date", "open", "high", "close", "low", "volume", "p_change", "ticker"]]


def main():
    parser = argparse.ArgumentParser(description="Download Yahoo Finance panel data for larger finance experiments.")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--provider", choices=["yahoo", "stooq"], default="stooq")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", type=Path, default=Path("data/us_tech_2010_2024.csv"))
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    frames = []
    failed = []
    for i, ticker in enumerate(args.tickers, 1):
        try:
            print(f"[{i:03d}/{len(args.tickers):03d}] downloading {ticker}")
            if args.provider == "stooq":
                frame = download_ticker_stooq(ticker, args.start, args.end)
            else:
                frame = download_ticker(ticker, args.start, args.end)
            if len(frame) < 200:
                failed.append(ticker)
                print(f"  skipped {ticker}: only {len(frame)} rows")
            else:
                frames.append(frame)
                print(f"  rows={len(frame)}")
        except Exception as exc:
            failed.append(ticker)
            print(f"  failed {ticker}: {exc}")
        time.sleep(args.sleep)

    if not frames:
        raise RuntimeError("No ticker data downloaded.")

    panel = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    print(f"saved {args.output} rows={len(panel)} tickers={panel['ticker'].nunique()}")
    if failed:
        print("failed/skipped:", ",".join(failed))


if __name__ == "__main__":
    main()
