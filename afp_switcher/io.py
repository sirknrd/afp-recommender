from __future__ import annotations

import pandas as pd


FUND_COLUMNS = ["A", "B", "C", "D", "E"]


def read_prices_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("CSV must include a 'date' column.")
    missing = [c for c in FUND_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing fund columns: {missing}. Expected {FUND_COLUMNS}.")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df = df.set_index("date")[FUND_COLUMNS].astype(float)
    return df


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().dropna(how="all")
    return rets

