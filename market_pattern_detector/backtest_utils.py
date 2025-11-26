from __future__ import annotations

import pandas as pd


def pattern_counts(data: pd.DataFrame) -> pd.Series:
    """
    Count occurrences of each pattern label.
    """
    return data["pattern"].value_counts()


def forward_return(data: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Simple forward return using close prices.
    """
    if "close" not in data:
        raise ValueError("close column is required to compute forward returns")
    future = data["close"].shift(-horizon)
    return (future - data["close"]) / data["close"]


def pattern_performance(data: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Aggregate mean/median forward returns by pattern label.
    """
    fwd = forward_return(data, horizon)
    grouped = pd.concat([data["pattern"], fwd.rename("fwd_return")], axis=1).dropna()
    return grouped.groupby("pattern")["fwd_return"].agg(["count", "mean", "median"])
