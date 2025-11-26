"""
Indicator and helper functions used by the pattern detectors.
All rolling windows are right aligned to avoid look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's smoothing.
    Expects columns: high, low, close.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def linear_regression_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling linear regression slope (price per bar) using right-aligned windows.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < EPS:
        denom = EPS

    def _slope(arr: np.ndarray) -> float:
        y = arr
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=0)


def detect_swings(high: pd.Series, low: pd.Series, lookback: int):
    """
    Detect swing highs/lows.
    A swing high at i requires high[i] greater than highs in the lookback bars on both sides.
    Uses symmetric lookback; the last `lookback` bars cannot be confirmed and will be False.
    """
    n = len(high)
    swing_high = pd.Series(False, index=high.index)
    swing_low = pd.Series(False, index=low.index)
    for i in range(lookback, n - lookback):
        window_high = high.iloc[i - lookback : i + lookback + 1]
        h = high.iloc[i]
        if h == window_high.max() and (window_high == h).sum() == 1:
            swing_high.iloc[i] = True

        window_low = low.iloc[i - lookback : i + lookback + 1]
        l = low.iloc[i]
        if l == window_low.min() and (window_low == l).sum() == 1:
            swing_low.iloc[i] = True
    return swing_high, swing_low


def regression_from_points(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Simple linear regression on arbitrary x indices.
    Returns slope and intercept.
    """
    if len(x) < 2:
        return np.nan, np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < EPS:
        return np.nan, np.nan
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def line_value(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept
