#!/usr/bin/env python3
"""
Indicator implementations: RSI, ATR, Alligator (jaw/teeth/lips), SMMA.
"""

import pandas as pd
import numpy as np


def smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed moving average (Wilder)."""
    alpha = 1 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = smma(gain, period)
    avg_loss = smma(loss, period)
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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


def alligator(close: pd.Series, jaw_period: int = 13, teeth_period: int = 8, lips_period: int = 5) -> pd.DataFrame:
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    return pd.DataFrame({"jaw": jaw, "teeth": teeth, "lips": lips})


def find_pivots(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    """
    返回 pivot_high_idx, pivot_low_idx 列表
    """
    pivot_highs = []
    pivot_lows = []
    n = len(high)
    for i in range(left, n - right):
        window_high = high[i - left : i + right + 1]
        if high[i] == window_high.max() and (window_high == high[i]).sum() == 1:
            pivot_highs.append(i)
        window_low = low[i - left : i + right + 1]
        if low[i] == window_low.min() and (window_low == low[i]).sum() == 1:
            pivot_lows.append(i)
    return pivot_highs, pivot_lows
