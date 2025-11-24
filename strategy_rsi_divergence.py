#!/usr/bin/env python3
"""
RSI 背离策略：
- 找 pivot 高低点
- 最近 lookback 内的高低点背离
- SL: swing 或 ATR
- TP: 固定 R
"""

from typing import List, Dict
import pandas as pd
import numpy as np

from indicators import rsi, atr, find_pivots


def detect_rsi_divergence_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
    lookback_bars: int = 20,
    pivot_left: int = 2,
    pivot_right: int = 2,
    min_rsi_diff: float = 3.0,
    sl_mode: str = "swing",
    atr_period: int = 14,
    k_sl: float = 1.5,
    tp_R: float = 2.0,
) -> List[Dict]:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必要列: {required - set(df.columns)}")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    rsi_series = rsi(close, period=rsi_period)
    atr_series = atr(df, period=atr_period)
    pivot_highs, pivot_lows = find_pivots(high, low, left=pivot_left, right=pivot_right)

    signals: List[Dict] = []

    # Bullish divergence
    for i in range(1, len(pivot_lows)):
        l2 = pivot_lows[i]
        # look back to find previous pivot low within lookback
        candidates = [idx for idx in pivot_lows[:i] if l2 - idx <= lookback_bars]
        if not candidates:
            continue
        l1 = candidates[-1]
        if low[l2] >= low[l1]:
            continue
        if rsi_series[l2] <= oversold and (rsi_series[l2] - rsi_series[l1]) >= min_rsi_diff:
            entry_idx = l2
            entry = close[entry_idx]
            if sl_mode == "swing":
                sl = low[l2]
            else:
                sl = entry - k_sl * atr_series[entry_idx]
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + tp_R * risk
            signals.append(
                {"idx": entry_idx, "side": "long", "entry": entry, "sl": sl, "tp": tp}
            )

    # Bearish divergence
    for i in range(1, len(pivot_highs)):
        h2 = pivot_highs[i]
        candidates = [idx for idx in pivot_highs[:i] if h2 - idx <= lookback_bars]
        if not candidates:
            continue
        h1 = candidates[-1]
        if high[h2] <= high[h1]:
            continue
        if rsi_series[h2] >= overbought and (rsi_series[h1] - rsi_series[h2]) >= min_rsi_diff:
            entry_idx = h2
            entry = close[entry_idx]
            if sl_mode == "swing":
                sl = high[h2]
            else:
                sl = entry + k_sl * atr_series[entry_idx]
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - tp_R * risk
            signals.append(
                {"idx": entry_idx, "side": "short", "entry": entry, "sl": sl, "tp": tp}
            )

    signals.sort(key=lambda x: x["idx"])
    return signals
