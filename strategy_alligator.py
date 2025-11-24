#!/usr/bin/env python3
"""
Bill Williams Alligator 趋势策略：
- lips/teeth/jaw 多空排列确认趋势
- 入场：趋势确认后收盘开仓
- 止损：结构位或 ATR
- 止盈：固定 R 或 反向信号/缠绕退出
"""

from typing import List, Dict
import pandas as pd

from indicators import alligator, atr, find_pivots


def detect_alligator_signals(
    df: pd.DataFrame,
    jaw_period: int = 13,
    teeth_period: int = 8,
    lips_period: int = 5,
    trend_confirm_bars: int = 3,
    entry_fresh_bars: int = 5,
) -> List[Dict]:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必要列: {required - set(df.columns)}")
    close = df["close"]
    ag = alligator(close, jaw_period, teeth_period, lips_period)
    jaw = ag["jaw"]
    teeth = ag["teeth"]
    lips = ag["lips"]

    signals: List[Dict] = []
    trend_state = None
    trend_start_idx = None

    for i in range(len(df)):
        if pd.isna(jaw.iloc[i]) or pd.isna(teeth.iloc[i]) or pd.isna(lips.iloc[i]):
            continue
        bullish = lips.iloc[i] > teeth.iloc[i] > jaw.iloc[i] and close.iloc[i] > lips.iloc[i]
        bearish = lips.iloc[i] < teeth.iloc[i] < jaw.iloc[i] and close.iloc[i] < lips.iloc[i]

        if bullish:
            if trend_state != "bull":
                trend_state = "bull"
                trend_start_idx = i
            if trend_start_idx is not None and (i - trend_start_idx) >= trend_confirm_bars and (i - trend_start_idx) <= trend_confirm_bars + entry_fresh_bars:
                signals.append({"idx": i, "side": "long", "entry": close.iloc[i]})
        elif bearish:
            if trend_state != "bear":
                trend_state = "bear"
                trend_start_idx = i
            if trend_start_idx is not None and (i - trend_start_idx) >= trend_confirm_bars and (i - trend_start_idx) <= trend_confirm_bars + entry_fresh_bars:
                signals.append({"idx": i, "side": "short", "entry": close.iloc[i]})
        else:
            trend_state = None
            trend_start_idx = None

    signals.sort(key=lambda x: x["idx"])
    return signals


def prepare_sl_tp(
    df: pd.DataFrame,
    signals: List[Dict],
    sl_mode: str = "atr",
    atr_period: int = 14,
    k_sl: float = 1.5,
    tp_R: float = 2.0,
) -> List[Dict]:
    """
    根据 sl_mode/tp_R 填充 sl/tp
    """
    atr_series = atr(df, period=atr_period)
    high = df["high"]
    low = df["low"]
    pivot_highs, pivot_lows = find_pivots(high, low, left=2, right=2)
    pivot_high_set = set(pivot_highs)
    pivot_low_set = set(pivot_lows)

    out = []
    for s in signals:
        idx = s["idx"]
        entry = s["entry"]
        side = s["side"]
        if sl_mode == "atr":
            risk = k_sl * atr_series.iloc[idx]
            if side == "long":
                sl = entry - risk
                tp = entry + tp_R * risk
            else:
                sl = entry + risk
                tp = entry - tp_R * risk
        else:  # swing
            if side == "long":
                # 最近 pivot low
                candidates = [p for p in pivot_low_set if p < idx]
                if not candidates:
                    continue
                pl = max(candidates)
                sl = low.iloc[pl]
                risk = entry - sl
                tp = entry + tp_R * risk
            else:
                candidates = [p for p in pivot_high_set if p < idx]
                if not candidates:
                    continue
                ph = max(candidates)
                sl = high.iloc[ph]
                risk = sl - entry
                tp = entry - tp_R * risk
        risk_abs = entry - sl if side == "long" else sl - entry
        if risk_abs <= 0:
            continue
        s2 = dict(s)
        s2.update({"sl": sl, "tp": tp})
        out.append(s2)
    return out
