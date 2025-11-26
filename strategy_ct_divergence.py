#!/usr/bin/env python3
"""
4H 下跌段内的底背离对冲策略（B1/B2 + 6根站上确认）。
基于 5m 数据，先聚合 4H 定义下跌段，再在 5m 检测底背离，生成信号。
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from indicators import rsi, atr, find_pivots


@dataclass
class CTDivergenceConfig:
    rsi_period: int = 14
    oversold: float = 30.0
    lookback_bars: int = 50  # 5m bars for divergence pairing
    pivot_left: int = 2
    pivot_right: int = 0  # no future data
    min_rsi_diff: float = 3.0
    atr_period: int = 14
    k_sl: float = 1.5
    tp_R: float = 1.5
    # 4H 下跌段定义
    n4: int = 4
    min_bearish_count: int = 3
    downleg_gap_bars_5m: int = 168  # 14h = 14*12
    # 站上确认
    confirm_bars: int = 6
    confirm_min_pct: float = 0.001
    # 回踩触发
    retest_tol_pct: float = 0.001


def aggregate_4h(df: pd.DataFrame) -> pd.DataFrame:
    df_ts = df.copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"], utc=True)
    df_ts = df_ts.set_index("timestamp")
    agg = df_ts.resample("4H").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    return agg.reset_index()


def compute_down_leg_4h(df_4h: pd.DataFrame, cfg: CTDivergenceConfig) -> pd.Series:
    close = df_4h["close"].to_numpy()
    open_ = df_4h["open"].to_numpy()
    flags = np.zeros(len(df_4h), dtype=bool)
    for i in range(cfg.n4 - 1, len(df_4h)):
        window_close = close[i - cfg.n4 + 1 : i + 1]
        window_open = open_[i - cfg.n4 + 1 : i + 1]
        bearish = np.sum(window_close < window_open)
        cond1 = bearish >= cfg.min_bearish_count
        cond2 = close[i] < close[i - cfg.n4 + 1]
        flags[i] = cond1 and cond2
    return pd.Series(flags, index=df_4h.index)


def map_4h_flag_to_5m(df_5m: pd.DataFrame, flag_4h: pd.Series, ts_4h: pd.Series) -> pd.Series:
    """
    Map 4H flag to 5m rows using timestamp bucket.
    """
    ts5 = pd.to_datetime(df_5m["timestamp"], utc=True)
    bucket = ts5.dt.floor("4H")
    flag_map = dict(zip(ts_4h, flag_4h))
    return bucket.map(flag_map).fillna(False)


def has_stood_above(level: float, closes: pd.Series, start_idx: int, confirm_bars: int, min_pct: float) -> Tuple[bool, Optional[int]]:
    count = 0
    for i in range(start_idx, len(closes)):
        if closes.iloc[i] > level * (1 + min_pct):
            count += 1
            if count >= confirm_bars:
                return True, i
        else:
            count = 0
    return False, None


def detect_ct_signals(df: pd.DataFrame, cfg: CTDivergenceConfig) -> List[Dict]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    rsi_series = rsi(df["close"], period=cfg.rsi_period)
    atr_series = atr(df, period=cfg.atr_period)
    pivot_highs, pivot_lows = find_pivots(df["high"], df["low"], left=cfg.pivot_left, right=cfg.pivot_right)

    # 4H 下跌段标记
    df_4h = aggregate_4h(df)
    down_flags_4h = compute_down_leg_4h(df_4h, cfg)
    flag_5m = map_4h_flag_to_5m(df, down_flags_4h, df_4h["timestamp"])
    df["is_down_leg_4h"] = flag_5m

    signals: List[Dict] = []
    pending_b1 = None  # (idx, entry_level, sl)
    last_div_idx = None
    last_div_4h_down = False

    # helper: next pivot high after idx
    pivot_highs_set = set(pivot_highs)
    pivot_highs_sorted = sorted(pivot_highs)

    def next_pivot_high_level(after_idx: int) -> Optional[Tuple[int, float]]:
        for ph in pivot_highs_sorted:
            if ph > after_idx:
                return ph, float(df.at[ph, "high"])
        return None

    for i in range(1, len(pivot_lows)):
        l2 = pivot_lows[i]
        if not df.at[l2, "is_down_leg_4h"]:
            pending_b1 = None
            last_div_idx = None
            continue
        candidates = [idx for idx in pivot_lows[:i] if l2 - idx <= cfg.lookback_bars]
        if not candidates:
            continue
        l1 = candidates[-1]
        if df.at[l2, "low"] >= df.at[l1, "low"]:
            continue
        if rsi_series.iloc[l2] <= cfg.oversold and (rsi_series.iloc[l2] - rsi_series.iloc[l1]) >= cfg.min_rsi_diff:
            # divergence found
            if last_div_idx is None or (l2 - last_div_idx) > cfg.downleg_gap_bars_5m or not last_div_4h_down:
                # new B1
                nh = next_pivot_high_level(l2)
                if nh is None:
                    continue
                ph_idx, entry_level = nh
                sl = float(df.at[l2, "low"])
                pending_b1 = {"idx": l2, "entry_level": entry_level, "sl": sl}
                last_div_idx = l2
                last_div_4h_down = True
                continue
            else:
                # B2+
                last_div_idx = l2
                last_div_4h_down = True
                if pending_b1 is None:
                    nh = next_pivot_high_level(l2)
                    if nh is None:
                        continue
                    ph_idx, entry_level = nh
                    pending_b1 = {"idx": l2, "entry_level": entry_level, "sl": float(df.at[l2, "low"])}
                # 检查站上 + 回踩
                entry_level = pending_b1["entry_level"]
                sl = pending_b1["sl"]
                stood, confirm_idx = has_stood_above(
                    entry_level, df["close"], l2, cfg.confirm_bars, cfg.confirm_min_pct
                )
                if not stood or confirm_idx is None:
                    continue
                # 回踩触发
                trigger_idx = None
                for j in range(confirm_idx + 1, len(df)):
                    if df.at[j, "low"] <= entry_level * (1 + cfg.retest_tol_pct) and df.at[j, "close"] >= entry_level:
                        trigger_idx = j
                        break
                if trigger_idx is None:
                    continue
                risk = entry_level - sl
                if risk <= 0:
                    continue
                tp = entry_level + cfg.tp_R * risk
                signals.append(
                    {
                        "idx": trigger_idx,
                        "side": "long",
                        "entry": entry_level,
                        "sl": sl,
                        "tp": tp,
                        "meta": {
                            "b1_idx": pending_b1["idx"],
                            "b2_idx": l2,
                            "entry_level": entry_level,
                            "confirm_idx": confirm_idx,
                        },
                    }
                )
                pending_b1 = None  # one trade per down-leg
    return signals
