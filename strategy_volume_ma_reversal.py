#!/usr/bin/env python3
"""
5m 放量反转策略（量20MA过滤，影线限制，非美股盘时段），基于收盘价入场，实体1倍止盈止损。
"""

from typing import List, Dict
import pandas as pd
import numpy as np
import pytz

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列: {missing}")


def _is_us_session(ts_utc) -> bool:
    """
    判断时间是否落在美股开盘时段(纽约 9:30-16:00)，用于过滤信号。
    """
    ts_ny = ts_utc.tz_convert("America/New_York")
    t = ts_ny.time()
    return (t.hour > 9 or (t.hour == 9 and t.minute >= 30)) and (t.hour < 16)


def detect_volume_ma_signals(
    df: pd.DataFrame,
    vol_ma: int = 20,
    vol_mult: float = 2.0,
    shadow_ratio: float = 2.0 / 3.0,
    side_mode: str = "reversal",  # "reversal" = 反向K体方向, "follow" = 顺向
    exclude_us_session: bool = True,
) -> List[Dict]:
    """
    筛选信号：当前成交量 > vol_ma*vol_mult，影线不超过实体的 shadow_ratio，
   （可选）过滤美股开盘时段，返回 {idx, side, entry, sl, tp}.
    """
    _validate_df(df)
    if vol_ma <= 0:
        raise ValueError("vol_ma 必须 > 0")
    side_mode = side_mode if side_mode in {"reversal", "follow"} else "reversal"

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.copy()
    df["ts_utc"] = ts
    df = df.dropna(subset=["ts_utc"])

    prices = df[["open", "high", "low", "close", "volume"]].to_numpy()
    vols = prices[:, 4]
    vol_ma_series = pd.Series(vols).rolling(vol_ma, min_periods=vol_ma).mean().to_numpy()

    signals: List[Dict] = []
    for i in range(len(df)):
        if np.isnan(vol_ma_series[i]):
            continue
        if exclude_us_session and _is_us_session(df.iloc[i]["ts_utc"]):
            continue
        o, h, l, c, v = prices[i]
        body = abs(c - o)
        if body <= 0:
            continue
        upper = h - max(o, c)
        lower = min(o, c) - l
        if upper > body * shadow_ratio or lower > body * shadow_ratio:
            continue
        if v <= vol_ma_series[i] * vol_mult:
            continue

        # 方向：默认反向K体
        if c > o:
            side = "short" if side_mode == "reversal" else "long"
        elif c < o:
            side = "long" if side_mode == "reversal" else "short"
        else:
            continue

        entry = c
        tp = entry + body if side == "long" else entry - body
        sl = entry - body if side == "long" else entry + body

        signals.append({"idx": i, "side": side, "entry": entry, "sl": sl, "tp": tp})
    return signals
