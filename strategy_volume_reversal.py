#!/usr/bin/env python3
"""
策略：低量区间后的放量大K，收盘价开仓，50% 反转止盈，外侧一倍 K 高度止损，冷却期 + 超时平仓。
"""

from typing import List, Dict, Optional

import numpy as np
import pandas as pd

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _maybe_tqdm(iterable, enable: bool, total: Optional[int] = None):
    if not enable:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total)


def _validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列: {missing}")


def detect_signals(
    df: pd.DataFrame,
    quiet_lookback: int = 20,
    vol_spike_mult: float = 2.0,
    quiet_max_mult: float = 1.2,
    body_mult: float = 1.5,
    sl_mode: str = "outer_bar",
    show_progress: bool = False,
) -> List[Dict]:
    """
    返回信号列表，每个元素包含 idx/side/entry/sl/tp/mid/height
    """
    _validate_df(df)
    if quiet_lookback <= 0:
        raise ValueError("quiet_lookback 必须 > 0")
    prices = df[["open", "high", "low", "close", "volume"]].to_numpy()
    bodies = np.abs(prices[:, 3] - prices[:, 0])  # close - open

    signals: List[Dict] = []
    n = len(df)
    iterator = _maybe_tqdm(range(quiet_lookback, n), enable=show_progress, total=n - quiet_lookback)
    for i in iterator:
        prev_slice = slice(i - quiet_lookback, i)
        vol_mean_prev = prices[prev_slice, 4].mean()
        vol_max_prev = prices[prev_slice, 4].max()
        body_mean_prev = bodies[prev_slice].mean()

        if vol_max_prev > vol_mean_prev * quiet_max_mult:
            continue

        volume_i = prices[i, 4]
        body_i = bodies[i]
        if volume_i < vol_mean_prev * vol_spike_mult:
            continue
        if body_i < body_mean_prev * body_mult:
            continue

        open_i, high_i, low_i, close_i = prices[i, 0], prices[i, 1], prices[i, 2], prices[i, 3]
        if close_i == open_i:
            continue

        side = "short" if close_i > open_i else "long"
        mid_i = (high_i + low_i) / 2
        height_i = high_i - low_i
        if height_i <= 0:
            continue
        if sl_mode == "outer_bar":
            if side == "long":
                sl = low_i - height_i
            else:
                sl = high_i + height_i
        else:
            raise ValueError(f"未知的 sl_mode: {sl_mode}")

        entry = close_i
        tp = mid_i
        signals.append(
            {
                "idx": i,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "mid": mid_i,
                "height": height_i,
            }
        )
    return signals


def simulate_trades(
    df: pd.DataFrame,
    signals: List[Dict],
    max_holding_bars: int = 20,
    cooldown_bars: int = 20,
    show_progress: bool = False,
) -> List[Dict]:
    """
    基于信号和 K 线模拟交易，返回每笔交易明细
    """
    _validate_df(df)
    prices = df[["high", "low", "close"]].to_numpy()
    trades: List[Dict] = []
    ignore_until = -1
    last_index = len(df) - 1

    iterator = _maybe_tqdm(signals, enable=show_progress, total=len(signals))
    for sig in iterator:
        idx = sig["idx"]
        if idx >= last_index:
            break
        if idx <= ignore_until:
            continue

        side = sig["side"]
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]

        exit_idx: Optional[int] = None
        outcome: Optional[str] = None
        exit_price: Optional[float] = None
        hit_sl = False
        hit_tp = False

        end_idx = min(last_index, idx + max_holding_bars)
        for bar in range(idx + 1, end_idx + 1):
            high_b, low_b, close_b = prices[bar]
            if side == "long":
                hit_sl = low_b <= sl
                hit_tp = high_b >= tp
            else:
                hit_sl = high_b >= sl
                hit_tp = low_b <= tp

            if hit_sl:
                exit_idx = bar
                exit_price = sl
                outcome = "sl"
                ignore_until = bar + cooldown_bars
                break
            if hit_tp:
                exit_idx = bar
                exit_price = tp
                outcome = "tp"
                break

        if exit_idx is None:
            exit_idx = end_idx
            exit_price = prices[end_idx][2]  # close
            outcome = "timeout"

        if side == "long":
            risk = entry - sl
            pnl = exit_price - entry
        else:
            risk = sl - entry
            pnl = entry - exit_price
        R = pnl / risk if risk > 0 else np.nan

        trades.append(
            {
                "signal_idx": idx,
                "entry_idx": idx,
                "exit_idx": exit_idx,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit_price": exit_price,
                "outcome": outcome,
                "R": R,
            }
        )
    return trades
