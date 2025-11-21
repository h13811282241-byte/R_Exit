#!/usr/bin/env python3
"""
策略：低量区间后的放量大K 50% 反转，收盘价开仓，外侧一倍 K 高度止损，冷却 + 超时平仓。
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _validate_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必需列: {missing}")


def detect_signals(
    df: pd.DataFrame,
    quiet_lookback: int = 20,
    vol_spike_mult: float = 2.0,
    quiet_max_mult: float = 0.0,
    body_mult: float = 1.5,
    sl_mode: str = "outer_bar",
    tp_ratio: float = 0.5,
    sl_offset_ratio: float = 1.0,
    tp_ref: str = "prev",  # "signal" 或 "prev"
    sl_ref: str = "prev",
    invert_side: bool = False,
) -> List[Dict]:
    """
    返回信号列表：idx/side/entry/sl/tp/mid/height
    """
    _validate_df(df)
    if quiet_lookback <= 0:
        raise ValueError("quiet_lookback 必须 > 0")
    # tp_ratio/sl_offset_ratio 允许大于 1，表示超出一倍K高
    tp_ratio = max(0.0, tp_ratio)
    sl_offset_ratio = max(0.0, sl_offset_ratio)
    if tp_ref not in {"signal", "prev"}:
        tp_ref = "prev"
    if sl_ref not in {"signal", "prev"}:
        sl_ref = "prev"

    prices = df[["open", "high", "low", "close", "volume"]].to_numpy()
    bodies = np.abs(prices[:, 3] - prices[:, 0])

    signals: List[Dict] = []
    n = len(df)
    for i in range(quiet_lookback, n):
        prev_slice = slice(i - quiet_lookback, i)
        vol_mean_prev = prices[prev_slice, 4].mean()
        vol_max_prev = prices[prev_slice, 4].max()
        body_mean_prev = bodies[prev_slice].mean()

        # quiet_max_mult <=0 表示不限制安静期最大量
        if quiet_max_mult > 0 and vol_max_prev > vol_mean_prev * quiet_max_mult:
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
        if invert_side:
            side = "long" if side == "short" else "short"
        height_i = high_i - low_i
        if height_i <= 0:
            continue

        # 参考高度：signal 本身，或前一根
        ref_h_tp = height_i
        ref_h_sl = height_i
        if i > 0 and tp_ref == "prev":
            h_prev = prices[i - 1, 1] - prices[i - 1, 2]  # high - low
            ref_h_tp = h_prev if h_prev > 0 else height_i
        if i > 0 and sl_ref == "prev":
            h_prev = prices[i - 1, 1] - prices[i - 1, 2]
            ref_h_sl = h_prev if h_prev > 0 else height_i

        tp_dist = ref_h_tp * tp_ratio
        sl_dist = ref_h_sl * sl_offset_ratio
        if tp_dist <= 0 or sl_dist <= 0:
            continue

        entry = close_i
        if side == "long":
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:
            tp = entry - tp_dist
            sl = entry + sl_dist

        signals.append(
            {
                "idx": i,
                "side": side,
                "entry": close_i,
                "sl": sl,
                "tp": tp,
                "mid": (high_i + low_i) / 2,
                "height": height_i,
            }
        )
    return signals


def simulate_trades(
    df: pd.DataFrame,
    signals: List[Dict],
    max_holding_bars: int = 20,
    cooldown_bars: int = 20,
) -> List[Dict]:
    """
    基于信号在 K 线模拟交易，返回每笔交易明细。
    """
    _validate_df(df)
    prices = df[["high", "low", "close"]].to_numpy()
    trades: List[Dict] = []
    ignore_until = -1
    last_index = len(df) - 1

    for sig in signals:
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
