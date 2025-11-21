#!/usr/bin/env python3
"""
趋势突破策略：
- 1h 主周期，EMA 趋势过滤 + Donchian 突破 + ATR/放量/波动过滤
- ATR 止损，固定 TP (R_target) + 移动止损 (k_trail)
- 可选下级周期判定同根 TP/SL 先后
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
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
    return tr.rolling(period, min_periods=period).mean()


def detect_breakouts(
    df: pd.DataFrame,
    ema_span: int = 100,
    donchian_n: int = 24,
    atr_period: int = 20,
    k_buffer: float = 0.1,
    vol_lookback: int = 20,
    vol_mult: float = 1.5,
    atr_median_lookback: int = 100,
) -> List[Dict]:
    """
    返回突破信号列表：idx/side/entry/atr/sl_info
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必要列: {required - set(df.columns)}")

    ema_trend = ema(df["close"], span=ema_span)
    atr_series = atr(df, atr_period)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    vols = df["volume"].to_numpy()
    atr_vals = atr_series.to_numpy()
    ema_vals = ema_trend.to_numpy()

    signals: List[Dict] = []
    n = len(df)
    start_idx = max(donchian_n, atr_period, atr_median_lookback) + 1

    for i in range(start_idx, n):
        atr_i = atr_vals[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        ema_i = ema_vals[i]
        close_i = closes[i]
        high_i = highs[i]
        low_i = lows[i]

        # 趋势方向
        if close_i > ema_i:
            dir_bias = "long"
        elif close_i < ema_i:
            dir_bias = "short"
        else:
            continue

        # Donchian 通道（不含当前）
        hh = highs[i - donchian_n : i].max()
        ll = lows[i - donchian_n : i].min()
        buffer = k_buffer * atr_i

        # 波动/放量过滤
        atr_window = atr_vals[i - atr_median_lookback : i]
        if np.isnan(atr_window).any():
            continue
        atr_median = np.median(atr_window)
        if atr_i <= atr_median:
            continue

        vol_mean = vols[i - vol_lookback : i].mean()
        if vol_mean <= 0 or vols[i] < vol_mean * vol_mult:
            continue

        if dir_bias == "long" and close_i > hh + buffer:
            side = "long"
        elif dir_bias == "short" and close_i < ll - buffer:
            side = "short"
        else:
            continue

        signals.append(
            {
                "idx": i,
                "side": side,
                "entry": close_i,
                "atr": atr_i,
            }
        )
    return signals


def simulate_trades(
    df: pd.DataFrame,
    signals: List[Dict],
    k_sl: float = 1.5,
    R_target: float = 3.0,
    k_trail: float = 2.0,
    fee_side: float = 0.000248,
    lower_df: Optional[pd.DataFrame] = None,
    upper_interval_sec: int = 3600,
    lower_interval_sec: int = 60,
    stop_loss_streak: int = 0,
    stop_duration_days: int = 0,
) -> List[Dict]:
    """
    固定 TP (R_target*ATR*k_sl) + 移动止损 (k_trail*ATR)
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必要列: {required - set(df.columns)}")
    prices = df[["high", "low", "close"]].to_numpy()
    ts_upper = pd.to_datetime(df["timestamp"], utc=True).to_numpy()

    lower_data = None
    if lower_df is not None:
        lower_data = {
            "ts": pd.to_datetime(lower_df["timestamp"], utc=True).to_numpy(),
            "high": lower_df["high"].to_numpy(),
            "low": lower_df["low"].to_numpy(),
        }

    trades: List[Dict] = []
    last_index = len(df) - 1
    fee_round = fee_side * 2.0
    block_until = None
    loss_streak = 0

    def resolve_conflict(side: str, sl: float, tp: float, start_ts, end_ts):
        if lower_data is None:
            return "sl", sl  # 保守
        mask = (lower_data["ts"] >= start_ts) & (lower_data["ts"] < end_ts)
        if not mask.any():
            return "sl", sl
        highs = lower_data["high"][mask]
        lows = lower_data["low"][mask]
        for h, l in zip(highs, lows):
            if side == "long":
                hit_sl = l <= sl
                hit_tp = h >= tp
            else:
                hit_sl = h >= sl
                hit_tp = l <= tp
            if hit_sl and not hit_tp:
                return "sl", sl
            if hit_tp and not hit_sl:
                return "tp", tp
            if hit_sl and hit_tp:
                return "sl", sl  # 同根仍保守
        return "sl", sl

    for sig in signals:
        idx = sig["idx"]
        if idx >= last_index:
            continue
        ts_entry = ts_upper[idx]
        if block_until is not None and ts_entry < block_until:
            continue
        side = sig["side"]
        entry = sig["entry"]
        atr_i = sig["atr"]
        risk_abs = k_sl * atr_i
        if risk_abs <= 0:
            continue
        if side == "long":
            sl = entry - risk_abs
            tp = entry + R_target * risk_abs
        else:
            sl = entry + risk_abs
            tp = entry - R_target * risk_abs

        trail_sl = sl
        highest_close = entry
        lowest_close = entry
        exit_idx = None
        exit_price = None
        outcome = None

        for bar in range(idx + 1, last_index + 1):
            high_b, low_b, close_b = prices[bar]
            # 更新 trailing
            if side == "long":
                highest_close = max(highest_close, close_b)
                trail_sl = max(trail_sl, highest_close - k_trail * atr_i)
            else:
                lowest_close = min(lowest_close, close_b)
                trail_sl = min(trail_sl, lowest_close + k_trail * atr_i)

            # 先判断 SL/TP，包含 trail
            if side == "long":
                hit_sl = low_b <= min(sl, trail_sl)
                hit_tp = high_b >= tp
            else:
                hit_sl = high_b >= max(sl, trail_sl)
                hit_tp = low_b <= tp

            if hit_sl and hit_tp:
                start_ts = ts_upper[idx]
                end_ts = start_ts + pd.Timedelta(seconds=upper_interval_sec)
                sub_outcome, sub_price = resolve_conflict(side, min(sl, trail_sl) if side == "long" else max(sl, trail_sl), tp, start_ts, end_ts)
                outcome = sub_outcome
                exit_price = sub_price
                exit_idx = bar
                break
            if hit_sl:
                outcome = "sl"
                exit_price = min(sl, trail_sl) if side == "long" else max(sl, trail_sl)
                exit_idx = bar
                break
            if hit_tp:
                outcome = "tp"
                exit_price = tp
                exit_idx = bar
                break

        if exit_idx is None:
            # 未触发，最后一根收盘平仓
            exit_idx = last_index
            exit_price = prices[last_index][2]
            outcome = "timeout"

        pnl_abs = (exit_price - entry) if side == "long" else (entry - exit_price)
        raw_R = pnl_abs / risk_abs
        risk_pct = risk_abs / entry
        fee_R = fee_round / risk_pct if risk_pct > 0 else 0.0
        net_R = raw_R - fee_R

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
                "raw_R": raw_R,
                "fee_R": fee_R,
                "net_R": net_R,
                "risk_pct": risk_pct,
            }
        )
        # 连亏控制
        if net_R < 0:
            loss_streak += 1
        else:
            loss_streak = 0
        if stop_loss_streak > 0 and stop_duration_days > 0 and loss_streak >= stop_loss_streak:
            block_until = ts_upper[exit_idx] + pd.Timedelta(days=stop_duration_days)
            loss_streak = 0
    return trades
