#!/usr/bin/env python3
"""
1h EMA220 趋势突破（多头版）：结构+放量+距离过滤，分批止盈+尾仓跟踪。
"""
from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def swing_lows(df: pd.DataFrame, left: int = 2, right: int = 2) -> List[int]:
    lows = df["low"].to_numpy()
    idxs = []
    n = len(df)
    for i in range(left, n - right):
        if lows[i] == lows[i - left : i + right + 1].min():
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                idxs.append(i)
    return idxs


def detect_signals(
    df: pd.DataFrame,
    ema_len: int = 220,
    atr_len: int = 14,
    lookback_box: int = 80,
    vol_ma_len: int = 20,
    volume_factor: float = 1.0,
    touch_atr: float = 0.1,
    near_high_frac: float = 0.2,
    dist_min: float = 0.0,
    dist_max: float = 10.0,
    slope_lookback: int = 30,
    slope_thresh: float = 0.0,
    env_window: int = 50,
    env_band_atr: float = 0.5,
    env_ratio_max: float = 0.7,
    enable_short: bool = True,
    trend_filter_mode: str = "none",  # "none" | "ema_gate" | "no_cross"
    ema_gate_len: int = 220,
    cross_lookback: int = 30,
) -> List[Dict]:
    """
    返回多空信号：
    {
      idx, side, entry, sl, tp1, tp2, box_edge, atr, ema220, risk
    }
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必需列: {required - set(df.columns)}")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    ema220 = ema(close, ema_len)
    ema_gate = ema(close, ema_gate_len)
    atrv = atr(df, atr_len)
    vol_ma = volume.rolling(vol_ma_len).mean()

    signals: List[Dict] = []
    for i in range(max(lookback_box, ema_len, atr_len, vol_ma_len, ema_gate_len), len(df)):
        # 趋势过滤：4h220ema上只做多，下只做空；频繁交叉则不做
        if trend_filter_mode in {"ema_gate", "no_cross"}:
            price_above = close.iloc[i] > ema_gate.iloc[i]
            if trend_filter_mode == "ema_gate":
                allow_long = price_above
                allow_short = (not price_above) and enable_short
            else:  # no_cross
                recent = close.iloc[i - cross_lookback : i]
                ema_recent = ema_gate.iloc[i - cross_lookback : i]
                crosses = ((recent > ema_recent) != (recent.shift(1) > ema_recent.shift(1))).sum()
                if crosses >= 3:
                    allow_long = allow_short = False
                else:
                    allow_long = price_above
                    allow_short = (not price_above) and enable_short
        else:
            allow_long = True
            allow_short = enable_short
        atr_i = atrv.iloc[i]
        if atr_i <= 0 or pd.isna(atr_i):
            continue
        ema_i = ema220.iloc[i]
        env_slice = close.iloc[i - env_window : i]
        env_band = atr_i * env_band_atr
        env_ratio = ((env_slice >= ema_i - env_band) & (env_slice <= ema_i + env_band)).mean()
        if env_ratio > env_ratio_max:
            continue
        box_high = high.iloc[i - lookback_box : i].max()
        box_low = low.iloc[i - lookback_box : i].min()
        box_range = box_high - box_low
        if not (1.0 * atr_i <= box_range <= 8 * atr_i):
            continue
        close_i = close.iloc[i]
        high_i = high.iloc[i]
        low_i = low.iloc[i]
        vol_i = volume.iloc[i]
        dist = abs(close_i - ema_i) / atr_i
        if dist < dist_min or dist > dist_max:
            continue
        if slope_lookback > 0 and slope_thresh > 0:
            if ema_i <= ema220.iloc[i - slope_lookback] * (1 + slope_thresh):
                ema_up = False
            else:
                ema_up = True
        else:
            ema_up = close_i > ema_i
        # 多头
        if close_i > ema_i and ema_up and allow_long:
            near_high = (high.iloc[i - lookback_box : i] >= (box_high - touch_atr * atr_i)).sum()
            if near_high < 1:
                pass
            else:
                if close_i > box_high + touch_atr * atr_i and close_i > high_i - near_high_frac * (high_i - low_i) and vol_i > vol_ma.iloc[i] * volume_factor:
                    entry = close_i
                    sl = box_low - touch_atr * atr_i
                    risk = entry - sl
                    if risk > 0:
                        tp1 = entry + 2 * risk
                        tp2 = entry + 4 * risk
                        signals.append(
                            {
                                "idx": i,
                                "side": "long",
                                "entry": entry,
                                "sl": sl,
                                "tp1": tp1,
                                "tp2": tp2,
                                "box_edge": box_low,
                                "atr": atr_i,
                                "ema220": ema_i,
                                "risk": risk,
                            }
                        )
        # 空头
        if allow_short and close_i < ema_i:
            near_low = (low.iloc[i - lookback_box : i] <= (box_low + touch_atr * atr_i)).sum()
            if near_low < 1:
                continue
            if close_i < box_low - touch_atr * atr_i and close_i < low_i + near_high_frac * (high_i - low_i) and vol_i > vol_ma.iloc[i] * volume_factor:
                entry = close_i
                sl = box_high + touch_atr * atr_i
                risk = sl - entry
                if risk > 0:
                    tp1 = entry - 2 * risk
                    tp2 = entry - 4 * risk
                    signals.append(
                        {
                            "idx": i,
                            "side": "short",
                            "entry": entry,
                            "sl": sl,
                            "tp1": tp1,
                            "tp2": tp2,
                            "box_edge": box_high,
                            "atr": atr_i,
                            "ema220": ema_i,
                            "risk": risk,
                        }
                    )
    return signals


def simulate_trades(
    df: pd.DataFrame,
    signals: List[Dict],
    fee_side_pct: float = 0.00045,
    trailing_mode: str = "ema50",  # "ema50" 或 "swing"
    swing_lookback: int = 5,
) -> List[Dict]:
    """
    分批止盈：tp1 50%，tp2 25%，尾仓25% 跟踪止损。支持多空。
    按实际持仓权重计算 R，止损命中时全仓按 -1R 计，不再缩减。
    """
    if df.empty or not signals:
        return []
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ema50 = ema(df["close"], 50).to_numpy()

    trades: List[Dict] = []
    fee_round = fee_side_pct * 2
    last_idx = len(df) - 1
    for sig in signals:
        i = sig["idx"]
        if i >= last_idx:
            continue
        side = sig.get("side", "long")
        entry = sig["entry"]
        sl = sig["sl"]
        tp1 = sig["tp1"]
        tp2 = sig["tp2"]
        risk = sig["risk"]
        filled_tp1 = False
        filled_tp2 = False
        realized_R = 0.0
        remaining = 1.0
        exit_idx = None
        exit_price = None
        trail = None

        for b in range(i + 1, last_idx + 1):
            h = high[b]
            l = low[b]
            c = close[b]
            if side == "long":
                hit_sl = l <= sl
                hit_tp1 = h >= tp1
                hit_tp2 = h >= tp2
            else:
                hit_sl = h >= sl
                hit_tp1 = l <= tp1
                hit_tp2 = l <= tp2

            # 止损优先（若未触达任何TP）
            if not filled_tp1 and hit_sl:
                exit_idx = b
                exit_price = sl
                realized_R += remaining * ((exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk)
                remaining = 0
                break

            # TP1
            if (not filled_tp1) and hit_tp1:
                realized_R += 0.5 * ((tp1 - entry) / risk if side == "long" else (entry - tp1) / risk)
                remaining -= 0.5
                filled_tp1 = True

            # 止损在已分仓后
            if filled_tp1 and (not filled_tp2) and hit_sl:
                exit_idx = b
                exit_price = sl
                realized_R += remaining * ((exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk)
                remaining = 0
                break

            # TP2
            if filled_tp1 and (not filled_tp2) and hit_tp2:
                realized_R += 0.25 * ((tp2 - entry) / risk if side == "long" else (entry - tp2) / risk)
                remaining -= 0.25
                filled_tp2 = True
                # 初始化尾仓trail
                if trailing_mode == "ema50":
                    trail = ema50[b]
                else:
                    start = max(0, b - swing_lookback)
                    trail = low[start:b].min() if side == "long" else high[start:b].max()

            # 尾仓跟踪
            if filled_tp2 and remaining > 0:
                if trailing_mode == "ema50":
                    trail = ema50[b]
                else:
                    start = max(0, b - swing_lookback)
                    trail = low[start:b].min() if side == "long" else high[start:b].max()
                if (side == "long" and c <= trail) or (side == "short" and c >= trail):
                    exit_idx = b
                    exit_price = c
                    realized_R += remaining * ((exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk)
                    remaining = 0
                    break

        # 收盘强平尾仓
        if remaining > 0:
            exit_idx = exit_idx if exit_idx is not None else last_idx
            exit_price = exit_price if exit_price is not None else close[last_idx]
            realized_R += remaining * ((exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk)

        risk_pct = risk / entry if entry else 0
        fee_R = fee_round / risk_pct if risk_pct else 0.0
        trades.append(
            {
                "entry_time": df.loc[i, "timestamp"],
                "exit_time": df.loc[exit_idx, "timestamp"] if exit_idx is not None else df.loc[last_idx, "timestamp"],
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "exit_price": exit_price,
                "raw_R": realized_R,
                "fee_R": fee_R,
                "net_R": realized_R - fee_R,
                "entry_idx": i,
                "exit_idx": exit_idx if exit_idx is not None else last_idx,
            }
        )
    return trades
