#!/usr/bin/env python3
"""
熊市顺势空单 + 对冲多单（底背离对冲）组合策略。
空单为主：4H 空头趋势下，顶背离破位做空，或反弹失败破前低做空。
对冲多单为辅：仅在已有空单信号后才允许触发，复用 CT 底背离逻辑。
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from indicators import rsi, find_pivots
from strategy_ct_divergence import detect_ct_signals, CTDivergenceConfig
from trend_htf import aggregate_htf, compute_trend_states_htf, map_trend_to_5m, TrendConfig


@dataclass
class BearComboConfig:
    # 趋势判定（4H EMA）
    ema_fast: int = 20
    ema_slow: int = 60
    ema_bias_bear: float = 0.998  # fast < slow * bias 判定空头
    # 顶背离参数
    rsi_period: int = 14
    swing_left: int = 2
    swing_right: int = 0  # 无未来数据
    div_price_tolerance: float = 0.001  # 允许 H2 比 H1 略低
    div_rsi_diff: float = 2.0
    lookback_div: int = 200  # 背离配对的最大跨度（5m 根数）
    # 空单入场
    short_sl_buffer_pct: float = 0.001
    short_tp_R: float = 2.0
    # 回落破位备胎
    enable_fallback_break: bool = True
    # 过滤
    bear_only: bool = True
    # 趋势模块
    trend_price_tol: float = 0.002
    trend_pivot_lookback: int = 2
    trend_timeframe: str = "1H"  # 使用 1H 高低点判定趋势
    trend_method: str = "ema"  # "ema", "donchian", "regression", "swing"
    trend_ema_up_bias: float = 1.002
    trend_ema_down_bias: float = 0.998
    trend_ema_slope_window: int = 5


def aggregate_4h(df: pd.DataFrame) -> pd.DataFrame:
    df_ts = df.copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"], utc=True)
    df_ts = df_ts.set_index("timestamp")
    agg = (
        df_ts.resample("4H")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    return agg.reset_index()


def compute_bear_flag(df_4h: pd.DataFrame, cfg: BearComboConfig) -> pd.Series:
    close = df_4h["close"]
    ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()
    bear = ema_fast < ema_slow * cfg.ema_bias_bear
    return bear


def map_flag_to_5m(df_5m: pd.DataFrame, flag_4h: pd.Series, ts_4h: pd.Series) -> pd.Series:
    ts5 = pd.to_datetime(df_5m["timestamp"], utc=True)
    bucket = ts5.dt.floor("4H")
    flag_map = dict(zip(ts_4h, flag_4h))
    return bucket.map(flag_map).fillna(False)


def detect_bear_short_signals(df: pd.DataFrame, cfg: BearComboConfig) -> List[Dict]:
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必要列: {required - set(df.columns)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    rsi_series = rsi(df["close"], period=cfg.rsi_period)

    df_htf = aggregate_htf(df, cfg.trend_timeframe)
    # 趋势状态由高低点结构判定
    trend_states_htf = compute_trend_states_htf(
        df_htf,
        TrendConfig(
            method=cfg.trend_method,
            price_tol=cfg.trend_price_tol,
            pivot_lookback_htf=cfg.trend_pivot_lookback,
            pivot_right_htf=0,
            ema_fast=cfg.ema_fast,
            ema_slow=cfg.ema_slow,
            ema_up_bias=cfg.trend_ema_up_bias,
            ema_down_bias=cfg.trend_ema_down_bias,
            ema_slope_window=cfg.trend_ema_slope_window,
        ),
    )
    trend_states_5m = map_trend_to_5m(df, trend_states_htf, df_htf["timestamp"], timeframe=cfg.trend_timeframe)
    df["trend_state"] = trend_states_5m
    df["bear_mode"] = trend_states_5m == "bear"

    swing_highs, swing_lows = find_pivots(df["high"], df["low"], left=cfg.swing_left, right=cfg.swing_right)
    swing_highs = sorted(swing_highs)
    swing_lows = sorted(swing_lows)

    signals: List[Dict] = []
    used_idx = set()

    # 顶背离 -> 跌破关键低点做空
    for i in range(1, len(swing_highs)):
        h1 = swing_highs[i - 1]
        h2 = swing_highs[i]
        if h2 - h1 > cfg.lookback_div:
            continue
        if not df.at[h2, "bear_mode"] and cfg.bear_only:
            continue
        price_cond = df.at[h2, "high"] >= df.at[h1, "high"] * (1 - cfg.div_price_tolerance)
        rsi_cond = rsi_series.iloc[h2] + cfg.div_rsi_diff < rsi_series.iloc[h1]
        if not (price_cond and rsi_cond):
            continue
        k_low = df.iloc[h1 : h2 + 1]["low"].min()
        trigger_idx = None
        for j in range(h2 + 1, len(df)):
            if cfg.bear_only and not df.at[j, "bear_mode"]:
                continue
            if df.at[j, "close"] < k_low:
                trigger_idx = j
                break
        if trigger_idx is None:
            continue
        if trigger_idx in used_idx:
            continue
        entry = df.at[trigger_idx, "close"]
        sl = df.at[h2, "high"] * (1 + cfg.short_sl_buffer_pct)
        risk = sl - entry
        if risk <= 0:
            continue
        tp = entry - cfg.short_tp_R * risk
        signals.append(
            {
                "idx": trigger_idx,
                "side": "short",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "tp_multipliers": [2, 4, 5, 6, 7],
                "tp_fractions": [0.33, 0.33, 0.17, 0.09, 0.08],
                "meta": {"type": "divergence", "h1": h1, "h2": h2, "k_low": k_low},
            }
        )
        used_idx.add(trigger_idx)

    if cfg.enable_fallback_break:
        # 反弹高点失败跌破前低做空
        for i in range(1, len(swing_highs)):
            h = swing_highs[i]
            lows_before = [l for l in swing_lows if l < h]
            if not lows_before:
                continue
            l_prev = lows_before[-1]
            if cfg.bear_only and not df.at[h, "bear_mode"]:
                continue
            trigger_idx = None
            for j in range(h + 1, len(df)):
                if cfg.bear_only and not df.at[j, "bear_mode"]:
                    continue
                if df.at[j, "close"] < df.at[l_prev, "low"]:
                    trigger_idx = j
                    break
            if trigger_idx is None or trigger_idx in used_idx:
                continue
            entry = df.at[trigger_idx, "close"]
            sl = df.at[h, "high"] * (1 + cfg.short_sl_buffer_pct)
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - cfg.short_tp_R * risk
            signals.append(
                {
                    "idx": trigger_idx,
                    "side": "short",
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "tp_multipliers": [2, 4, 5, 6, 7],
                    "tp_fractions": [0.33, 0.33, 0.17, 0.09, 0.08],
                    "meta": {"type": "break_low", "h": h, "l_prev": l_prev},
                }
            )
            used_idx.add(trigger_idx)

    signals.sort(key=lambda x: x["idx"])
    return signals


def detect_bear_combo_signals(
    df: pd.DataFrame, cfg: BearComboConfig, hedge_cfg: Optional[CTDivergenceConfig] = None
) -> Tuple[List[Dict], Dict[str, int]]:
    shorts = detect_bear_short_signals(df, cfg)
    # 计算最新趋势（再算一次，便于外部消费）
    df_htf = aggregate_htf(df, cfg.trend_timeframe)
    trend_states_htf = compute_trend_states_htf(
        df_htf,
        TrendConfig(
            method=cfg.trend_method,
            price_tol=cfg.trend_price_tol,
            pivot_lookback_htf=cfg.trend_pivot_lookback,
            pivot_right_htf=0,
            ema_fast=cfg.ema_fast,
            ema_slow=cfg.ema_slow,
            ema_up_bias=cfg.trend_ema_up_bias,
            ema_down_bias=cfg.trend_ema_down_bias,
            ema_slope_window=cfg.trend_ema_slope_window,
        ),
    )
    trend_states_5m = map_trend_to_5m(df, trend_states_htf, df_htf["timestamp"], timeframe=cfg.trend_timeframe)
    trend_latest = trend_states_5m.iloc[-1] if len(trend_states_5m) else "range"
    hedge_cfg = hedge_cfg or CTDivergenceConfig()
    hedges = detect_ct_signals(df, hedge_cfg)
    # 仅在已有空单信号后才允许对冲多单
    if shorts:
        first_short_idx = min(s["idx"] for s in shorts)
        hedges = [h for h in hedges if h["idx"] > first_short_idx]
    else:
        hedges = []
    signals = shorts + hedges
    signals.sort(key=lambda x: x["idx"])
    stats = {"short_signals": len(shorts), "hedge_long_signals": len(hedges), "trend_latest": trend_latest}
    return signals, stats
