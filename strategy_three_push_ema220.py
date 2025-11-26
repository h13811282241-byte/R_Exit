#!/usr/bin/env python3
"""
5m 三推反转 + 4H EMA220 趋势过滤策略（多空对称），分批止盈/逆势半仓，含震荡过滤与入场后行为分类。
不重绘：摆动点/三推结构/趋势判定均只用已收盘数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ThreePushParams:
    atr_period: int = 14
    swing_lookback: int = 2
    three_push_max_lookback: int = 200
    max_push_slope_atr: float = 0.5  # 三推高/低连线的归一化斜率上限（每根K的ATR倍数）
    buffer_atr: float = 0.1
    break_scan_forward: int = 50
    trend_lookback_4h: int = 5
    trend_slope_threshold: float = 0.001
    allow_trend_none: bool = False
    base_position_size: float = 1.0
    countertrend_position_factor: float = 0.5
    enable_countertrend: bool = True
    avoid_trading_in_chop: bool = True
    chop_lookback_bars: int = 20
    chop_atr_factor: float = 1.5
    lookahead_bars: int = 20
    impulse_R: float = 2.0
    range_factor: float = 1.5
    fee_side: float = 0.00045  # 单边


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def find_swings(df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        if highs[i] == highs[i - lookback : i + lookback + 1].max():
            swing_high[i] = True
        if lows[i] == lows[i - lookback : i + lookback + 1].min():
            swing_low[i] = True
    out = df.copy()
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out


def compute_trend_4h(df_4h: pd.DataFrame, trend_lookback: int, slope_thresh: float) -> pd.DataFrame:
    out = df_4h.copy()
    out["ema220"] = ema(out["close"], 220)
    up = (out["close"] > out["ema220"]) & (
        out["ema220"] > out["ema220"].shift(trend_lookback) * (1 + slope_thresh)
    )
    down = (out["close"] < out["ema220"]) & (
        out["ema220"] < out["ema220"].shift(trend_lookback) * (1 - slope_thresh)
    )
    trend = np.where(up, "bull", np.where(down, "bear", "none"))
    out["trend_state"] = trend
    return out


def align_4h_to_5m(df_5m: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    为 5m 数据追加最近一根已收盘 4H 的 close/ema220/trend_state。
    """
    df_5m = df_5m.copy()
    ts_5m = pd.to_datetime(df_5m["timestamp"], utc=True)
    ts_4h = pd.to_datetime(df_4h["timestamp"], utc=True)
    df_4h = df_4h.set_index(ts_4h)
    aligned_idx = ts_5m.map(lambda t: df_4h.index[df_4h.index <= t].max() if (df_4h.index <= t).any() else pd.NaT)
    df_5m["ema220_4h"] = df_4h.loc[aligned_idx, "ema220"].to_numpy()
    df_5m["trend_state_4h"] = df_4h.loc[aligned_idx, "trend_state"].to_numpy()
    df_5m["close_4h"] = df_4h.loc[aligned_idx, "close"].to_numpy()
    return df_5m


def in_chop_env(df: pd.DataFrame, idx: int, lookback: int, atr_factor: float) -> bool:
    if idx < lookback:
        return False
    window = df.iloc[idx - lookback + 1 : idx + 1]
    win_range = window["high"].max() - window["low"].min()
    atr_here = df.iloc[idx]["atr"]
    if pd.isna(atr_here) or atr_here <= 0:
        return False
    return win_range < atr_factor * atr_here


def detect_three_push_structures(df: pd.DataFrame, direction: str, max_lookback: int, max_slope_atr: float) -> List[Dict]:
    """
    direction: "up" (三推向上，用于做空) or "down" (三推向下，用于做多)
    使用 swing_high / swing_low 列。
    """
    swings_idx = df.index[df["swing_high"]].tolist() if direction == "up" else df.index[df["swing_low"]].tolist()
    opp_idx = df.index[df["swing_low"]].tolist() if direction == "up" else df.index[df["swing_high"]].tolist()
    structures = []
    for i in range(len(swings_idx) - 2):
        H1 = swings_idx[i]
        H2 = swings_idx[i + 1]
        H3 = swings_idx[i + 2]
        if H3 - H1 > max_lookback:
            continue
        if direction == "up":
            if not (df.loc[H1, "high"] < df.loc[H2, "high"] < df.loc[H3, "high"]):
                continue
            atr_ref = df.loc[H3, "atr"] if not pd.isna(df.loc[H3, "atr"]) else df["atr"].iloc[H1:H3+1].mean()
            bars = H3 - H1
            if atr_ref <= 0 or bars <= 0:
                continue
            slope_norm = abs(df.loc[H3, "high"] - df.loc[H1, "high"]) / (bars * atr_ref)
            if slope_norm > max_slope_atr:
                continue
            # L1 between H1,H2; L2 between H2,H3
            L1_candidates = [j for j in opp_idx if H1 < j < H2]
            L2_candidates = [j for j in opp_idx if H2 < j < H3]
            if not L1_candidates or not L2_candidates:
                continue
            L1 = min(L1_candidates, key=lambda j: df.loc[j, "low"])
            L2 = min(L2_candidates, key=lambda j: df.loc[j, "low"])
            # 低点应抬高，避免 L2 < L1 的畸形结构
            if df.loc[L2, "low"] <= df.loc[L1, "low"]:
                continue
            L3_candidates = [j for j in opp_idx if j > H3]
            L3 = min(L3_candidates, key=lambda j: df.loc[j, "low"]) if L3_candidates else None
            structures.append({"direction": "up", "H1": H1, "H2": H2, "H3": H3, "L1": L1, "L2": L2, "L3": L3})
        else:
            if not (df.loc[H1, "low"] > df.loc[H2, "low"] > df.loc[H3, "low"]):
                continue
            atr_ref = df.loc[H3, "atr"] if not pd.isna(df.loc[H3, "atr"]) else df["atr"].iloc[H1:H3+1].mean()
            bars = H3 - H1
            if atr_ref <= 0 or bars <= 0:
                continue
            slope_norm = abs(df.loc[H3, "low"] - df.loc[H1, "low"]) / (bars * atr_ref)
            if slope_norm > max_slope_atr:
                continue
            H1c = [j for j in opp_idx if H1 < j < H2]
            H2c = [j for j in opp_idx if H2 < j < H3]
            if not H1c or not H2c:
                continue
            H1p = max(H1c, key=lambda j: df.loc[j, "high"])
            H2p = max(H2c, key=lambda j: df.loc[j, "high"])
            # 高点应递减，避免 H2 >= H1
            if df.loc[H2p, "high"] >= df.loc[H1p, "high"]:
                continue
            H3p = max([j for j in opp_idx if j > H3], key=lambda j: df.loc[j, "high"]) if [j for j in opp_idx if j > H3] else None
            structures.append({"direction": "down", "L1": H1, "L2": H2, "L3": H3, "H1": H1p, "H2": H2p, "H3": H3p})
    return structures


def generate_signals(
    df_5m: pd.DataFrame,
    df_4h: pd.DataFrame,
    params: ThreePushParams,
) -> List[Dict]:
    df_5m = find_swings(df_5m.copy(), params.swing_lookback)
    df_5m["atr"] = atr(df_5m, params.atr_period)
    df_4h_trend = compute_trend_4h(df_4h.copy(), params.trend_lookback_4h, params.trend_slope_threshold)
    df_5m = align_4h_to_5m(df_5m, df_4h_trend)

    signals: List[Dict] = []
    structures_up = detect_three_push_structures(df_5m, "up", params.three_push_max_lookback, params.max_push_slope_atr)
    structures_dn = detect_three_push_structures(df_5m, "down", params.three_push_max_lookback, params.max_push_slope_atr)

    def trend_align(side: str, trend_state: str) -> str:
        if trend_state == "bull":
            return "with_trend" if side == "long" else "countertrend"
        if trend_state == "bear":
            return "with_trend" if side == "short" else "countertrend"
        return "none"

    for st in structures_up + structures_dn:
        dir_up = st["direction"] == "up"
        if dir_up:
            key_low = st["L2"]
            key_high = st["H3"]
            atr_here = df_5m.loc[key_low, "atr"]
            trigger = df_5m.loc[key_low, "low"] - params.buffer_atr * atr_here
            # 从 L2 之后寻找突破
            scan_slice = df_5m.loc[key_low + 1 : key_low + params.break_scan_forward]
            hit_idx = None
            for idx, row in scan_slice.iterrows():
                if row["close"] < trigger:
                    hit_idx = idx
                    break
            if hit_idx is None:
                continue
            entry_price = df_5m.loc[hit_idx, "close"]
            sl_price = df_5m.loc[key_high, "high"] + params.buffer_atr * atr_here
            if sl_price <= entry_price:
                continue
            side = "short"
        else:
            key_high = st["H2"]
            key_low = st["L3"]
            atr_here = df_5m.loc[key_high, "atr"]
            trigger = df_5m.loc[key_high, "high"] + params.buffer_atr * atr_here
            scan_slice = df_5m.loc[key_high + 1 : key_high + params.break_scan_forward]
            hit_idx = None
            for idx, row in scan_slice.iterrows():
                if row["close"] > trigger:
                    hit_idx = idx
                    break
            if hit_idx is None:
                continue
            entry_price = df_5m.loc[hit_idx, "close"]
            sl_price = df_5m.loc[key_low, "low"] - params.buffer_atr * atr_here
            if sl_price >= entry_price:
                continue
            side = "long"

        trend_state = df_5m.loc[hit_idx, "trend_state_4h"]
        align = trend_align(side, trend_state)
        if align == "none" and not params.allow_trend_none:
            continue
        if align == "countertrend" and not params.enable_countertrend:
            continue
        chop_flag = in_chop_env(df_5m, hit_idx, params.chop_lookback_bars, params.chop_atr_factor) if params.avoid_trading_in_chop else False
        if chop_flag and params.avoid_trading_in_chop:
            continue

        risk = entry_price - sl_price if side == "long" else sl_price - entry_price
        if risk <= 0:
            continue

        signals.append(
            {
                "idx": hit_idx,
                "side": side,
                "entry": entry_price,
                "sl": sl_price,
                "risk": risk,
                "trend_alignment": align,
                "opened_in_chop_env": chop_flag,
                "structure": st,
            }
        )
    return signals


def simulate_trades(df_5m: pd.DataFrame, signals: List[Dict], params: ThreePushParams) -> List[Dict]:
    if not signals:
        return []
    close = df_5m["close"].to_numpy()
    high = df_5m["high"].to_numpy()
    low = df_5m["low"].to_numpy()
    fee_round = params.fee_side * 2
    last_idx = len(df_5m) - 1
    trades: List[Dict] = []

    for sig in signals:
        i = sig["idx"]
        if i >= last_idx:
            continue
        side = sig["side"]
        entry = sig["entry"]
        sl = sig["sl"]
        risk = sig["risk"]
        align = sig["trend_alignment"]
        pos_mult = params.base_position_size if align == "with_trend" else params.base_position_size * params.countertrend_position_factor

        # TP 设置
        if align == "with_trend":
            tps = [2, 4, 5, 6, 7]
            weights = [0.33, 0.33, 0.17, 0.09, 0.08]
        else:
            tps = [1]
            weights = [1.0]
        tp_prices = [entry + m * risk if side == "long" else entry - m * risk for m in tps]
        tp_hit = [False] * len(tp_prices)
        realized_R = 0.0
        remaining = 1.0
        exit_idx = None
        exit_price = None

        for b in range(i + 1, last_idx + 1):
            h = high[b]
            l = low[b]
            c = close[b]
            hit_sl = l <= sl if side == "long" else h >= sl
            # 止损优先
            if hit_sl and not any(tp_hit):
                exit_idx = b
                exit_price = sl
                realized_R += remaining * ((exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk)
                remaining = 0.0
                break
            # 处理 TP，按序
            for j, tp in enumerate(tp_prices):
                if tp_hit[j]:
                    continue
                hit_tp = h >= tp if side == "long" else l <= tp
                if hit_tp:
                    tp_hit[j] = True
                    realized_R += weights[j] * ((tp - entry) / risk if side == "long" else (entry - tp) / risk)
                    remaining -= weights[j]
            # 如果是逆势单，只有一个 TP1R
            if align != "with_trend" and any(tp_hit):
                exit_idx = b
                exit_price = tp_prices[0]
                remaining = 0.0
                break
            # 如果触达SL在已部分止盈后
            if hit_sl and remaining > 0:
                exit_idx = b
                exit_price = sl
                realized_R += remaining * ((exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk)
                remaining = 0.0
                break
        if remaining > 0:
            exit_idx = exit_idx if exit_idx is not None else last_idx
            exit_price = exit_price if exit_price is not None else close[last_idx]
            realized_R += remaining * ((exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk)

        risk_pct = risk / entry if entry else 0
        fee_R = fee_round / risk_pct if risk_pct else 0.0

        # 入场后行为分类
        end_idx = min(i + params.lookahead_bars, last_idx)
        sub_high = high[i + 1 : end_idx + 1]
        sub_low = low[i + 1 : end_idx + 1]
        if side == "long":
            max_favor = (sub_high.max() - entry) if len(sub_high) else 0
            max_against = (entry - sub_low.min()) if len(sub_low) else 0
        else:
            max_favor = (entry - sub_low.min()) if len(sub_low) else 0
            max_against = (sub_high.max() - entry) if len(sub_high) else 0
        price_range = (sub_high.max() - sub_low.min()) if len(sub_high) and len(sub_low) else 0
        direct_reversal = max_favor >= params.impulse_R * risk
        range_chop = (max_favor < 1.0 * risk) and (price_range < params.range_factor * risk)

        trades.append(
            {
                "entry_time": df_5m.loc[i, "timestamp"],
                "exit_time": df_5m.loc[exit_idx, "timestamp"] if exit_idx is not None else df_5m.loc[last_idx, "timestamp"],
                "side": side,
                "entry_idx": i,
                "entry": entry,
                "sl": sl,
                "risk": risk,
                "trend_alignment": align,
                "opened_in_chop_env": sig["opened_in_chop_env"],
                "direct_reversal": direct_reversal,
                "range_chop": range_chop,
                "max_favor": max_favor,
                "max_against": max_against,
                "structure": sig["structure"],
                "raw_R": realized_R * pos_mult,
                "fee_R": fee_R * pos_mult,
                "net_R": realized_R * pos_mult - fee_R * pos_mult,
            }
        )
    return trades


def plot_random_three_push_samples(
    df_5m: pd.DataFrame, trades: List[Dict], n_samples: int = 5, seed: Optional[int] = None, save_dir: Optional[str] = None
):
    import matplotlib.pyplot as plt
    import random
    random.seed(seed)
    if not trades:
        return
    samples = trades if len(trades) <= n_samples else random.sample(trades, n_samples)
    ts = pd.to_datetime(df_5m["timestamp"])
    for k, tr in enumerate(samples):
        st = tr["structure"]
        idxs = [v for v in st.values() if isinstance(v, int)]
        start = max(0, min(idxs) - 20)
        base_idx = tr.get("entry_idx", min(idxs) if idxs else 0)
        end = min(len(df_5m) - 1, (st.get("H3") or st.get("L3") or base_idx) + 20)
        sub = df_5m.iloc[start : end + 1].copy()
        sub = sub.reset_index(drop=True)
        sub["ts"] = ts.iloc[start : end + 1].to_numpy()
        sub["x"] = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(12, 5))
        width = 0.6
        for _, row in sub.iterrows():
            c = "green" if row["close"] >= row["open"] else "red"
            ax.vlines(row["x"], row["low"], row["high"], color=c, linewidth=1)
            ax.add_patch(
                plt.Rectangle(
                    (row["x"] - width / 2, min(row["open"], row["close"])),
                    width,
                    abs(row["close"] - row["open"]),
                    facecolor=c,
                    edgecolor=c,
                    alpha=0.7,
                )
            )

        def mark(idx, label, color):
            if idx < start or idx > end:
                return
            x = idx - start
            y = df_5m.loc[idx, "close"]
            ax.scatter(x, y, color=color, s=80, zorder=6, edgecolors="k", linewidths=0.8)
            ax.text(x, y, label, color=color, fontsize=10, fontweight="bold", ha="center", va="bottom", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        if st["direction"] == "up":
            mark(st["H1"], "H1", "red")
            mark(st["H2"], "H2", "red")
            mark(st["H3"], "H3", "red")
            mark(st["L1"], "L1", "blue")
            mark(st["L2"], "L2", "blue")
            if st.get("L3") is not None:
                mark(st["L3"], "L3", "blue")
        else:
            mark(st["L1"], "L1", "blue")
            mark(st["L2"], "L2", "blue")
            mark(st["L3"], "L3", "blue")
            mark(st["H1"], "H1", "red")
            mark(st["H2"], "H2", "red")
            if st.get("H3") is not None:
                mark(st["H3"], "H3", "red")

        # entry/sl lines
        entry_x = tr["entry_idx"] - start if "entry_idx" in tr and start <= tr["entry_idx"] <= end else None
        if entry_x is not None:
            ax.axvline(entry_x, color="gray", linestyle=":", alpha=0.7, label="Entry")
            ax.scatter(entry_x, tr["entry"], color="black", s=80, zorder=7, edgecolors="k", linewidths=0.8)
            ax.text(entry_x, tr["entry"], "ENTRY", color="black", fontsize=10, fontweight="bold", ha="center", va="bottom", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        # SL
        ax.axhline(tr["sl"], color="orange", linestyle="--", label="SL")
        ax.text(sub["x"].iloc[0], tr["sl"], "SL", color="orange", fontsize=9, fontweight="bold", va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        # 若有分批TP，标出水平线
        for key, color in [("tp1", "green"), ("tp2", "green")]:
            if key in tr:
                ax.axhline(tr[key], color=color, linestyle="--", alpha=0.6)
                ax.text(sub["x"].iloc[-1], tr[key], key.upper(), color=color, fontsize=9, fontweight="bold", va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        ax.set_xlim(-1, len(sub))
        xticks = np.linspace(0, len(sub) - 1, num=min(8, len(sub))).astype(int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([sub.loc[x, "ts"].strftime("%m-%d %H:%M") for x in xticks], rotation=45, ha="right")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"{tr['side']} | {tr.get('trend_alignment','')} | net_R={tr.get('net_R',0):.2f}")
        ax.legend()
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f"{save_dir}/three_push_sample_{k}_{tr['side']}_R{tr.get('net_R',0):.2f}.png", bbox_inches="tight")
        plt.close(fig)
