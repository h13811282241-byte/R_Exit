#!/usr/bin/env python3
"""
震荡盒子高抛低吸策略：
- 4H EMA220 + ADX 低过滤
- 5m 盒子 RangeHigh/RangeLow 检测
- 底/顶25%区域反转形态 + RSI/量过滤
- TP1 中轴 50% 减仓+BE，TP2 盒顶/盒底附近全平
- 无重绘：信号仅用当时已收盘数据
"""

from typing import List, Dict
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


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_wilder = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_wilder)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_wilder)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.ewm(alpha=1 / length, adjust=False).mean()
    return adx_val


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def align_4h(df_5m: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    ts_5 = pd.to_datetime(df_5m["timestamp"], utc=True)
    ts_4 = pd.to_datetime(df_4h["timestamp"], utc=True)
    df_4h = df_4h.set_index(ts_4)
    aligned = ts_5.map(lambda t: df_4h.index[df_4h.index <= t].max() if (df_4h.index <= t).any() else pd.NaT)
    out = df_5m.copy()
    out["ema220_4h"] = df_4h.loc[aligned, "ema220"].to_numpy()
    out["adx_4h"] = df_4h.loc[aligned, "adx"].to_numpy()
    out["close_4h"] = df_4h.loc[aligned, "close"].to_numpy()
    return out


def build_box(df: pd.DataFrame, idx: int, n_box: int, atr_mult: float) -> Dict:
    if idx < n_box:
        return {}
    window = df.iloc[idx - n_box + 1 : idx + 1]
    rh = window["high"].max()
    rl = window["low"].min()
    atr_here = df.loc[idx, "atr20"]
    if pd.isna(atr_here) or atr_here <= 0:
        return {}
    if (rh - rl) >= atr_mult * atr_here:
        return {}  # 盒子过宽
    return {"range_high": rh, "range_low": rl, "range_mid": (rh + rl) / 2, "range_w": rh - rl}


def detect_signals(
    df5: pd.DataFrame,
    df4h: pd.DataFrame,
    n_box: int = 80,
    atr_mult_box: float = 3.0,
    rsi_low: float = 35.0,
    rsi_high: float = 65.0,
    wick_body_ratio: float = 1.2,
    vol_mult: float = 2.0,
    adx_thresh: float = 20.0,
    ema_dev: float = 0.03,
    swing_lookback: int = 2,
) -> List[Dict]:
    """
    返回信号：idx/side/entry/sl/tp1/tp2/risk/box info
    """
    df5 = df5.copy()
    df5 = find_swings(df5, swing_lookback)
    df5["atr20"] = atr(df5, 20)
    df5["rsi"] = rsi(df5["close"], 14)
    df5["vol_ma20"] = df5["volume"].rolling(20).mean()

    df4h = df4h.copy()
    df4h["ema220"] = ema(df4h["close"], 220)
    df4h["adx"] = adx(df4h, 14)

    df5 = align_4h(df5, df4h)

    signals = []
    for i in range(len(df5)):
        adx_ok = df5.loc[i, "adx_4h"] < adx_thresh if not pd.isna(df5.loc[i, "adx_4h"]) else False
        ema_ok = (
            abs(df5.loc[i, "close_4h"] - df5.loc[i, "ema220_4h"]) / df5.loc[i, "close_4h"] < ema_dev
            if not pd.isna(df5.loc[i, "close_4h"]) and not pd.isna(df5.loc[i, "ema220_4h"])
            else False
        )
        if not (adx_ok and ema_ok):
            continue
        box = build_box(df5, i, n_box, atr_mult_box)
        if not box:
            continue
        rh, rl, rm, rw = box["range_high"], box["range_low"], box["range_mid"], box["range_w"]
        close_i = df5.loc[i, "close"]
        open_i = df5.loc[i, "open"]
        high_i = df5.loc[i, "high"]
        low_i = df5.loc[i, "low"]
        body = abs(close_i - open_i)
        upper = high_i - max(close_i, open_i)
        lower = min(close_i, open_i) - low_i
        vol_ok = df5.loc[i, "volume"] <= df5.loc[i, "vol_ma20"] * vol_mult if not pd.isna(df5.loc[i, "vol_ma20"]) else False
        rsi_now = df5.loc[i, "rsi"]
        rsi_prev = df5.loc[i - 1, "rsi"] if i > 0 else np.nan

        # Long zone: bottom 25%
        if close_i <= rl + 0.25 * rw:
            cond_wick = lower >= wick_body_ratio * body and close_i >= (high_i + low_i) / 2
            cond_rsi = (rsi_now < rsi_low) and (rsi_prev < rsi_now if not np.isnan(rsi_prev) else True)
            if cond_wick and cond_rsi and vol_ok:
                recent_sw_low = df5.loc[:i, "low"][df5.loc[:i, "swing_low"]].iloc[-1] if df5.loc[:i, "swing_low"].any() else rl
                sl = min(rl, recent_sw_low) - 0.3 * df5.loc[i, "atr20"]
                risk = close_i - sl
                if risk > 0:
                    tp1 = rm
                    tp2 = rh - 0.1 * rw
                    signals.append(
                        {
                            "idx": i,
                            "side": "long",
                            "entry": close_i,
                            "sl": sl,
                            "tp1": tp1,
                            "tp2": tp2,
                            "risk": risk,
                            "box": box,
                        }
                    )
        # Short zone: top 25%
        if close_i >= rh - 0.25 * rw:
            cond_wick = upper >= wick_body_ratio * body and close_i <= (high_i + low_i) / 2
            cond_rsi = (rsi_now > rsi_high) and (rsi_prev > rsi_now if not np.isnan(rsi_prev) else True)
            if cond_wick and cond_rsi and vol_ok:
                recent_sw_high = df5.loc[:i, "high"][df5.loc[:i, "swing_high"]].iloc[-1] if df5.loc[:i, "swing_high"].any() else rh
                sl = max(rh, recent_sw_high) + 0.3 * df5.loc[i, "atr20"]
                risk = sl - close_i
                if risk > 0:
                    tp1 = rm
                    tp2 = rl + 0.1 * rw
                    signals.append(
                        {
                            "idx": i,
                            "side": "short",
                            "entry": close_i,
                            "sl": sl,
                            "tp1": tp1,
                            "tp2": tp2,
                            "risk": risk,
                            "box": box,
                        }
                    )
    return signals


def simulate_trades(df5: pd.DataFrame, signals: List[Dict]) -> List[Dict]:
    if not signals:
        return []
    close = df5["close"].to_numpy()
    high = df5["high"].to_numpy()
    low = df5["low"].to_numpy()
    last_idx = len(df5) - 1
    trades = []
    for sig in signals:
        i = sig["idx"]
        if i >= last_idx:
            continue
        side = sig["side"]
        entry = sig["entry"]
        sl = sig["sl"]
        tp1 = sig["tp1"]
        tp2 = sig["tp2"]
        risk = sig["risk"]
        filled_tp1 = False
        exit_idx = None
        exit_price = None
        for b in range(i + 1, last_idx + 1):
            h = high[b]
            l = low[b]
            if side == "long":
                hit_sl = l <= sl
                hit_tp1 = h >= tp1
                hit_tp2 = h >= tp2
            else:
                hit_sl = h >= sl
                hit_tp1 = l <= tp1
                hit_tp2 = l <= tp2
            if hit_sl and not filled_tp1:
                exit_idx = b
                exit_price = sl
                pnl_R = -1.0
                break
            if hit_tp1 and not filled_tp1:
                filled_tp1 = True
                # move SL to BE
                sl_be = entry
                # continue scanning for TP2 or SL_BE
                for k in range(b, last_idx + 1):
                    hh = high[k]
                    ll = low[k]
                    if side == "long":
                        hit_sl_be = ll <= sl_be
                        hit_tp2_after = hh >= tp2
                    else:
                        hit_sl_be = hh >= sl_be
                        hit_tp2_after = ll <= tp2
                    if hit_tp2_after:
                        exit_idx = k
                        exit_price = tp2
                        pnl_R = 0.5 * ((tp1 - entry) / risk if side == "long" else (entry - tp1) / risk) + 0.5 * (
                            (tp2 - entry) / risk if side == "long" else (entry - tp2) / risk
                        )
                        break
                    if hit_sl_be:
                        exit_idx = k
                        exit_price = sl_be
                        pnl_R = 0.5 * ((tp1 - entry) / risk if side == "long" else (entry - tp1) / risk)
                        break
                else:
                    exit_idx = last_idx
                    exit_price = close[last_idx]
                    pnl_R = 0.5 * ((tp1 - entry) / risk if side == "long" else (entry - tp1) / risk) + 0.5 * (
                        (exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk
                    )
                break
        if exit_idx is None:
            exit_idx = last_idx
            exit_price = close[last_idx]
            pnl_R = (exit_price - entry) / risk if side == "long" else (entry - exit_price) / risk

        trades.append(
            {
                "entry_time": df5.loc[i, "timestamp"],
                "exit_time": df5.loc[exit_idx, "timestamp"],
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "exit_price": exit_price,
                "raw_R": pnl_R,
                "net_R": pnl_R,  # 手续费可在外部折算
                "entry_idx": i,
                "exit_idx": exit_idx,
                "box_high": sig["box"]["range_high"],
                "box_low": sig["box"]["range_low"],
                "box_mid": sig["box"]["range_mid"],
            }
        )
    return trades
