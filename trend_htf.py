#!/usr/bin/env python3
"""
高周期趋势判定模块（默认 4H，可调整 1H），提供多种方法：
- EMA+斜率
- Donchian 中轴偏移
- 线性回归斜率+R²
- 高低点结构（HH/HL vs LH/LL）
并可映射趋势状态到 5m 时间轴。
"""

from dataclasses import dataclass
from typing import List, Dict, Literal

import numpy as np
import pandas as pd

TrendState = Literal["bull", "bear", "range"]


@dataclass
class TrendConfig:
    method: str = "ema"  # "ema", "donchian", "regression", "swing"
    pivot_lookback_htf: int = 2
    pivot_right_htf: int = 0  # 0 表示无未来数据
    price_tol: float = 0.002  # 0.2% 容差
    # EMA
    ema_fast: int = 50
    ema_slow: int = 200
    ema_up_bias: float = 1.002
    ema_down_bias: float = 0.998
    ema_slope_window: int = 5
    # Donchian
    donchian_n: int = 50
    donchian_band: float = 0.2  # 20% 偏移
    # Regression
    reg_window: int = 30
    reg_slope_th: float = 0.0
    reg_r2_th: float = 0.3


def aggregate_htf(df_5m: pd.DataFrame, timeframe: str = "4H") -> pd.DataFrame:
    df_ts = df_5m.copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"], utc=True)
    df_ts = df_ts.set_index("timestamp")
    agg = (
        df_ts.resample(timeframe)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    return agg.reset_index()


def aggregate_4h(df_5m: pd.DataFrame) -> pd.DataFrame:
    """兼容旧接口，默认聚合 4H。"""
    return aggregate_htf(df_5m, "4H")


def detect_htf_swings(high: pd.Series, low: pd.Series, lookback: int = 2, right: int = 0) -> List[Dict]:
    swings: List[Dict] = []
    n = len(high)
    for i in range(lookback, n - right):
        h = high.iloc[i]
        l = low.iloc[i]
        win_high = high.iloc[i - lookback : i + right + 1]
        win_low = low.iloc[i - lookback : i + right + 1]
        is_high = (h == win_high.max()) and (win_high.idxmax() == high.index[i])
        is_low = (l == win_low.min()) and (win_low.idxmin() == low.index[i])
        if is_high:
            swings.append({"idx": i, "price": float(h), "type": "high"})
        elif is_low:
            swings.append({"idx": i, "price": float(l), "type": "low"})
    return swings


def get_trend_from_swings(swings: List[Dict], price_tol: float = 0.002) -> TrendState:
    lows = [s for s in swings if s["type"] == "low"]
    highs = [s for s in swings if s["type"] == "high"]
    if len(lows) < 2 or len(highs) < 2:
        return "range"
    L_last, L_prev = lows[-1], lows[-2]
    H_last, H_prev = highs[-1], highs[-2]
    bull_low = L_last["price"] > L_prev["price"] * (1 + price_tol)
    bull_high = H_last["price"] > H_prev["price"] * (1 + price_tol)
    bear_low = L_last["price"] < L_prev["price"] * (1 - price_tol)
    bear_high = H_last["price"] < H_prev["price"] * (1 - price_tol)
    if bull_low and bull_high:
        return "bull"
    if bear_low and bear_high:
        return "bear"
    return "range"


def compute_trend_states_4h(df_4h: pd.DataFrame, cfg: TrendConfig = TrendConfig()) -> pd.Series:
    """
    默认方法向后兼容：使用 swing 结构判定。
    """
    return compute_trend_states_htf(df_4h, cfg)


def compute_trend_states_htf(df_htf: pd.DataFrame, cfg: TrendConfig = TrendConfig()) -> pd.Series:
    method = cfg.method
    trend_states = ["range"] * len(df_htf)
    last_state: TrendState = "range"
    if method == "ema":
        close = df_htf["close"]
        ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()
        slope_fast = ema_fast.diff(cfg.ema_slope_window)
        for i in range(len(df_htf)):
            if ema_fast.iloc[i] > ema_slow.iloc[i] * cfg.ema_up_bias and slope_fast.iloc[i] > 0:
                last_state = "bull"
            elif ema_fast.iloc[i] < ema_slow.iloc[i] * cfg.ema_down_bias and slope_fast.iloc[i] < 0:
                last_state = "bear"
            else:
                last_state = "range"
            trend_states[i] = last_state
    elif method == "donchian":
        high = df_htf["high"]
        low = df_htf["low"]
        for i in range(len(df_htf)):
            if i < cfg.donchian_n - 1:
                trend_states[i] = last_state
                continue
            window_high = high.iloc[i - cfg.donchian_n + 1 : i + 1].max()
            window_low = low.iloc[i - cfg.donchian_n + 1 : i + 1].min()
            mid = (window_high + window_low) / 2
            rng = window_high - window_low
            upper_band = mid + cfg.donchian_band * rng
            lower_band = mid - cfg.donchian_band * rng
            close_i = df_htf["close"].iloc[i]
            if close_i > upper_band:
                last_state = "bull"
            elif close_i < lower_band:
                last_state = "bear"
            else:
                last_state = "range"
            trend_states[i] = last_state
    elif method == "regression":
        close = df_htf["close"]
        for i in range(len(df_htf)):
            if i < cfg.reg_window - 1:
                trend_states[i] = last_state
                continue
            y = np.log(close.iloc[i - cfg.reg_window + 1 : i + 1].to_numpy())
            x = np.arange(cfg.reg_window, dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            if slope > cfg.reg_slope_th and r2 > cfg.reg_r2_th:
                last_state = "bull"
            elif slope < -cfg.reg_slope_th and r2 > cfg.reg_r2_th:
                last_state = "bear"
            else:
                last_state = "range"
            trend_states[i] = last_state
    else:  # swing 结构
        swings = detect_htf_swings(
            df_htf["high"],
            df_htf["low"],
            lookback=cfg.pivot_lookback_htf,
            right=cfg.pivot_right_htf,
        )
        for i in range(len(df_htf)):
            swings_up_to_i = [s for s in swings if s["idx"] <= i]
            if len(swings_up_to_i) >= 4:
                last_state = get_trend_from_swings(swings_up_to_i, price_tol=cfg.price_tol)
            trend_states[i] = last_state
    return pd.Series(trend_states, index=df_htf.index, name="trend_state")


def map_trend_to_5m(df_5m: pd.DataFrame, trend_htf: pd.Series, ts_htf: pd.Series, timeframe: str = "4H") -> pd.Series:
    ts5 = pd.to_datetime(df_5m["timestamp"], utc=True)
    bucket = ts5.dt.floor(timeframe)
    trend_map = dict(zip(ts_htf, trend_htf))
    return bucket.map(trend_map).fillna("range")
