#!/usr/bin/env python3
"""
Generic backtest engine: simulate trades given signals and SL/TP.
"""

from typing import List, Dict, Optional
import math
import pandas as pd
import numpy as np


def summarize_trades(trades: List[Dict], key: str = "net_R") -> Dict:
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_raw_R": 0.0,
            "avg_R": 0.0,
            "median_R": 0.0,
            "max_R": 0.0,
            "min_R": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
        }
    def get_R(t, k):
        if k in t and t[k] is not None and not math.isnan(t[k]):
            return float(t[k])
        if "net_R" in t and t["net_R"] is not None and not math.isnan(t["net_R"]):
            return float(t["net_R"])
        if "R" in t and t["R"] is not None and not math.isnan(t["R"]):
            return float(t["R"])
        return math.nan

    R_vals = [get_R(t, key) for t in trades]
    R_vals = [r for r in R_vals if not math.isnan(r)]
    if not R_vals:
        return {
            "num_trades": len(trades),
            "win_rate": 0.0,
            "avg_raw_R": 0.0,
            "avg_R": 0.0,
            "median_R": 0.0,
            "max_R": 0.0,
            "min_R": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
        }
    wins = [r for r in R_vals if r > 0]
    losses = [r for r in R_vals if r < 0]
    cum = np.array(R_vals).cumsum()
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf') if wins else 0.0
    return {
        "num_trades": len(R_vals),
        "win_rate": len(wins) / len(R_vals),
        "avg_raw_R": float(np.mean(R_vals)),
        "avg_R": float(np.mean(R_vals)),
        "median_R": float(np.median(R_vals)),
        "max_R": float(np.max(R_vals)),
        "min_R": float(np.min(R_vals)),
        "avg_win_R": float(np.mean(wins)) if wins else 0.0,
        "avg_loss_R": float(np.mean(losses)) if losses else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_R": float(max_dd),
    }


def simulate_basic(
    df: pd.DataFrame,
    signals: List[Dict],
    fee_side_pct: float = 0.000248,
    max_holding_bars: Optional[int] = None,
    lower_df: Optional[pd.DataFrame] = None,
    upper_interval_sec: int = 0,
    lower_interval_sec: int = 0,
) -> List[Dict]:
    """
    signals: list of dict with idx, side, entry, sl, tp (optional), exit_idx (optional)
    """
    required = {"timestamp", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必需列: {required - set(df.columns)}")
    prices = df[["high", "low", "close"]].to_numpy()
    ts_upper = pd.to_datetime(df["timestamp"], utc=True).to_numpy()
    lower_data = None
    if lower_df is not None:
        lower_data = {
            "ts": pd.to_datetime(lower_df["timestamp"], utc=True).to_numpy(),
            "high": lower_df["high"].to_numpy(),
            "low": lower_df["low"].to_numpy(),
        }

    fee_round = fee_side_pct * 2.0
    trades: List[Dict] = []
    last_index = len(df) - 1

    def resolve_conflict(side: str, sl: float, tp: float, start_ts, end_ts):
        if lower_data is None:
            return "sl", sl
        mask = (lower_data["ts"] >= start_ts) & (lower_data["ts"] < end_ts)
        if not mask.any():
            return "sl", sl
        for h, l in zip(lower_data["high"][mask], lower_data["low"][mask]):
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
                return "sl", sl
        return "sl", sl

    for sig in signals:
        idx = sig["idx"]
        if idx >= last_index:
            continue
        side = sig["side"]
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig.get("tp")
        risk = entry - sl if side == "long" else sl - entry
        if risk <= 0:
            continue
        exit_idx = None
        exit_price = None
        outcome = None
        end_idx = last_index if max_holding_bars is None else min(last_index, idx + max_holding_bars)

        for bar in range(idx + 1, end_idx + 1):
            high_b, low_b, close_b = prices[bar]
            if side == "long":
                hit_sl = low_b <= sl
                hit_tp = (tp is not None) and (high_b >= tp)
            else:
                hit_sl = high_b >= sl
                hit_tp = (tp is not None) and (low_b <= tp)
            if hit_sl and hit_tp:
                start_ts = ts_upper[idx]
                end_ts = start_ts + pd.Timedelta(seconds=upper_interval_sec) if upper_interval_sec else ts_upper[bar]
                sub_outcome, sub_price = resolve_conflict(side, sl, tp, start_ts, end_ts)
                outcome = sub_outcome
                exit_price = sub_price
                exit_idx = bar
                break
            if hit_sl:
                outcome = "sl"
                exit_price = sl
                exit_idx = bar
                break
            if hit_tp:
                outcome = "tp"
                exit_price = tp
                exit_idx = bar
                break
        if exit_idx is None:
            exit_idx = end_idx
            exit_price = prices[end_idx][2]
            outcome = "timeout"

        pnl = (exit_price - entry) if side == "long" else (entry - exit_price)
        raw_R = pnl / risk
        risk_pct = risk / entry
        fee_R = fee_round / risk_pct if risk_pct > 0 else 0.0
        net_R = raw_R - fee_R

        trades.append(
            {
                "entry_time": df.loc[idx, "timestamp"],
                "exit_time": df.loc[exit_idx, "timestamp"],
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
                "entry_idx": idx,
                "exit_idx": exit_idx,
            }
        )
    return trades
