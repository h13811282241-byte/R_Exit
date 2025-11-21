#!/usr/bin/env python3
"""
回测汇总与权益曲线工具。
"""

from typing import Dict, List
import math
import numpy as np
import pandas as pd


def summarize_trades(trades: List[Dict]) -> Dict:
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_R": 0.0,
            "median_R": 0.0,
            "max_R": 0.0,
            "min_R": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "expectancy_R": 0.0,
            "win_loss_ratio": 0.0,
        }
    R_vals = [t["R"] for t in trades if t["R"] is not None and not math.isnan(t["R"])]
    if not R_vals:
        return {
            "num_trades": len(trades),
            "win_rate": 0.0,
            "avg_R": 0.0,
            "median_R": 0.0,
            "max_R": 0.0,
            "min_R": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "expectancy_R": 0.0,
            "win_loss_ratio": 0.0,
        }

    R_arr = np.array(R_vals, dtype=float)
    win_mask = R_arr > 0
    loss_mask = R_arr < 0

    win_R = R_arr[win_mask]
    loss_R = R_arr[loss_mask]

    avg_win_R = win_R.mean() if len(win_R) else 0.0
    avg_loss_R = loss_R.mean() if len(loss_R) else 0.0

    summary = {
        "num_trades": len(R_arr),
        "win_rate": float(win_mask.mean()),
        "avg_R": float(R_arr.mean()),
        "median_R": float(np.median(R_arr)),
        "max_R": float(R_arr.max()),
        "min_R": float(R_arr.min()),
        "avg_win_R": float(avg_win_R),
        "avg_loss_R": float(avg_loss_R),
        "expectancy_R": float(R_arr.mean()),
        "win_loss_ratio": float(avg_win_R / abs(avg_loss_R)) if avg_loss_R < 0 else 0.0,
    }
    return summary


def equity_curve(trades: List[Dict]) -> pd.DataFrame:
    curve = []
    cum_R = 0.0
    for i, t in enumerate(trades, start=1):
        R = t["R"]
        if R is None or math.isnan(R):
            R = 0.0
        cum_R += R
        curve.append({"trade_index": i, "cumulative_R": cum_R})
    return pd.DataFrame(curve)
