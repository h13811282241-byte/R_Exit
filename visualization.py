#!/usr/bin/env python3
"""
简单可视化：价格 + 信号标记，权益曲线。
"""

from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_with_signals(df: pd.DataFrame, trades: List[Dict], out_file: Optional[str] = None):
    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["close"], label="Close", color="black", linewidth=1)

    long_entries_x = []
    long_entries_y = []
    short_entries_x = []
    short_entries_y = []
    for t in trades:
        idx = t["entry_idx"]
        if idx >= len(df):
            continue
        ts = df.loc[idx, "timestamp"]
        price = t["entry"]
        if t["side"] == "long":
            long_entries_x.append(ts)
            long_entries_y.append(price)
        else:
            short_entries_x.append(ts)
            short_entries_y.append(price)

    if long_entries_x:
        plt.scatter(long_entries_x, long_entries_y, marker="^", color="green", label="Long entry", zorder=3)
    if short_entries_x:
        plt.scatter(short_entries_x, short_entries_y, marker="v", color="red", label="Short entry", zorder=3)

    plt.xticks(rotation=45)
    plt.title("Price with entry signals")
    plt.legend()
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=150)
    plt.close()


def plot_equity_curve(trades: List[Dict], out_file: Optional[str] = None):
    cum = 0.0
    xs = []
    ys = []
    for i, t in enumerate(trades, start=1):
        R = t.get("R", 0) or 0
        cum += R
        xs.append(i)
        ys.append(cum)

    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, label="Cumulative R", color="blue")
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative R")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=150)
    plt.close()
