from __future__ import annotations

import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_candles(ax, segment: pd.DataFrame):
    x = np.arange(len(segment))
    width = 0.6
    for i, (_, row) in enumerate(segment.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#2ca02c" if c >= o else "#d62728"
        ax.vlines(x=i, ymin=l, ymax=h, color=color, linewidth=1)
        ax.add_patch(
            plt.Rectangle(
                (i - width / 2, min(o, c)),
                width,
                max(abs(c - o), 1e-8),
                color=color,
                alpha=0.4,
            )
        )
    ax.set_xlim(-1, len(segment))


def plot_pattern_sample(
    data: pd.DataFrame,
    pattern_name: str,
    n: int = 5,
    window: int = 60,
    random_state: Optional[int] = None,
) -> List[plt.Figure]:
    """
    Plot random samples of a given pattern for quick visual inspection.
    """
    rng = random.Random(random_state)
    candidates = [idx for idx, val in enumerate(data["pattern"]) if val == pattern_name]
    if not candidates:
        return []
    picks = rng.sample(candidates, k=min(n, len(candidates)))

    figures: List[plt.Figure] = []
    for idx in picks:
        end = idx
        start = max(0, end - window + 1)
        segment = data.iloc[start : end + 1]
        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_candles(ax, segment)
        x = np.arange(len(segment))

        if "range_high" in segment and segment["range_high"].notna().any():
            ax.plot(x, segment["range_high"], label="range_high", color="blue", linestyle="--")
            ax.plot(x, segment["range_low"], label="range_low", color="blue", linestyle="--")
        if "channel_upper" in segment and segment["channel_upper"].notna().any():
            ax.plot(x, segment["channel_upper"], label="channel_upper", color="purple")
            ax.plot(x, segment["channel_lower"], label="channel_lower", color="purple")
        if "triangle_upper" in segment and segment["triangle_upper"].notna().any():
            ax.plot(x, segment["triangle_upper"], label="triangle_upper", color="orange")
            ax.plot(x, segment["triangle_lower"], label="triangle_lower", color="orange")
        if "wedge_upper" in segment and segment["wedge_upper"].notna().any():
            ax.plot(x, segment["wedge_upper"], label="wedge_upper", color="brown")
            ax.plot(x, segment["wedge_lower"], label="wedge_lower", color="brown")

        ax.set_title(f"{pattern_name} @ {segment.index[-1]}")
        ax.legend(loc="best")
        figures.append(fig)
    return figures
