from __future__ import annotations

import numpy as np
import pandas as pd

from .breakout_detector import BreakoutDetector
from .channel_detector import ChannelDetector
from .config import PatternConfig
from .indicators import atr
from .pattern_detector import PatternDetector
from .range_detector import RangeDetector


class PatternClassifier:
    def __init__(self, config: PatternConfig | None = None):
        self.config = config or PatternConfig()
        self.range_detector = RangeDetector(self.config)
        self.channel_detector = ChannelDetector(self.config)
        self.pattern_detector = PatternDetector(self.config)
        self.breakout_detector = BreakoutDetector(self.config)

    def classify(self, data: pd.DataFrame) -> pd.DataFrame:
        working = data.copy()
        if "atr" not in working:
            working["atr"] = atr(working, self.config.atr_period)
        range_df = self.range_detector.detect(working)
        working = working.join(range_df)

        pattern_df = self.pattern_detector.detect(working)
        working = working.join(pattern_df)

        channel_df = self.channel_detector.detect(working)
        working = working.join(channel_df)

        breakout_df = self.breakout_detector.detect(working)
        working = working.join(breakout_df)

        pattern = working.apply(self._classify_row, axis=1)
        if self.config.smoothing_window > 1:
            pattern = self._smooth_pattern(pattern, self.config.smoothing_window)

        working["pattern"] = pattern
        working["pattern_confidence"] = np.where(pattern == "OTHER", 0.0, 1.0)
        return working

    def _classify_row(self, row: pd.Series) -> str:
        flag = lambda name: bool(row.get(name, False))
        if flag("is_three_push_bull_breakout"):
            return "THREE_PUSH_BULL_BREAKOUT"
        if flag("is_three_push_bear_breakout"):
            return "THREE_PUSH_BEAR_BREAKOUT"
        if flag("is_bull_breakout"):
            return "BULL_BREAKOUT"
        if flag("is_bear_breakout"):
            return "BEAR_BREAKOUT"
        if flag("is_wedge"):
            wtype = str(row.get("wedge_type", "") or "")
            if wtype == "RISING_WEDGE":
                return "THREE_PUSH_RISING_WEDGE"
            if wtype == "FALLING_WEDGE":
                return "THREE_PUSH_FALLING_WEDGE"
            return "WEDGE"
        if flag("is_triangle"):
            return "CONTRACTING_TRIANGLE"
        if flag("is_up_narrow_channel"):
            return "UP_NARROW_CHANNEL"
        if flag("is_up_wide_channel"):
            return "UP_WIDE_CHANNEL"
        if flag("is_down_narrow_channel"):
            return "DOWN_NARROW_CHANNEL"
        if flag("is_down_wide_channel"):
            return "DOWN_WIDE_CHANNEL"
        if flag("is_range_wide"):
            return "RANGE_WIDE"
        if flag("is_range_narrow"):
            return "RANGE_NARROW"
        return "OTHER"

    @staticmethod
    def _smooth_pattern(series: pd.Series, window: int) -> pd.Series:
        values = series.to_numpy()
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_vals = values[start : i + 1]
            counts = {}
            for v in window_vals:
                counts[v] = counts.get(v, 0) + 1
            smoothed.append(max(counts.items(), key=lambda kv: kv[1])[0])
        return pd.Series(smoothed, index=series.index)
