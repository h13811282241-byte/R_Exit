from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PatternConfig
from .indicators import atr, linear_regression_slope


class RangeDetector:
    def __init__(self, config: PatternConfig):
        self.config = config

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        w = cfg.window_range
        result = pd.DataFrame(index=data.index)

        atr_series = data.get("atr")
        if atr_series is None:
            atr_series = atr(data, cfg.atr_period)

        max_high = data["high"].rolling(w, min_periods=w).max()
        min_low = data["low"].rolling(w, min_periods=w).min()
        range_span = max_high - min_low
        atr_mean = atr_series.rolling(w, min_periods=w).mean()
        slope = linear_regression_slope(data["close"], w)
        span_atr_ratio = range_span / atr_mean

        def _mid_band_ratio(arr: np.ndarray) -> float:
            hi = arr.max()
            lo = arr.min()
            span = hi - lo
            if span <= 0:
                return 0.0
            mid = (hi + lo) / 2
            band = span * cfg.range_mid_band_ratio
            inside = (arr >= mid - band) & (arr <= mid + band)
            return float(inside.mean())

        mid_band_ratio = data["close"].rolling(w, min_periods=w).apply(_mid_band_ratio, raw=True)

        trend_weak = (slope.abs() / atr_mean) <= cfg.range_max_slope_atr
        mid_concentrated = mid_band_ratio >= cfg.range_mid_band_min_ratio

        is_range_narrow = (
            trend_weak
            & mid_concentrated
            & (span_atr_ratio <= cfg.range_narrow_max_span_atr)
            & span_atr_ratio.notna()
        )

        # 避免与窄震荡重叠，宽震荡只接受更大的跨度
        is_range_wide = (
            trend_weak
            & mid_concentrated
            & (span_atr_ratio > cfg.range_narrow_max_span_atr)
            & (span_atr_ratio <= cfg.range_wide_max_span_atr)
            & span_atr_ratio.notna()
        )

        result["is_range_narrow"] = is_range_narrow.fillna(False)
        result["is_range_wide"] = is_range_wide.fillna(False)
        result["range_high"] = max_high.where(result["is_range_narrow"] | result["is_range_wide"])
        result["range_low"] = min_low.where(result["is_range_narrow"] | result["is_range_wide"])
        return result
