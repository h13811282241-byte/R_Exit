from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PatternConfig
from .indicators import atr


class ChannelDetector:
    def __init__(self, config: PatternConfig):
        self.config = config

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        w = cfg.window_channel
        result = pd.DataFrame(index=data.index)

        atr_series = data.get("atr")
        if atr_series is None:
            atr_series = atr(data, cfg.atr_period)

        close = data["close"]
        x = np.arange(w, dtype=float)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum() or 1.0

        def _slope(arr: np.ndarray) -> float:
            y = arr
            y_mean = y.mean()
            return float(((x - x_mean) * (y - y_mean)).sum() / denom)

        def _intercept(arr: np.ndarray) -> float:
            y = arr
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / denom
            return float(y_mean - slope * x_mean)

        def _resid_std(arr: np.ndarray) -> float:
            slope = _slope(arr)
            intercept = _intercept(arr)
            fit = slope * x + intercept
            resid = arr - fit
            return float(np.std(resid))

        def _inliers(arr: np.ndarray) -> float:
            slope = _slope(arr)
            intercept = _intercept(arr)
            fit = slope * x + intercept
            resid = arr - fit
            std = np.std(resid)
            if std <= 0:
                return 1.0
            return float((np.abs(resid) <= std).mean())

        slope_series = close.rolling(w, min_periods=w).apply(_slope, raw=True)
        intercept_series = close.rolling(w, min_periods=w).apply(_intercept, raw=True)
        resid_std = close.rolling(w, min_periods=w).apply(_resid_std, raw=True)
        inliers_ratio = close.rolling(w, min_periods=w).apply(_inliers, raw=True)

        width = resid_std * 2.0
        atr_mean = atr_series.rolling(w, min_periods=w).mean()
        width_atr = width / atr_mean

        slope_norm = slope_series / atr_mean
        inliers_ok = inliers_ratio >= cfg.channel_min_inliers_ratio

        is_up = slope_norm >= cfg.channel_min_slope_atr
        is_down = slope_norm <= -cfg.channel_min_slope_atr

        is_up_narrow = inliers_ok & is_up & (width_atr <= cfg.channel_narrow_width_atr)
        is_up_wide = inliers_ok & is_up & (width_atr > cfg.channel_narrow_width_atr) & (
            width_atr <= cfg.channel_wide_width_atr
        )
        is_down_narrow = inliers_ok & is_down & (width_atr <= cfg.channel_narrow_width_atr)
        is_down_wide = inliers_ok & is_down & (width_atr > cfg.channel_narrow_width_atr) & (
            width_atr <= cfg.channel_wide_width_atr
        )

        x_last = w - 1
        fit_last = slope_series * x_last + intercept_series
        channel_upper = fit_last + width / 2.0
        channel_lower = fit_last - width / 2.0

        active_channel = is_up_narrow | is_up_wide | is_down_narrow | is_down_wide
        result["is_up_narrow_channel"] = is_up_narrow.fillna(False)
        result["is_up_wide_channel"] = is_up_wide.fillna(False)
        result["is_down_narrow_channel"] = is_down_narrow.fillna(False)
        result["is_down_wide_channel"] = is_down_wide.fillna(False)
        result["channel_upper"] = channel_upper.where(active_channel)
        result["channel_lower"] = channel_lower.where(active_channel)
        return result
