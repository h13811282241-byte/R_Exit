from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PatternConfig
from .indicators import atr


class BreakoutDetector:
    def __init__(self, config: PatternConfig):
        self.config = config

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        result = pd.DataFrame(index=data.index)

        atr_series = data.get("atr")
        if atr_series is None:
            atr_series = atr(data, cfg.atr_period)

        # Most recent boundaries propagated forward; no future data used.
        upper_ref = (
            data.get("range_high", pd.Series(index=data.index, dtype=float)).ffill()
        )
        if "triangle_upper" in data:
            upper_ref = upper_ref.combine_first(data["triangle_upper"].ffill())
        wedge_upper_ref = data.get("wedge_upper", pd.Series(index=data.index, dtype=float)).ffill()
        if "wedge_upper" in data:
            upper_ref = upper_ref.combine_first(wedge_upper_ref)

        lower_ref = (
            data.get("range_low", pd.Series(index=data.index, dtype=float)).ffill()
        )
        if "triangle_lower" in data:
            lower_ref = lower_ref.combine_first(data["triangle_lower"].ffill())
        wedge_lower_ref = data.get("wedge_lower", pd.Series(index=data.index, dtype=float)).ffill()
        if "wedge_lower" in data:
            lower_ref = lower_ref.combine_first(wedge_lower_ref)

        wedge_type_ref = data.get("wedge_type", pd.Series(index=data.index, dtype=object)).replace("", np.nan).ffill()

        body = (data["close"] - data["open"]).abs()
        atr_buffer = cfg.breakout_buffer_atr * atr_series

        volume_ok = pd.Series(True, index=data.index)
        if "volume" in data:
            vol_ma = data["volume"].rolling(20, min_periods=1).mean()
            volume_ok = data["volume"] > cfg.breakout_volume_factor * vol_ma

        bull_raw = (
            upper_ref.notna()
            & (data["close"] > upper_ref + atr_buffer)
            & (body >= cfg.breakout_min_body_atr * atr_series)
            & (data["close"] >= data["high"] - 0.25 * atr_series)
            & volume_ok
        )

        bear_raw = (
            lower_ref.notna()
            & (data["close"] < lower_ref - atr_buffer)
            & (body >= cfg.breakout_min_body_atr * atr_series)
            & (data["close"] <= data["low"] + 0.25 * atr_series)
            & volume_ok
        )

        def _all_true(arr: np.ndarray) -> float:
            return float(np.all(arr))

        def _apply_confirm(mask: pd.Series) -> pd.Series:
            if cfg.breakout_confirm_bars > 1:
                return (
                    mask.rolling(cfg.breakout_confirm_bars, min_periods=cfg.breakout_confirm_bars)
                    .apply(_all_true, raw=True)
                    .astype(bool)
                )
            return mask

        bull_mask = _apply_confirm(bull_raw)
        bear_mask = _apply_confirm(bear_raw)

        # 三推突破：以楔形上下沿为参考
        wedge_bull_raw = (
            wedge_upper_ref.notna()
            & (data["close"] > wedge_upper_ref + atr_buffer)
            & (body >= cfg.breakout_min_body_atr * atr_series)
            & (data["close"] >= data["high"] - 0.25 * atr_series)
            & volume_ok
        )
        wedge_bear_raw = (
            wedge_lower_ref.notna()
            & (data["close"] < wedge_lower_ref - atr_buffer)
            & (body >= cfg.breakout_min_body_atr * atr_series)
            & (data["close"] <= data["low"] + 0.25 * atr_series)
            & volume_ok
        )

        wedge_bull_mask = _apply_confirm(wedge_bull_raw)
        wedge_bear_mask = _apply_confirm(wedge_bear_raw)

        if cfg.breakout_confirm_bars > 1:
            wedge_bull_mask = wedge_bull_mask.astype(bool)
            wedge_bear_mask = wedge_bear_mask.astype(bool)

        result["is_bull_breakout"] = bull_mask.fillna(False)
        result["is_bear_breakout"] = bear_mask.fillna(False)
        result["breakout_reference_upper"] = upper_ref.where(bull_mask)
        result["breakout_reference_lower"] = lower_ref.where(bear_mask)
        result["is_three_push_bull_breakout"] = (wedge_bull_mask & wedge_type_ref.notna()).fillna(False)
        result["is_three_push_bear_breakout"] = (wedge_bear_mask & wedge_type_ref.notna()).fillna(False)
        return result
