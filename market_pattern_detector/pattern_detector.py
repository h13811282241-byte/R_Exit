from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PatternConfig
from .indicators import atr, detect_swings, line_value, regression_from_points


class PatternDetector:
    def __init__(self, config: PatternConfig):
        self.config = config

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        n = len(data)
        result = pd.DataFrame(index=data.index)

        atr_series = data.get("atr")
        if atr_series is None:
            atr_series = atr(data, cfg.atr_period)

        swing_high, swing_low = detect_swings(data["high"], data["low"], cfg.swing_lookback)
        swing_high_idx = np.where(swing_high.to_numpy())[0]
        swing_low_idx = np.where(swing_low.to_numpy())[0]

        wedge_mask = np.zeros(n, dtype=bool)
        wedge_upper = np.full(n, np.nan)
        wedge_lower = np.full(n, np.nan)
        wedge_type = np.array([""] * n, dtype=object)
        wedge_push_highs: list[list[int] | None] = [None] * n
        wedge_push_lows: list[list[int] | None] = [None] * n

        triangle_mask = np.zeros(n, dtype=bool)
        triangle_upper = np.full(n, np.nan)
        triangle_lower = np.full(n, np.nan)

        atr_window_mean = atr_series.rolling(cfg.window_pattern, min_periods=cfg.pattern_min_length).mean()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()

        def find_three_push_sequence(high_idx: np.ndarray, low_idx: np.ndarray, rising: bool) -> tuple[list[int] | None, list[int] | None]:
            """
            返回按时间顺序的三个高点索引和三个低点索引，要求交替顺序:
            上涨三推: low1 < high1 < low2 < high2 < low3 < high3
            下跌三推: high1 > low1 > high2 > low2 > high3 > low3 (时间顺序同样递增索引，但价格递减)
            采用最新完成的序列，间隔与跨度受限。
            """
            expected = ["low", "high", "low", "high", "low", "high"] if rising else ["high", "low", "high", "low", "high", "low"]
            merged = [(int(idx), "high") for idx in high_idx] + [(int(idx), "low") for idx in low_idx]
            merged.sort(key=lambda x: x[0])
            if len(merged) < 6:
                return None, None
            exp_rev = list(reversed(expected))
            seq_rev: list[tuple[int, str]] = []
            last_chosen_idx: int | None = None
            for idx, typ in reversed(merged):
                expected_typ = exp_rev[len(seq_rev)]
                if typ != expected_typ:
                    continue
                if last_chosen_idx is not None and (last_chosen_idx - idx) < cfg.three_push_min_separation:
                    continue
                seq_rev.append((idx, typ))
                last_chosen_idx = idx
                if len(seq_rev) == 6:
                    break
            if len(seq_rev) != 6:
                return None, None
            seq = list(reversed(seq_rev))
            if (seq[-1][0] - seq[0][0]) > cfg.three_push_max_length:
                return None, None
            highs_sel = [idx for idx, typ in seq if typ == "high"]
            lows_sel = [idx for idx, typ in seq if typ == "low"]
            return highs_sel, lows_sel

        for i in range(n):
            start = max(0, i - cfg.window_pattern + 1)
            window_len = i - start + 1
            if window_len < cfg.pattern_min_length:
                continue

            max_pivot_idx = i - cfg.swing_lookback
            if max_pivot_idx < start:
                continue

            sel_high_mask = (swing_high_idx >= start) & (swing_high_idx <= max_pivot_idx)
            sel_low_mask = (swing_low_idx >= start) & (swing_low_idx <= max_pivot_idx)
            sel_high_idx = swing_high_idx[sel_high_mask]
            sel_low_idx = swing_low_idx[sel_low_mask]

            if len(sel_high_idx) < cfg.three_push_min_points or len(sel_low_idx) < cfg.three_push_min_points:
                continue

            highs_seq_idx, lows_seq_idx = find_three_push_sequence(sel_high_idx, sel_low_idx, rising=True)
            rising_seq = highs_seq_idx is not None and lows_seq_idx is not None
            if not rising_seq:
                highs_seq_idx, lows_seq_idx = find_three_push_sequence(sel_high_idx, sel_low_idx, rising=False)
            falling_seq = highs_seq_idx is not None and lows_seq_idx is not None and not rising_seq
            if not (rising_seq or falling_seq):
                continue

            highs_seq_idx_arr = np.array(highs_seq_idx, dtype=float)
            lows_seq_idx_arr = np.array(lows_seq_idx, dtype=float)
            highs_seq_vals = highs[highs_seq_idx_arr.astype(int)]
            lows_seq_vals = lows[lows_seq_idx_arr.astype(int)]

            upper_slope, upper_intercept = regression_from_points(highs_seq_idx_arr, highs_seq_vals)
            lower_slope, lower_intercept = regression_from_points(lows_seq_idx_arr, lows_seq_vals)
            if np.isnan(upper_slope) or np.isnan(lower_slope):
                continue

            d_start = line_value(upper_slope, upper_intercept, highs_seq_idx_arr[0]) - line_value(
                lower_slope, lower_intercept, lows_seq_idx_arr[0]
            )
            d_end = line_value(upper_slope, upper_intercept, i) - line_value(lower_slope, lower_intercept, i)
            if d_start <= 0 or d_end <= 0:
                continue

            atr_mean_window = atr_window_mean.iloc[i]
            if np.isnan(atr_mean_window):
                continue

            converging = (d_end / d_start) <= (1 - cfg.convergence_min_ratio)
            if not converging or d_start < atr_mean_window:
                continue

            if upper_slope > 0 and lower_slope > 0 and rising_seq:
                wedge_mask[i] = True
                wedge_type[i] = "RISING_WEDGE"
                wedge_upper[i] = line_value(upper_slope, upper_intercept, i)
                wedge_lower[i] = line_value(lower_slope, lower_intercept, i)
                wedge_push_highs[i] = highs_seq_idx
                wedge_push_lows[i] = lows_seq_idx
                continue

            if upper_slope < 0 and lower_slope < 0 and falling_seq:
                wedge_mask[i] = True
                wedge_type[i] = "FALLING_WEDGE"
                wedge_upper[i] = line_value(upper_slope, upper_intercept, i)
                wedge_lower[i] = line_value(lower_slope, lower_intercept, i)
                wedge_push_highs[i] = highs_seq_idx
                wedge_push_lows[i] = lows_seq_idx
                continue

            if upper_slope < 0 and lower_slope > 0:
                triangle_mask[i] = True
                triangle_upper[i] = line_value(upper_slope, upper_intercept, i)
                triangle_lower[i] = line_value(lower_slope, lower_intercept, i)

        result["is_wedge"] = wedge_mask
        result["wedge_upper"] = wedge_upper
        result["wedge_lower"] = wedge_lower
        result["wedge_type"] = wedge_type
        result["wedge_push_highs"] = wedge_push_highs
        result["wedge_push_lows"] = wedge_push_lows
        result["is_triangle"] = triangle_mask
        result["triangle_upper"] = triangle_upper
        result["triangle_lower"] = triangle_lower
        return result
