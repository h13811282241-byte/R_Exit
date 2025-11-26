from dataclasses import dataclass


@dataclass
class PatternConfig:
    # Core indicator windows
    atr_period: int = 14
    window_range: int = 40
    window_channel: int = 40
    window_pattern: int = 60

    # Range (narrow / wide) parameters
    range_max_slope_atr: float = 0.2
    range_mid_band_ratio: float = 0.3
    range_mid_band_min_ratio: float = 0.45
    range_narrow_max_span_atr: float = 4.0
    range_wide_min_span_atr: float = 2.0
    range_wide_max_span_atr: float = 5.0

    # Channel parameters
    channel_min_slope_atr: float = 0.15
    channel_narrow_width_atr: float = 2.0
    channel_wide_width_atr: float = 4.0
    channel_min_inliers_ratio: float = 0.7

    # Breakout parameters
    breakout_buffer_atr: float = 0.3
    breakout_confirm_bars: int = 1
    breakout_min_body_atr: float = 0.5
    breakout_volume_factor: float = 1.5

    # Swing / pattern detection
    swing_lookback: int = 3
    pattern_min_swings: int = 4
    convergence_min_ratio: float = 0.3
    pattern_min_length: int = 20

    # Three-push wedge specifics
    three_push_min_points: int = 3
    three_push_min_separation: int = 1
    three_push_max_length: int = 50

    # Optional label smoothing (majority vote over past N bars, uses only history)
    smoothing_window: int = 1
