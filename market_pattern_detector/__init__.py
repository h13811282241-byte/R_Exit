"""
Market pattern detection package: configuration, indicators, detectors, classifier, and visualization helpers.
"""

from .config import PatternConfig
from .range_detector import RangeDetector
from .channel_detector import ChannelDetector
from .breakout_detector import BreakoutDetector
from .pattern_detector import PatternDetector
from .classifier import PatternClassifier
from .visualize import plot_pattern_sample
from .backtest_utils import pattern_counts, pattern_performance, forward_return

__all__ = [
    "PatternConfig",
    "RangeDetector",
    "ChannelDetector",
    "BreakoutDetector",
    "PatternDetector",
    "PatternClassifier",
    "plot_pattern_sample",
    "pattern_counts",
    "pattern_performance",
    "forward_return",
]
