from mari.metrics.mari import MarginAwareRobustnessIndex
from mari.metrics.ri import RobustnessIndex
from mari.metrics.tail import lower_tail_mean, tail_percentile

RI = RobustnessIndex
MaRI = MarginAwareRobustnessIndex
__version__ = "0.1.0"

__all__ = [
    "RI",
    "MaRI",
    "__version__",
    "tail_percentile",
    "lower_tail_mean",
]
