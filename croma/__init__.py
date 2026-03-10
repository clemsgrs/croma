from croma.metrics.ccmr import CrossConfounderMarginRatio
from croma.metrics.mari import MarginAwareRobustnessIndex
from croma.metrics.ri import RobustnessIndex

RI = RobustnessIndex
MaRI = MarginAwareRobustnessIndex
CCMR = CrossConfounderMarginRatio
__version__ = "0.1.0"

__all__ = [
    "RI",
    "MaRI",
    "CCMR",
    "__version__",
]
