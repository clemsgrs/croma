from mari.metrics.ccrr import CrossConfounderRetrievalRatio
from mari.metrics.mari import MarginAwareRobustnessIndex
from mari.metrics.ri import RobustnessIndex

RI = RobustnessIndex
MaRI = MarginAwareRobustnessIndex
CCRR = CrossConfounderRetrievalRatio
__version__ = "0.1.0"

__all__ = [
    "RI",
    "MaRI",
    "CCRR",
    "__version__",
]
