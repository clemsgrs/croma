from croma.alignment import expand_features_to_manifest
from croma.metrics.ccmr import CrossConfounderMarginRatio
from croma.metrics.mari import MarginAwareRobustnessIndex
from croma.metrics.ri import RobustnessIndex

RI = RobustnessIndex
MaRI = MarginAwareRobustnessIndex
CCMR = CrossConfounderMarginRatio
__version__ = "1.0.0"

__all__ = [
    "RI",
    "MaRI",
    "CCMR",
    "expand_features_to_manifest",
    "__version__",
]
