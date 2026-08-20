from croma.alignment import expand_features_to_manifest
from croma.downstream import apd, nipd, probe_sweep
from croma.metrics.croma import CrossConfounderRobustnessMargin
from croma.metrics.mari import MarginAwareRobustnessIndex
from croma.metrics.ri import RobustnessIndex

RI = RobustnessIndex
MaRI = MarginAwareRobustnessIndex
CRoMa = CrossConfounderRobustnessMargin
__version__ = "1.0.0"

__all__ = [
    "RI",
    "MaRI",
    "CRoMa",
    "expand_features_to_manifest",
    "apd",
    "nipd",
    "probe_sweep",
    "__version__",
]
