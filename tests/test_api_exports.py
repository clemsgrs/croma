from croma import CCMR, MaRI, RI, __version__
from croma.metrics.ccmr import CrossConfounderMarginRatio
from croma.metrics.mari import MarginAwareRobustnessIndex
from croma.metrics.ri import RobustnessIndex


def test_public_api_exports_aliases() -> None:
    assert RI is RobustnessIndex
    assert MaRI is MarginAwareRobustnessIndex
    assert CCMR is CrossConfounderMarginRatio
    assert isinstance(__version__, str)
    assert len(__version__) > 0
