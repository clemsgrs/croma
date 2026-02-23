from mari import MaRI, RI, __version__
from mari.metrics.mari import MarginAwareRobustnessIndex
from mari.metrics.ri import RobustnessIndex


def test_public_api_exports_aliases() -> None:
    assert RI is RobustnessIndex
    assert MaRI is MarginAwareRobustnessIndex
    assert isinstance(__version__, str)
    assert len(__version__) > 0
