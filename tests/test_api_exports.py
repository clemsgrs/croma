from mari import MaRI, RI, __version__, lower_tail_mean, tail_percentile
from mari.metrics.mari import MarginAwareRobustnessIndex
from mari.metrics.ri import RobustnessIndex


def test_public_api_exports_aliases() -> None:
    assert RI is RobustnessIndex
    assert MaRI is MarginAwareRobustnessIndex
    assert callable(tail_percentile)
    assert callable(lower_tail_mean)
    assert isinstance(__version__, str)
    assert len(__version__) > 0
