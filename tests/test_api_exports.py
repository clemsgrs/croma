import croma
from croma import CRoMa, MaRI, RI, __version__, napd
from croma.downstream import napd as downstream_napd
from croma.metrics.croma import CrossConfounderRobustnessMargin
from croma.metrics.mari import MarginAwareRobustnessIndex
from croma.metrics.ri import RobustnessIndex


def test_public_api_exports_aliases() -> None:
    assert RI is RobustnessIndex
    assert MaRI is MarginAwareRobustnessIndex
    assert CRoMa is CrossConfounderRobustnessMargin
    assert napd is downstream_napd
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_public_api_surface_is_exact() -> None:
    # An exact surface proves the legacy flagship alias was dropped (clean break).
    assert set(croma.__all__) == {
        "RI",
        "MaRI",
        "CRoMa",
        "expand_features_to_manifest",
        "napd",
        "__version__",
    }
    legacy_alias = "C" + "CMR"
    assert not hasattr(croma, legacy_alias)
    assert legacy_alias not in croma.__all__


def test_legacy_metric_class_is_removed() -> None:
    legacy_class = "CrossConfounder" + "MarginRatio"
    assert not hasattr(croma.metrics.croma, legacy_class)
