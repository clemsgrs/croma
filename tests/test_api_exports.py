import croma
from croma import CRoMa, MaRI, RI, __version__, apd, napd, probe_sweep
from croma.downstream import apd as downstream_apd
from croma.downstream import napd as downstream_napd
from croma.downstream import probe_sweep as downstream_probe_sweep
from croma.metrics.croma import CrossConfounderRobustnessMargin
from croma.metrics.mari import MarginAwareRobustnessIndex
from croma.metrics.ri import RobustnessIndex


def test_public_api_exports_aliases() -> None:
    assert RI is RobustnessIndex
    assert MaRI is MarginAwareRobustnessIndex
    assert CRoMa is CrossConfounderRobustnessMargin
    assert apd is downstream_apd
    assert napd is downstream_napd
    assert probe_sweep is downstream_probe_sweep
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_public_api_surface_is_exact() -> None:
    # An exact surface proves the legacy flagship alias was dropped (clean break).
    assert set(croma.__all__) == {
        "RI",
        "MaRI",
        "CRoMa",
        "expand_features_to_manifest",
        "apd",
        "napd",
        "probe_sweep",
        "__version__",
    }
    legacy_alias = "C" + "CMR"
    assert not hasattr(croma, legacy_alias)
    assert legacy_alias not in croma.__all__


def test_only_the_protocol_entry_point_is_promoted_to_top_level() -> None:
    # Minimal-first (ADR-0002): the sweep's multi-test form and PathoROB's own schedules
    # are reachable, and documented, but they carry no stability promise -- removing a
    # top-level name later is breaking, so promotion waits for demand.
    for supporting_name in ("probe_sweep_over_test_sets", "pathorob_schedule"):
        assert hasattr(croma.downstream, supporting_name)
        assert not hasattr(croma, supporting_name)


def test_legacy_metric_class_is_removed() -> None:
    legacy_class = "CrossConfounder" + "MarginRatio"
    assert not hasattr(croma.metrics.croma, legacy_class)
