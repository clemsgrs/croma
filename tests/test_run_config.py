"""The run config sidecar: what a run was invoked with, and how a re-run recovers it.

The bug these guard against is silent: a re-run that sweeps a different k grid still
produces a full, plausible metrics.csv -- just at a different operating point. Nothing
crashes, so only an explicit check catches it.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_config as rc

# ------------------------------------------------------------------------------- grids


def test_dense_grid_is_every_integer() -> None:
    assert rc.resolve_sweep_k_values(10) == list(range(1, 11))
    assert rc.resolve_sweep_k_values(10, "dense") == list(range(1, 11))


def test_sparse_grid_matches_pathorob() -> None:
    """PathoROB's k_max is exclusive in the arange tail: k_max=100 tops out at 91."""
    assert rc.resolve_sweep_k_values(100, "sparse") == [
        1,
        3,
        5,
        7,
        9,
        11,
        21,
        31,
        41,
        51,
        61,
        71,
        81,
        91,
    ]


def test_unknown_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown k grid"):
        rc.resolve_sweep_k_values(10, "bogus")


# --------------------------------------------------------------------------- inference


@pytest.mark.parametrize("grid", ["dense", "sparse"])
def test_grid_inference_round_trips(grid: str) -> None:
    """A run's ``k_values`` column identifies its grid exactly -- no heuristics involved."""
    signature = ",".join(str(k) for k in rc.resolve_sweep_k_values(100, grid))
    assert rc.infer_k_grid(signature, 100) == grid


def test_grid_inference_rejects_a_grid_it_cannot_generate() -> None:
    with pytest.raises(ValueError, match="matches no known grid"):
        rc.infer_k_grid("2,4,6,8", 100)


def test_dense_and_sparse_are_distinguishable_at_every_realistic_k_max() -> None:
    """If the two grids ever coincided, inference would silently pick the wrong one."""
    for k_max in range(2, 101):
        assert rc.resolve_sweep_k_values(k_max, "dense") != rc.resolve_sweep_k_values(
            k_max, "sparse"
        )


# ------------------------------------------------------------------------- replay args


def test_replay_omits_tau_in_auto_mode() -> None:
    """Auto tau is the *absence* of --tau; passing a number would pin all models to one."""
    args = rc.replay_args({"replay": {"k_max": 100, "k_grid": "sparse", "tau": "auto"}})
    assert "--tau" not in args
    assert args[:4] == ["--k-max", "100", "--k-grid", "sparse"]


def test_replay_passes_an_explicit_tau() -> None:
    args = rc.replay_args({"replay": {"k_max": 50, "k_grid": "dense", "tau": 0.25}})
    assert args[args.index("--tau") + 1] == "0.25"


def test_replay_requires_the_grid() -> None:
    with pytest.raises(ValueError, match="missing replay key"):
        rc.replay_args({"replay": {"k_max": 100}})


def test_replay_renders_floats_readably() -> None:
    """A replayed command line should read like one a human would have typed."""
    args = rc.replay_args(
        {
            "replay": {
                "k_max": 100,
                "k_grid": "sparse",
                "tau": "auto",
                "croma_alpha": 0.1,
                "croma_k_growth_factor": 2.0,
            },
        }
    )
    assert args[args.index("--croma-alpha") + 1] == "0.1"
    assert args[args.index("--croma-k-growth-factor") + 1] == "2.0"


# ------------------------------------------------------------------------ resolved block


def test_resolved_mismatch_is_fatal() -> None:
    """The check that would have caught the PROTOCOL='k-star' pin."""
    config = {"replay": {}, "resolved": {"benchmark": "panda", "protocol": "k-star"}}
    with pytest.raises(ValueError, match="the manifest says"):
        rc.check_resolved(config, benchmark="panda", protocol="median-k")


def test_resolved_absent_is_tolerated() -> None:
    """Backfilled configs may know nothing beyond the replay block."""
    rc.check_resolved({"replay": {}}, benchmark="panda", protocol="k-star")


# ------------------------------------------------------------------------ write / read


def test_write_read_round_trip(tmp_path: Path) -> None:
    replay = {"k_max": 100, "k_grid": "sparse", "tau": "auto"}
    rc.write_run_config(
        results_dir=tmp_path / "results",
        replay=replay,
        resolved={"benchmark": "toy", "protocol": "median-k"},
    )
    loaded = rc.read_run_config(tmp_path)
    assert loaded is not None
    assert loaded["replay"] == replay
    assert loaded["schema_version"] == rc.SCHEMA_VERSION


def test_config_path_accepts_run_dir_or_results_dir(tmp_path: Path) -> None:
    assert rc.config_path(tmp_path) == rc.config_path(tmp_path / "results")


def test_missing_config_reads_as_none(tmp_path: Path) -> None:
    assert rc.read_run_config(tmp_path) is None


def test_a_newer_schema_is_refused_rather_than_guessed_at(tmp_path: Path) -> None:
    path = rc.config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version": 99, "replay": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="newer croma"):
        rc.read_run_config(tmp_path)
