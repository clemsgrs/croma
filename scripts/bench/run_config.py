"""What a benchmark run was invoked with, recorded beside that run's own metrics.

A re-run reproduces a published number only if it sweeps the same k grid: the operating
point is chosen from the swept values, so the grid bounds which k a model can select. The
grid is therefore part of the protocol, not a display setting -- and until this file existed
nothing recorded it. It was recoverable only by parsing the ``k_values`` column back out of
``metrics.csv``, and ``reproduce_faithful.py`` -- the documented one-command repro -- passed
no ``--k-grid`` at all, so it swept the *dense* grid over runs the paper had computed on the
*sparse* one. On TCGA-4x4 that moves the operating point from k=71 to k=69.

The fix ADR-0010 points at is *not* to pin the grid in the driver. A constant in the driver
is exactly the second source of truth that ``reproduce_faithful.py``'s own docstring records
removing twice already -- once for a hard-coded model roster that kept the paper at 16 models
after the panel grew to 21, once for a ``PROTOCOL = "k-star"`` pin that rendered tables from
one protocol while the prose macros came from another. So instead: the run records what it
was asked for, and the driver replays the run.

Two blocks, deliberately separated:

``replay``
    The invocation knobs, and the only thing a driver may replay. Every key maps to a
    ``benchmark.py`` flag.
``resolved``
    Facts derived from the benchmark registry or from the sweep itself. Provenance for a
    reader, and something a driver may *check*, but never something it replays -- turning a
    derived fact back into an invocation argument is how the roster and protocol pins went
    wrong. ``benchmark`` and ``protocol`` live here for that reason: ``paper_manifest`` owns
    them, and this file only lets a driver notice when the two disagree.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

#: Bumped when the on-disk shape changes in a way a reader has to notice.
SCHEMA_VERSION = 1

RUN_CONFIG_NAME = "run_config.json"

#: How the k sweep is discretised. Lives here rather than in ``benchmark.py`` because it is
#: precisely the knob a re-run must be told about; ``benchmark.py`` imports it back.
K_GRIDS = ("dense", "sparse")


def resolve_sweep_k_values(k_max: int, grid: str = "dense") -> list[int]:
    """The k values swept, for a ceiling and a grid.

    ``dense`` sweeps every integer ``1..k_max``.

    ``sparse`` reproduces PathoROB's grid, ``[1, 3, 5, 7, 9] + arange(11, k_max, 10)``
    (``robustness_index_utils.get_k_values``). Note ``k_max`` is *exclusive* in the
    arange tail, exactly as upstream: with ``k_max=100`` the largest swept k is 91.
    Values above ``k_max`` are dropped, so small ceilings degrade gracefully.
    """
    if int(k_max) <= 0:
        raise ValueError("k_max must be strictly positive")
    k_max = int(k_max)
    if grid == "dense":
        return list(range(1, k_max + 1))
    if grid == "sparse":
        candidates = [1, 3, 5, 7, 9, *range(11, k_max, 10)]
        values = sorted({k for k in candidates if 1 <= k <= k_max})
        if not values:
            raise ValueError(f"sparse grid is empty for k_max={k_max}")
        return values
    raise ValueError(f"unknown k grid {grid!r}; expected one of {list(K_GRIDS)}")


# --------------------------------------------------------------------------- write / read


def config_path(run_dir: Path) -> Path:
    """Where the sidecar lives, given either the run dir or its ``results/`` subdir.

    Callers hold whichever is convenient -- ``benchmark.py`` has ``results_dir``, a driver
    reading ``paper_manifest`` has ``run_rel``. Accepting both keeps that asymmetry out of
    every call site.
    """
    run_dir = Path(run_dir)
    results_dir = run_dir if run_dir.name == "results" else run_dir / "results"
    return results_dir / RUN_CONFIG_NAME


def write_run_config(*, results_dir: Path, replay: dict, resolved: dict) -> Path:
    """Record the invocation beside the metrics it produced."""
    out_path = config_path(Path(results_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "replay": dict(replay),
        "resolved": dict(resolved),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def read_run_config(run_dir: Path) -> dict | None:
    """The recorded config, or ``None`` if this run predates the sidecar."""
    path = config_path(Path(run_dir))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or "replay" not in payload:
        raise ValueError(f"{path} is not a run config (no 'replay' block)")
    version = int(payload.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{path} was written by a newer croma (schema {version} > {SCHEMA_VERSION}); "
            "upgrade rather than guessing at the fields"
        )
    return payload


def replay_args(config: dict) -> list[str]:
    """The ``benchmark.py`` flags that reproduce the recorded run.

    ``--benchmark`` and ``--protocol`` are deliberately absent: they belong to the benchmark,
    and the caller takes them from ``paper_manifest``. Use :func:`check_resolved` to confirm
    the two still agree.
    """
    replay = dict(config.get("replay") or {})
    missing = [key for key in ("k_max", "k_grid") if key not in replay]
    if missing:
        raise ValueError(f"run config is missing replay key(s) {missing}")
    args = ["--k-max", str(int(replay["k_max"])), "--k-grid", str(replay["k_grid"])]
    # "auto" is the absence of the flag, not a value it accepts: omitting --tau is what puts
    # benchmark.py in per-model auto mode.
    tau = replay.get("tau", "auto")
    if tau not in (None, "auto"):
        # repr(), not %.17g: both round-trip exactly, but repr gives the shortest such form,
        # so a replayed command line reads like the one a human would have typed.
        args += ["--tau", repr(float(tau))]
    for key, flag, cast in (
        ("croma_m_max", "--croma-m-max", int),
        ("croma_start_k", "--croma-start-k", int),
        ("croma_k_growth_factor", "--croma-k-growth-factor", float),
        ("croma_alpha", "--croma-alpha", float),
    ):
        if key in replay:
            value = cast(replay[key])
            args += [flag, str(value) if cast is int else repr(value)]
    return args


def check_resolved(config: dict, *, benchmark: str, protocol: str) -> None:
    """Fail loudly when the manifest and the recorded run disagree about what this is.

    Cheap, but it is the check that would have caught the ``PROTOCOL = "k-star"`` pin: the
    tables were rendered from k-star runs while the macros came from median-k, and nothing
    compared the two.
    """
    resolved = dict(config.get("resolved") or {})
    for key, expected in (("benchmark", benchmark), ("protocol", protocol)):
        found = resolved.get(key)
        if found is not None and str(found) != str(expected):
            raise ValueError(
                f"run config records {key}={found!r} but the manifest says {expected!r}; "
                "the run directory and the manifest have diverged"
            )


# ------------------------------------------------------------- backfill for legacy runs


def _parse_croma_search(signature: str) -> dict:
    """Invert ``metrics_io.croma_search_signature``."""
    parts: dict[str, str] = {}
    for chunk in str(signature).split(";"):
        if "=" in chunk:
            name, _, value = chunk.partition("=")
            parts[name.strip()] = value.strip()
    out: dict[str, float | int] = {}
    if "start" in parts:
        out["croma_start_k"] = int(float(parts["start"]))
    if "growth" in parts:
        out["croma_k_growth_factor"] = float(parts["growth"])
    if "alpha" in parts:
        out["croma_alpha"] = float(parts["alpha"])
    return out


def infer_k_grid(k_values_signature: str, k_max: int) -> str:
    """Which grid produced this ``k_values`` column.

    Exact, not heuristic: the signature is the swept values verbatim, so it either equals
    what a grid generates at this ``k_max`` or it does not.
    """
    swept = sorted({int(v) for v in str(k_values_signature).split(",") if v.strip()})
    for grid in K_GRIDS:
        if swept == resolve_sweep_k_values(int(k_max), grid):
            return grid
    raise ValueError(
        f"k_values={k_values_signature!r} matches no known grid at k_max={k_max}; "
        "this run was swept with a grid croma can no longer generate"
    )


def infer_replay_from_metrics(metrics_csv: Path) -> dict:
    """Recover the invocation of a run written before the sidecar existed.

    Everything needed is already in ``metrics.csv``: it records ``k_max``, the swept
    ``k_values`` verbatim, and the CRoMa search signature. This exists so the runs the paper
    was built from become self-describing without being recomputed -- backfill once, and the
    sidecar is the source of truth from then on.
    """
    metrics_csv = Path(metrics_csv)
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{metrics_csv} has no rows to infer a config from")

    def one(column: str) -> str:
        if column not in rows[0]:
            raise ValueError(f"{metrics_csv} has no {column!r} column")
        values = {str(row[column]) for row in rows}
        if len(values) != 1:
            raise ValueError(
                f"{metrics_csv} mixes {column} values {sorted(values)}; it is not one run"
            )
        return values.pop()

    k_max = int(float(one("k_max")))
    replay: dict = {"k_max": k_max, "k_grid": infer_k_grid(one("k_values"), k_max)}
    replay.update(_parse_croma_search(one("croma_search")))

    # tau: in auto mode every model uses its own median typed-neighbour distance, so the
    # column varies across models; an explicit --tau makes it constant.
    taus = {float(row["tau"]) for row in rows}
    if len(taus) > 1 or len(rows) < 2:
        # A single-model run is genuinely ambiguous -- one row is constant either way. Resolve
        # it to auto: that is what benchmark.py does when the flag is omitted, and the opposite
        # guess is the harmful one, since replaying an explicit tau pins every model to one
        # value and moves MaRI for all of them.
        replay["tau"] = "auto"
    else:
        replay["tau"] = taus.pop()

    m_sweep = metrics_csv.parent / "croma_m_sweep_metrics.csv"
    if m_sweep.exists():
        with m_sweep.open(newline="", encoding="utf-8") as handle:
            m_values = [int(float(r["m"])) for r in csv.DictReader(handle) if r.get("m")]
        if m_values:
            replay["croma_m_max"] = max(m_values)
    return replay
