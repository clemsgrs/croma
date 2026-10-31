"""The distance-variants study: whitening, the cosine reproduction check, and the measures.

Paper-free on purpose (it stays in the sdist): nothing here imports ``scripts/repro`` or
reads ``output/``. The table and macros built from the study are tested in
``test_distance_variants_table.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "scripts" / "studies", ROOT / "scripts" / "bench"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import distance_variants as dv  # noqa: E402

# --------------------------------------------------------------------------------------
# Whitening.
# --------------------------------------------------------------------------------------

#: Population variances along six orthogonal directions; cumulative shares 0.50, 0.80,
#: 0.94, 0.98, 0.995, 1.0. So 90% needs three components, 95% four and 99% five.
_SPECTRUM = np.array([50.0, 30.0, 14.0, 4.0, 1.5, 0.5])


def _known_spectrum(n: int = 400, seed: int = 3) -> np.ndarray:
    """Rows whose *sample* covariance is exactly ``R diag(_SPECTRUM) R^T`` for a rotation R.

    Centred orthonormal columns scaled by ``sqrt(n - 1)`` have identity sample covariance,
    so the spectrum is the sample spectrum, not merely its expectation.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, len(_SPECTRUM)))
    q, _ = np.linalg.qr(raw - raw.mean(axis=0))
    rotation, _ = np.linalg.qr(rng.standard_normal((len(_SPECTRUM),) * 2))
    return 7.0 + np.sqrt(n - 1) * (q * np.sqrt(_SPECTRUM)) @ rotation.T


@pytest.mark.parametrize("variance, expected", [(0.90, 3), (0.95, 4), (0.99, 5)])
def test_whitening_keeps_the_fewest_components_reaching_the_variance_share(variance, expected):
    whitened = dv.whiten(_known_spectrum(), variance)

    assert whitened.n_components == expected
    assert whitened.explained == pytest.approx(np.cumsum(_SPECTRUM)[expected - 1] / 100.0)
    assert whitened.features.shape == (400, expected)


def test_whitened_components_have_identity_covariance():
    whitened = dv.whiten(_known_spectrum(), 0.95)

    covariance = np.cov(whitened.features, rowvar=False, ddof=1)
    np.testing.assert_allclose(covariance, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(whitened.features.mean(axis=0), 0.0, atol=1e-10)


def test_a_tile_scored_in_several_subsets_is_whitened_once_not_weighted_by_its_count():
    tiles = _known_spectrum(n=60)
    occurrences = np.r_[np.arange(60), np.zeros(40, dtype=int)]  # tile 0 occurs 41 times
    variant = dv.variant("whitened")

    once, _ = dv.prepare(tiles, np.arange(60), variant)
    repeated, record = dv.prepare(tiles[occurrences], occurrences, variant)

    np.testing.assert_allclose(repeated, once[occurrences], atol=1e-12)
    assert record["whitening_rows"] == 60
    assert record["n_components"] == 4


def test_normalised_euclidean_scores_unit_rows():
    rows = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    prepared, _ = dv.prepare(rows, np.arange(2), dv.variant("normalised-euclidean"))

    np.testing.assert_allclose(prepared, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)


# --------------------------------------------------------------------------------------
# The measures.
# --------------------------------------------------------------------------------------


def _summary() -> pd.DataFrame:
    """Four ranked encoders and the control, scored under cosine and one variant.

    By hand: the variant keeps the median ranking (rho = 1) and reverses the LTM10 one
    (rho = -1); E2's median crosses zero (0.10 -> -0.05), and so does the control's,
    which must not count; |delta F(0)| is 0.05, 0.10, 0, 0.20, median 0.075; the
    per-sample sign agreements 0.9, 0.8, 0.95, 0.7 have median 0.85.
    """
    rows = []
    cosine = {
        "E1": (0.30, 0.10, -0.40, 1.0),
        "E2": (0.10, 0.20, -0.30, 1.0),
        "E3": (-0.10, 0.60, -0.20, 1.0),
        "E4": (-0.20, 0.70, -0.10, 1.0),
        dv.CONTROL_MODEL: (0.05, 0.40, -0.50, 1.0),
    }
    other = {
        "E1": (0.25, 0.15, -0.10, 0.90),
        "E2": (-0.05, 0.30, -0.20, 0.80),
        "E3": (-0.15, 0.60, -0.30, 0.95),
        "E4": (-0.30, 0.50, -0.40, 0.70),
        dv.CONTROL_MODEL: (-0.10, 0.60, -0.90, 0.10),
    }
    for name, table in (("cosine", cosine), ("euclidean", other)):
        for model, (median, f0, ltm10, agreement) in table.items():
            rows.append(
                {
                    "benchmark": "toy",
                    "model": model,
                    "variant": name,
                    "median": median,
                    "f0": f0,
                    "ltm10": ltm10,
                    "sign_agreement": agreement,
                    "n_components": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _row(table: pd.DataFrame, variant: str) -> pd.Series:
    return table.set_index("variant").loc[variant]


def test_measures_compare_each_variant_with_cosine_over_the_ranked_encoders():
    row = _row(dv.measures(_summary()), "euclidean")

    assert row["n_encoders"] == 4
    assert row["rank_rho_median"] == pytest.approx(1.0)
    assert (row["rank_rho_median_lo"], row["rank_rho_median_hi"]) == pytest.approx((1.0, 1.0))
    assert row["rank_rho_ltm10"] == pytest.approx(-1.0)
    assert row["sign_changes"] == 1
    assert row["median_abs_delta_f0"] == pytest.approx(0.075)
    assert row["median_sign_agreement"] == pytest.approx(0.85)


def test_the_nipd_association_is_reported_beside_cosines_with_an_exact_permutation_p():
    nipd = pd.DataFrame(
        {
            "benchmark": ["toy"] * 5,
            "model": ["E1", "E2", "E3", "E4", dv.CONTROL_MODEL],
            "nipd_id": [-0.01, -0.02, -0.03, -0.04, -0.5],
            "nipd_ood": [-0.04, -0.03, -0.02, -0.01, 0.5],
        }
    )

    table = dv.measures(_summary(), nipd)

    for variant in ("cosine", "euclidean"):
        row = _row(table, variant)
        assert row["rho_nipd_id"] == pytest.approx(1.0)
        assert row["rho_nipd_ood"] == pytest.approx(-1.0)
        # Four encoders: all 24 orderings, of which the identity and the reversal reach |rho| = 1.
        assert row["rho_nipd_id_p"] == pytest.approx(2 / 24)
    assert np.isnan(_row(table, "cosine")["rank_rho_median"])


def test_sign_agreement_reads_zero_as_confounder_dominant_as_f0_does():
    assert dv.sign_agreement([0.0, 0.2, -0.1, 0.3], [-0.2, 0.0, -0.3, 0.1]) == 0.75


# --------------------------------------------------------------------------------------
# One encoder x benchmark, end to end against a real driver run.
# --------------------------------------------------------------------------------------

_CELLS = [(label, centre) for label in ("A", "B") for centre in ("C1", "C2")]


def _toy_run(bench_env):
    """A 4 x 10-tile benchmark scored by the driver at m up to 5, as a production run is."""
    rng = np.random.default_rng(5)
    rows, labels, centres = [], [], []
    axes = {key: rng.standard_normal(12) for key in ("A", "B", "C1", "C2")}
    for label, centre in _CELLS:
        for _ in range(10):
            rows.append(axes[label] + 0.8 * axes[centre] + rng.standard_normal(12))
            labels.append(label)
            centres.append(centre)
    n = len(rows)
    tileset = pd.DataFrame(
        {
            "sample_id": [f"t{i}" for i in range(n)],
            "image_path": [f"/tmp/t{i}.png" for i in range(n)],
            "label": labels,
            "medical_center": centres,
            "group_id": [f"slide{i}" for i in range(n)],
        }
    )
    bench_env.write_tileset("toy-tiles", tileset, {"M1": np.asarray(rows)})
    bench_env.register("toy", tileset="toy-tiles", manifest=tileset.assign(dataset="toy"), k_max=5)
    assert bench_env.run("toy", "k-star", "--croma-m-max", "5", "--progress", "off") == 0
    results_dir = bench_env.results_dir("toy")
    run = dv.Run(
        benchmark="toy",
        design="all",
        results_dir=results_dir,
        search=dv.Search.from_run_config(results_dir / "run_config.json"),
    )
    from views import load_view

    return run, load_view("toy")


def _run_unit(root, run, view, *, last=False):
    return dv.run_unit(
        root,
        run,
        "M1",
        features=view.features("M1"),
        rows=view.rows,
        manifest=view.eval_manifest,
        kernel="test",
        last=last,
        embedding_sha256="0" * 64,
    )


def test_the_cosine_variant_reproduces_the_runs_stored_values_bit_for_bit(bench_env, tmp_path):
    run, view = _toy_run(bench_env)
    root = tmp_path / "study"

    assert _run_unit(root, run, view) == 0

    cosine = dv.read_cell(root, "toy", "M1", "cosine")
    assert cosine["reproduced"] is True
    assert all(cosine["reproduction"]["pooled_identical"].values())
    stored = pd.read_csv(run.per_sample_csv("M1"), float_precision="round_trip")
    values = dv.read_cell_values(root, "toy", "M1", "cosine")
    assert values["sample_id"].tolist() == stored["sample_id"].astype(str).tolist()
    assert values["croma"].tolist() == stored["croma_m5"].tolist()
    for name in ("normalised-euclidean", "euclidean", "whitened"):
        cell = dv.read_cell(root, "toy", "M1", name)
        assert cell["n_units"] == 40 and cell["kernel"] == "test"
    assert 1 <= dv.read_cell(root, "toy", "M1", "whitened")["n_components"] <= 12


def test_a_reference_that_does_not_reproduce_blocks_the_summary(bench_env, tmp_path):
    run, view = _toy_run(bench_env)
    stored = pd.read_csv(run.per_sample_csv("M1"), float_precision="round_trip")
    stored.loc[3, "croma_m5"] = np.nextafter(stored.loc[3, "croma_m5"], 2.0)
    stored.to_csv(run.per_sample_csv("M1"), index=False)
    root = tmp_path / "study"

    assert _run_unit(root, run, view) == dv.NOT_REPRODUCED
    assert dv.read_cell(root, "toy", "M1", "cosine") is None

    assert _run_unit(root, run, view, last=True) == 0
    cosine = dv.read_cell(root, "toy", "M1", "cosine")
    assert cosine["reproduced"] is False
    assert cosine["reproduction"]["n_mismatched"] == 1
    record = {
        "benchmarks": {
            "toy": {
                "results_dir": str(run.results_dir),
                "models": ["M1"],
                "variants": [{"name": v.name} for v in dv.VARIANTS],
            }
        }
    }
    (root / dv.DESIGN).write_text("{}")
    with pytest.raises(dv.IncompleteStudyError, match="does not reproduce"):
        dv.summarise(root, record, apd_correlation_csv=None)
