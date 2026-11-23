"""Every study that resamples the encoder panel draws from one seed.

The same correlation is printed in several of the paper's tables: rho of median CRoMa with
nIPD sits in the downstream, distance, exposure, batch-mixing, AUROC and low-data tables.
While each study kept its own seed, one correlation carried up to three different intervals.
They agree only if every study draws from ``PANEL_SEED`` with its encoders in name order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
for _path in (ROOT / "scripts" / "repro", STUDIES, STUDIES / "apd", STUDIES / "auc"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _exposure  # noqa: E402
import apd_croma_correlation  # noqa: E402
import distance_variants  # noqa: E402
import probe_reductions  # noqa: E402
import report as auc_report  # noqa: E402
import silhouette_baselines  # noqa: E402
from croma.metrics.bootstrap import PANEL_SEED, spearman_inference  # noqa: E402


def test_every_panel_study_draws_from_the_shared_seed() -> None:
    seeds = {
        "apd_croma_correlation": apd_croma_correlation.SEED,
        "distance_variants": distance_variants.SEED,
        "silhouette_baselines": silhouette_baselines.SEED,
        "auc.report": auc_report.SEED,
        "probe_reductions (sensitivity and specificity)": probe_reductions.SENS_SPEC_SEED,
        "probe_reductions (low data)": probe_reductions.LOW_DATA_SEED,
        "_exposure": _exposure._INFERENCE["seed"],
    }
    assert seeds == dict.fromkeys(seeds, PANEL_SEED)


def test_one_seed_and_one_encoder_order_give_one_interval() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=27)
    y = x + rng.normal(scale=0.6, size=27)
    first = spearman_inference(x, y, seed=PANEL_SEED)
    again = spearman_inference(x, y, seed=PANEL_SEED)
    assert (first.ci.lo, first.ci.hi) == (again.ci.lo, again.ci.hi)
    assert first.draws_sha256 == again.draws_sha256

    # The draw indexes positions: the same encoders in another order are another resample.
    order = rng.permutation(27)
    shuffled = spearman_inference(x[order], y[order], seed=PANEL_SEED)
    assert shuffled.rho == first.rho
    assert (shuffled.ci.lo, shuffled.ci.hi) != (first.ci.lo, first.ci.hi)
