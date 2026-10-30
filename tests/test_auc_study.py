"""The threshold-free (AUROC) arm of the downstream study.

``scripts/studies/auc/`` re-runs the APD study's probe sweep and scores every probe it
trains with AUROC beside balanced accuracy. It owns no protocol: the sweep is the APD
study's own call, so what is tested here is what the AUROC arm adds -- which cells exist,
whether a stored cell still reproduces the APD cell it rides on, and the reduction and
correlations the supplementary table is rendered from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "scripts" / "studies" / "auc"
for _path in (STUDY, ROOT / "scripts" / "studies" / "apd"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_the_cell_list_is_the_apd_studys_cells(tmp_path: Path) -> None:
    # The roster is never declared: a cell exists for exactly the (cohort, encoder) pairs
    # the APD study has a cell for, on the cohorts the downstream correlation covers.
    # Prostate-shift has APD cells but no correlation, so it has no AUROC cells either.
    import auc_cells

    for dataset, model in [
        ("camelyon", "Alpha"),
        ("camelyon", "Beta"),
        ("pcabiop", "Gamma"),
        ("prostate", "Delta"),
    ]:
        _write_json(tmp_path / dataset / f"{model}.json", {})

    cells = auc_cells.expected_cells(apd_dir=tmp_path)

    assert sorted(cells) == [("camelyon", "Alpha"), ("camelyon", "Beta"), ("pcabiop", "Gamma")]


def test_the_command_line_lists_the_work_and_counts_current_cells(tmp_path: Path) -> None:
    # The sweep driver reads its work list from here, and whoever is waiting on a sweep
    # asks here whether it is done: `check` fails until every expected cell is current.
    import subprocess

    id_acc, ood_acc = _accuracies(), _accuracies(start=0.8)
    for model in ("RudolfV 2", "Beta"):
        _apd_cell(tmp_path / "apd" / "camelyon" / f"{model}.json", id_acc, ood_acc)
    _auc_cell(tmp_path / "auc" / "camelyon" / "Beta.json", id_acc, ood_acc)
    dirs = ["--auc-dir", str(tmp_path / "auc"), "--apd-dir", str(tmp_path / "apd")]

    def cli(command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(STUDY / "auc_cells.py"), command, *dirs],
            capture_output=True,
            text=True,
        )

    pending, check = cli("pending"), cli("check")

    # Model names carry spaces, so the list is tab-separated, one cell per line.
    assert (pending.returncode, pending.stdout) == (0, "camelyon\tRudolfV 2\n")
    assert check.returncode == 1
    assert check.stdout.splitlines()[-1] == "1 of 2 expected cells current (1 missing, 0 stale)"


def _accuracies(n_splits: int = 8, iterations: int = 3, start: float = 0.9) -> np.ndarray:
    """A plausible (n_splits, iterations) accuracy matrix that falls with the split."""
    split = np.arange(n_splits)[:, None]
    replicate = np.arange(iterations)[None, :]
    return start - 0.02 * split - 0.001 * replicate


def _apd_cell(path: Path, id_acc: np.ndarray, ood_acc: np.ndarray) -> None:
    """The two matrices of an APD study cell, which is all an AUROC cell is checked against."""
    _write_json(
        path,
        {"id_test_accuracies": id_acc.tolist(), "ood_test_accuracies": ood_acc.tolist()},
    )


def _auc_cell(
    path: Path,
    id_acc: np.ndarray,
    ood_acc: np.ndarray,
    *,
    id_auroc: np.ndarray | None = None,
    ood_auroc: np.ndarray | None = None,
) -> None:
    """A stored AUROC cell: per (split, replicate) AUROC and balanced accuracy, both arms."""
    id_auroc = id_acc + 0.05 if id_auroc is None else id_auroc
    ood_auroc = ood_acc + 0.05 if ood_auroc is None else ood_auroc
    rows = np.stack([id_auroc, id_acc, ood_auroc, ood_acc, np.ones_like(id_acc)], axis=-1)
    _write_json(
        path,
        {
            "columns": ["auroc_id", "balacc_id", "auroc_ood", "balacc_ood", "best_c"],
            "iterations": int(id_acc.shape[1]),
            "rows": rows.tolist(),
        },
    )


@pytest.mark.parametrize(
    ("offset", "state"), [(0.0, "current"), (5e-10, "current"), (2e-9, "stale")]
)
def test_a_cell_is_current_only_while_it_reproduces_the_current_apd_cell(
    tmp_path: Path, offset: float, state: str
) -> None:
    # The check reads the APD cell as it is *now*, not as it was when the AUROC cell was
    # written: a re-preprocessed encoder moves its APD cell, and every AUROC cell riding on
    # the old one must then show up as stale without anyone remembering to look.
    import auc_cells

    id_acc, ood_acc = _accuracies(), _accuracies(start=0.8)
    _apd_cell(tmp_path / "apd" / "camelyon" / "Alpha.json", id_acc, ood_acc)
    drifted = ood_acc.copy()
    drifted[3, 1] += offset
    _auc_cell(tmp_path / "auc" / "camelyon" / "Alpha.json", id_acc, drifted)

    status = auc_cells.cell_status(
        "camelyon", "Alpha", auc_dir=tmp_path / "auc", apd_dir=tmp_path / "apd"
    )

    assert status.state == state
    assert status.gap == pytest.approx(offset, abs=1e-15)


def test_a_cell_is_missing_until_written_and_stale_when_its_apd_cell_changed_shape(
    tmp_path: Path,
) -> None:
    import auc_cells

    id_acc, ood_acc = _accuracies(), _accuracies(start=0.8)
    _apd_cell(tmp_path / "apd" / "camelyon" / "Alpha.json", id_acc, ood_acc)
    _apd_cell(tmp_path / "apd" / "camelyon" / "Beta.json", id_acc, ood_acc)
    # Beta's AUROC cell ran fewer replicates than its APD cell holds now.
    _auc_cell(tmp_path / "auc" / "camelyon" / "Beta.json", id_acc[:, :2], ood_acc[:, :2])

    def state(model: str) -> str:
        return auc_cells.cell_status(
            "camelyon", model, auc_dir=tmp_path / "auc", apd_dir=tmp_path / "apd"
        ).state

    assert state("Alpha") == "missing"
    assert state("Beta") == "stale"


def test_the_work_list_is_every_missing_or_stale_cell_slowest_cohort_first(
    tmp_path: Path,
) -> None:
    # A re-launched sweep resumes: it skips a cell only when it exists *and* is current.
    # Camelyon cells take hours where PCaBiop's take minutes, so a pool that starts on
    # Camelyon finishes soonest; the order is read off the cohorts' sweep sizes.
    import auc_cells

    id_acc, ood_acc = _accuracies(), _accuracies(start=0.8)
    for dataset, model in [
        ("pcabiop", "Missing"),
        ("camelyon", "Current"),
        ("camelyon", "Stale"),
        ("tolkach", "Missing"),
    ]:
        _apd_cell(tmp_path / "apd" / dataset / f"{model}.json", id_acc, ood_acc)
    _auc_cell(tmp_path / "auc" / "camelyon" / "Current.json", id_acc, ood_acc)
    _auc_cell(tmp_path / "auc" / "camelyon" / "Stale.json", id_acc, ood_acc - 0.01)

    pending = auc_cells.pending_cells(auc_dir=tmp_path / "auc", apd_dir=tmp_path / "apd")

    assert pending == [("camelyon", "Stale"), ("tolkach", "Missing"), ("pcabiop", "Missing")]


def _toy_cohort(seed: int = 0):
    """A separable PCaBiop-shaped cohort: 2 centres x 2 classes x 250 one-row slides.

    The centre shifts the embedding too, so a confounded training split has a shortcut to
    take -- the situation the study measures.
    """
    from loaders import Cohort

    rng = np.random.default_rng(seed)
    confounders = np.repeat([0, 0, 1, 1], 250)
    labels = np.tile(np.repeat([0, 1], 250), 2)
    embeddings = rng.normal(size=(1000, 3))
    embeddings[:, 0] += 1.5 * labels
    embeddings[:, 1] += 1.5 * confounders
    ood_labels = np.repeat([0, 1], 60)
    ood_embeddings = rng.normal(size=(120, 3))
    ood_embeddings[:, 0] += 1.5 * ood_labels
    return Cohort(
        embeddings=embeddings,
        confounders=confounders,
        labels=labels,
        ood_embeddings=ood_embeddings,
        ood_labels=ood_labels,
        group_ids=[f"slide-{i}" for i in range(1000)],
    )


def test_a_cell_rides_on_the_apd_sweep_and_reproduces_its_apd_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The AUROC cell restates no protocol: it is the APD study's own sweep call, observed.
    # So an APD cell and an AUROC cell computed from the same cohort agree on every
    # balanced accuracy, exactly, and the AUROC cell is current the moment it is written.
    import apd_experiment
    import auc_cells
    import probe_auc_cell

    monkeypatch.setattr(apd_experiment, "load_data", lambda model, dataset: _toy_cohort())
    published = apd_experiment.compute("Toy", "pcabiop", iterations=2)
    _write_json(tmp_path / "apd" / "pcabiop" / "Toy.json", published)

    outcome = probe_auc_cell.run_cell(
        "pcabiop", "Toy", iterations=2, auc_dir=tmp_path / "auc", apd_dir=tmp_path / "apd"
    )

    status = auc_cells.cell_status(
        "pcabiop", "Toy", auc_dir=tmp_path / "auc", apd_dir=tmp_path / "apd"
    )
    record = json.loads((tmp_path / "auc" / "pcabiop" / "Toy.json").read_text())
    rows = np.asarray(record["rows"])
    assert outcome == "done"
    assert (status.state, status.gap) == ("current", 0.0)
    assert rows.shape == (11, 2, 5)
    # The probe ranks a separable cohort well above chance on both arms.
    assert (rows[0, :, record["columns"].index("auroc_id")] > 0.8).all()
    assert (rows[0, :, record["columns"].index("auroc_ood")] > 0.8).all()


def test_a_current_cell_is_skipped_without_running_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import apd_experiment
    import probe_auc_cell

    id_acc, ood_acc = _accuracies(), _accuracies(start=0.8)
    _apd_cell(tmp_path / "apd" / "camelyon" / "Alpha.json", id_acc, ood_acc)
    _auc_cell(tmp_path / "auc" / "camelyon" / "Alpha.json", id_acc, ood_acc)
    monkeypatch.setattr(
        apd_experiment, "compute", lambda *a, **k: pytest.fail("a current cell was re-run")
    )

    outcome = probe_auc_cell.run_cell(
        "camelyon", "Alpha", auc_dir=tmp_path / "auc", apd_dir=tmp_path / "apd"
    )

    assert outcome == "skip"


def test_a_stale_cell_is_replaced_and_an_unreproducible_one_is_never_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import apd_experiment
    import auc_cells
    import probe_auc_cell

    monkeypatch.setattr(apd_experiment, "load_data", lambda model, dataset: _toy_cohort())
    published = apd_experiment.compute("Toy", "pcabiop", iterations=2)
    id_acc = np.asarray(published["id_test_accuracies"])
    ood_acc = np.asarray(published["ood_test_accuracies"])
    apd_dir, auc_dir = tmp_path / "apd", tmp_path / "auc"
    # "Toy" has a stale AUROC cell (written before its APD cell moved); "Moved" has an APD
    # cell this sweep cannot reproduce -- as when the APD cell was re-run on new embeddings
    # that the AUROC sweep has not been pointed at.
    _write_json(apd_dir / "pcabiop" / "Toy.json", published)
    _auc_cell(auc_dir / "pcabiop" / "Toy.json", id_acc - 0.01, ood_acc)
    _apd_cell(apd_dir / "pcabiop" / "Moved.json", id_acc - 0.01, ood_acc)
    _auc_cell(auc_dir / "pcabiop" / "Moved.json", id_acc - 0.02, ood_acc)
    before = (auc_dir / "pcabiop" / "Moved.json").read_bytes()

    replaced = probe_auc_cell.run_cell(
        "pcabiop", "Toy", iterations=2, auc_dir=auc_dir, apd_dir=apd_dir
    )
    with pytest.raises(ValueError, match=r"pcabiop/Moved.*does not reproduce the APD cell"):
        probe_auc_cell.run_cell("pcabiop", "Moved", iterations=2, auc_dir=auc_dir, apd_dir=apd_dir)

    assert replaced == "done"
    assert auc_cells.cell_status("pcabiop", "Toy", auc_dir=auc_dir, apd_dir=apd_dir).state == (
        "current"
    )
    assert (auc_dir / "pcabiop" / "Moved.json").read_bytes() == before
    assert list((auc_dir / "pcabiop").glob("*.tmp")) == []


@pytest.mark.parametrize("n_classes", [2, 4])
def test_auroc_ignores_a_decision_boundary_shift_that_moves_balanced_accuracy(
    n_classes: int,
) -> None:
    # The study exists to separate threshold drift from lost ranking. Shifting a probe's
    # intercepts moves every decision boundary and so its balanced accuracy, but ranks no
    # sample differently against any class -- on the raw margins, which is why AUROC is
    # taken there rather than on the softmax.
    from probe_auc_cell import auroc
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(n_classes), 200)
    features = rng.normal(size=(len(labels), 3)) + np.eye(n_classes, 3)[labels] * 2.0
    probe = LogisticRegression().fit(features, labels)
    before = auroc(probe, features, labels), balanced_accuracy_score(
        labels, probe.predict(features)
    )

    probe.intercept_ = probe.intercept_ + np.linspace(3.0, -3.0, len(probe.intercept_))
    after = auroc(probe, features, labels), balanced_accuracy_score(labels, probe.predict(features))

    assert after[0] == pytest.approx(before[0], abs=1e-12)
    assert after[1] < before[1] - 0.1


STUDY_FILES = sorted(path for path in STUDY.iterdir() if path.suffix in {".py", ".sh"})
#: The APD study's protocol constants. A copy here would fingerprint a different run the
#: day the APD study changes one.
PROTOCOL_CONSTANTS = {"VAL_FRACTION", "PROBE_SEED", "CANONICAL_ITERATIONS"}


@pytest.mark.parametrize("path", STUDY_FILES, ids=lambda path: path.name)
def test_the_study_names_no_absolute_path(path: Path) -> None:
    # The exploratory driver cd'd into one machine's checkout. Paths resolve relative to
    # the script now, so any checkout runs the study.
    lines = path.read_text(encoding="utf-8").splitlines()
    body = lines[1:] if lines and lines[0].startswith("#!") else lines

    assert str(ROOT) not in "\n".join(lines)
    assert not [line for line in body if "/data/" in line or "/home/" in line]


def test_the_study_keeps_no_hand_maintained_cell_list() -> None:
    assert [path.name for path in STUDY.iterdir() if path.suffix in {".tsv", ".csv"}] == []


@pytest.mark.parametrize(
    "path", [p for p in STUDY_FILES if p.suffix == ".py"], ids=lambda path: path.name
)
def test_the_study_restates_no_protocol_constant(path: Path) -> None:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }

    assert not assigned & PROTOCOL_CONSTANTS


def _falling(v: np.ndarray, start: float, floor: float, depth: float, iterations: int = 2):
    """Matrix whose replicate mean falls linearly from ``start`` by ``depth`` of the way to
    ``floor`` at V = 1. Its nIPD against chance ``floor`` is ``-depth / 2`` exactly."""
    means = start - depth * (start - floor) * np.asarray(v)
    spread = np.linspace(-0.001, 0.001, iterations)
    return means[:, None] + spread[None, :]


def _study_root(root: Path, encoders: dict[str, list[str]], *, metrics: dict | None = None):
    """A checkout-shaped tree: APD cells, current AUROC cells and each cohort's run.

    The first encoder's balanced accuracy falls all the way to chance in domain and half
    way out of domain, and its AUROC does the same against 0.5; each later encoder falls a
    little less. The run's ``metrics.csv`` sits where ``paper_manifest`` says the paper
    reads it.
    """
    import pandas as pd
    from loaders import DATASETS, metric_csv, training_correlations

    for dataset, models in encoders.items():
        v = training_correlations(dataset)
        chance = 1.0 / len(DATASETS[dataset]["biological_classes"])
        for i, model in enumerate(models):
            id_depth, ood_depth = 1.0 - 0.1 * i, 0.5 - 0.05 * i
            id_acc, ood_acc = _falling(v, 0.8, chance, id_depth), _falling(
                v, 0.7, chance, ood_depth
            )
            id_auc, ood_auc = _falling(v, 0.9, 0.5, id_depth), _falling(v, 0.85, 0.5, ood_depth)
            _apd_cell(root / "output/studies/apd" / dataset / f"{model}.json", id_acc, ood_acc)
            _auc_cell(
                root / "output/studies/auc" / dataset / f"{model}.json",
                id_acc,
                ood_acc,
                id_auroc=id_auc,
                ood_auroc=ood_auc,
            )
        run = (metrics or {}).get(dataset) or {
            model: dict(croma=0.1 * i, ri=0.5 + 0.01 * i, mari=0.5 + 0.02 * i, k=11)
            for i, model in enumerate(models)
        }
        path = root / metric_csv(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(model=m, **row) for m, row in run.items()]).to_csv(path, index=False)
    return root


def test_the_reduction_scores_each_regime_against_its_own_chance(tmp_path: Path) -> None:
    # Tolkach-ESCA has six classes: balanced accuracy's chance is 1/6, AUROC's is 0.5
    # whatever the class count. A linear fall to chance at V = 1 has nIPD -0.5 exactly, so
    # both nIPDs are -0.5 in domain only if each score is normalised by its own chance.
    import report

    root = _study_root(tmp_path, {"tolkach": ["Alpha", "DINOv2-B"]})

    reduction = report.reduce(root)

    alpha = reduction[reduction["model"] == "Alpha"].set_index("regime")
    assert list(reduction[["dataset", "model", "regime"]].itertuples(index=False)) == [
        ("tolkach", "Alpha", "id"),
        ("tolkach", "Alpha", "ood"),
        ("tolkach", "DINOv2-B", "id"),
        ("tolkach", "DINOv2-B", "ood"),
    ]
    assert alpha.loc["id", "nipd_auroc"] == pytest.approx(-0.5)
    assert alpha.loc["id", "nipd_balacc"] == pytest.approx(-0.5)
    assert alpha.loc["ood", "nipd_auroc"] == pytest.approx(-0.25)
    assert alpha.loc["ood", "nipd_balacc"] == pytest.approx(-0.25)
    # The V = 0 and V = 1 means, and the normalised endpoint g(1) they give.
    assert alpha.loc["id", ["auroc_v0", "auroc_v1"]].tolist() == pytest.approx([0.9, 0.5])
    assert alpha.loc["id", ["balacc_v0", "balacc_v1"]].tolist() == pytest.approx([0.8, 1 / 6])
    assert alpha.loc["ood", ["endpoint_auroc", "endpoint_balacc"]].tolist() == pytest.approx(
        [-0.5, -0.5]
    )
    # The representation metrics come from the run the paper reports, at its operating k.
    assert alpha.loc["id", ["croma", "ri", "mari", "k"]].tolist() == pytest.approx(
        [0.0, 0.5, 0.5, 11]
    )
    assert reduction.groupby("model")["control"].first().to_dict() == {
        "Alpha": False,
        "DINOv2-B": True,
    }


def _remove_cell(root: Path) -> None:
    (root / "output/studies/auc/camelyon/Beta.json").unlink()


def _drift_cell(root: Path) -> None:
    path = root / "output/studies/auc/camelyon/Beta.json"
    record = json.loads(path.read_text())
    record["rows"][0][0][1] += 1e-6  # in-domain balanced accuracy, split 0, replicate 0
    path.write_text(json.dumps(record))


def _drop_run_row(root: Path) -> None:
    import pandas as pd
    from loaders import metric_csv

    path = root / metric_csv("camelyon")
    frame = pd.read_csv(path)
    frame[frame["model"] != "Beta"].to_csv(path, index=False)


def _add_run_row(root: Path) -> None:
    # The run reports an encoder the downstream study never probed.
    import pandas as pd
    from loaders import metric_csv

    path = root / metric_csv("camelyon")
    frame = pd.read_csv(path)
    extra = frame.iloc[[0]].assign(model="Mettle")
    pd.concat([frame, extra]).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (_remove_cell, r"camelyon/Beta \(missing\)"),
        (_drift_cell, r"camelyon/Beta \(stale\)"),
        (_drop_run_row, r"camelyon/Beta has an AUROC cell but no row in .*metrics.csv"),
        (_add_run_row, r"camelyon: .*metrics.csv reports Mettle, which has no APD cell"),
    ],
    ids=["missing-cell", "stale-cell", "absent-from-run", "unprobed-run-encoder"],
)
def test_the_reduction_never_skips_a_cell(tmp_path: Path, damage, message: str) -> None:
    # A table built from whatever cells happen to be current would silently change its
    # panel; every expected cell is reduced, or nothing is.
    import report

    root = _study_root(tmp_path, {"camelyon": ["Alpha", "Beta", "Gamma"]})
    damage(root)

    with pytest.raises(ValueError, match=message):
        report.reduce(root)


def _synthetic_reduction(panels: dict[str, int], seed: int = 3):
    """A reduction frame: ``n`` ranked encoders per cohort plus, on tile cohorts, a control
    whose values are extreme enough that letting it into a rank correlation would show."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for dataset, n in panels.items():
        models = [f"M{i}" for i in range(n)] + ([] if dataset == "pcabiop" else ["DINOv2-B"])
        for regime in ("id", "ood"):
            base = rng.normal(size=len(models))
            for i, model in enumerate(models):
                control = model == "DINOv2-B"
                rows.append(
                    dict(
                        dataset=dataset,
                        model=model,
                        regime=regime,
                        control=control,
                        croma=9.0 if control else base[i] + rng.normal(scale=0.5),
                        ri=9.0 if control else rng.normal(),
                        mari=9.0 if control else base[i] + rng.normal(scale=1.0),
                        nipd_auroc=-9.0 if control else base[i] + rng.normal(scale=0.3),
                        nipd_balacc=-9.0 if control else base[i] + rng.normal(scale=0.6),
                    )
                )
    return pd.DataFrame(rows)


def test_the_correlations_cover_both_scores_and_their_agreement_over_ranked_encoders() -> None:
    from scipy.stats import spearmanr

    import report

    reduction = _synthetic_reduction({"camelyon": 10, "tolkach": 10, "pcabiop": 5})

    corr = report.correlate(reduction, n_boot=200, n_permutations=500)

    # Per cohort and regime: three metrics x two scores, then the agreement row.
    camelyon_id = corr[(corr["dataset"] == "camelyon") & (corr["regime"] == "id")]
    assert list(zip(camelyon_id["kind"], camelyon_id["x"], camelyon_id["y"])) == [
        ("association", "croma", "nipd_auroc"),
        ("association", "croma", "nipd_balacc"),
        ("association", "ri", "nipd_auroc"),
        ("association", "ri", "nipd_balacc"),
        ("association", "mari", "nipd_auroc"),
        ("association", "mari", "nipd_balacc"),
        ("agreement", "nipd_auroc", "nipd_balacc"),
    ]
    assert len(corr) == 3 * 2 * 7
    # The control is a floor, not a ranked encoder: it enters no rho.
    ranked = reduction[~reduction["control"]]
    for row in corr.itertuples():
        panel = ranked[(ranked["dataset"] == row.dataset) & (ranked["regime"] == row.regime)]
        assert row.n == len(panel)
        assert row.rho == pytest.approx(spearmanr(panel[row.x], panel[row.y]).statistic)
        assert row.ci_lo <= row.ci_hi
    # PCaBiop's five encoders get the exact null over all 120 orderings, and the tile
    # cohorts the seeded Monte-Carlo one.
    slide = corr["dataset"] == "pcabiop"
    assert set(corr.loc[slide, "permutation"]) == {"exact"}
    assert set(corr.loc[slide, "n_permutations"]) == {120}
    assert set(corr.loc[~slide, "permutation"]) == {"monte-carlo"}
    assert set(corr.loc[~slide, "n_boot"]) == {200}


def test_the_bh_family_is_every_tile_association_and_nothing_else() -> None:
    from croma.metrics.bootstrap import benjamini_hochberg

    import report

    corr = report.correlate(
        _synthetic_reduction({"camelyon": 10, "tolkach": 10, "pcabiop": 5}),
        n_boot=200,
        n_permutations=500,
    )

    family = (corr["kind"] == "association") & (corr["dataset"] != "pcabiop")
    assert corr["in_family"].tolist() == family.tolist()
    assert int(family.sum()) == 2 * 2 * 3 * 2  # tile cohorts x regimes x metrics x scores
    # Agreement rows and the descriptive PCaBiop column carry no q value.
    assert corr.loc[~family, "q_value"].isna().all()
    assert corr.loc[family, "q_value"].to_numpy() == pytest.approx(
        benjamini_hochberg(corr.loc[family, "p_value"].to_numpy())
    )


def test_the_correlations_are_reproducible_from_their_seed() -> None:
    import report

    reduction = _synthetic_reduction({"camelyon": 10, "pcabiop": 5})

    first = report.correlate(reduction, n_boot=200, n_permutations=500)
    second = report.correlate(reduction, n_boot=200, n_permutations=500)

    assert first.equals(second)


def test_the_study_writes_the_reduction_and_the_correlations_it_reports(
    tmp_path: Path,
) -> None:
    import pandas as pd

    import report

    root = _study_root(tmp_path, {"camelyon": ["Alpha", "Beta", "Gamma", "Delta", "DINOv2-B"]})

    reduction, corr = report.main(root)

    written = pd.read_csv(root / "output/studies/auc/nipd_auroc.csv")
    written_corr = pd.read_csv(root / "output/studies/auc/nipd_auroc_correlation.csv")
    assert len(written) == 5 * 2
    pd.testing.assert_frame_equal(written, reduction, check_dtype=False)
    pd.testing.assert_frame_equal(written_corr, corr, check_dtype=False)
    assert set(corr["n"]) == {4}


def _sweep_tree(tmp_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Three APD cells, one of which already has its current AUROC cell."""
    id_acc, ood_acc = _accuracies(), _accuracies(start=0.8)
    for dataset, model in [("pcabiop", "Gamma"), ("camelyon", "Alpha"), ("camelyon", "Beta")]:
        _apd_cell(tmp_path / "apd" / dataset / f"{model}.json", id_acc, ood_acc)
    _auc_cell(tmp_path / "auc" / "camelyon" / "Beta.json", id_acc, ood_acc)
    return dict(auc_dir=tmp_path / "auc", apd_dir=tmp_path / "apd"), id_acc, ood_acc


def test_the_sweep_runs_each_pending_cell_once_and_then_has_nothing_to_do(
    tmp_path: Path,
) -> None:
    import sweep

    dirs, id_acc, ood_acc = _sweep_tree(tmp_path)
    ran = []

    def runner(dataset: str, model: str) -> int:
        ran.append((dataset, model))
        _auc_cell(dirs["auc_dir"] / dataset / f"{model}.json", id_acc, ood_acc)
        return 0

    assert sweep.run(jobs=1, runner=runner, **dirs) == 0
    # The current cell is left alone, and the slowest cohort goes first.
    assert ran == [("camelyon", "Alpha"), ("pcabiop", "Gamma")]
    # Resumable: a second sweep finds every cell current and runs nothing.
    assert sweep.run(jobs=2, runner=runner, **dirs) == 0
    assert len(ran) == 2


def test_the_sweep_fails_until_every_cell_is_current(tmp_path: Path) -> None:
    # Success is the cells' state, not the workers' exit codes: a worker that exits 0
    # without leaving a current cell still fails the sweep.
    import sweep

    dirs, id_acc, ood_acc = _sweep_tree(tmp_path)

    def runner(dataset: str, model: str) -> int:
        if dataset == "pcabiop":
            _auc_cell(dirs["auc_dir"] / dataset / f"{model}.json", id_acc + 1e-6, ood_acc)
        return 0

    assert sweep.run(jobs=1, runner=runner, **dirs) == 1


def test_the_full_reproduction_runs_the_study_after_the_apd_cells_it_rides_on() -> None:
    repro = ROOT / "scripts" / "repro"
    if str(repro) not in sys.path:
        sys.path.insert(0, str(repro))
    import reproduce_faithful

    labels = [label for label, _, _ in reproduce_faithful.STUDIES]
    scripts = {label: script for label, script, _ in reproduce_faithful.STUDIES}

    assert labels.index("apd-sweep") < labels.index("auc-sweep") < labels.index("auc-report")
    assert scripts["auc-sweep"] == "scripts/studies/auc/sweep.py"
    assert scripts["auc-report"] == "scripts/studies/auc/report.py"
