"""The ``all`` evaluation design: the default, and the only alternative to ``paired_2x2``.

``all`` scores every supplied manifest row as one evaluation scope. It replaces the
former ``dataset_wide`` outright -- there is no alias and no migration shim -- and it is
what every public entry point selects when the caller says nothing.
"""

import json

import numpy as np
import pandas as pd
import pytest

from croma import CRoMa, MaRI, RI
from croma.metrics.base import EVALUATION_DESIGN_ALL, VALID_EVALUATION_DESIGNS
from croma.types import CRoMaResult, RobustnessResult


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a_c1", "a_c2", "b_c1", "b_c2"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V2", "V1", "V2"],
            "group_id": [f"slide-{i}" for i in range(4)],
            "dataset": ["toy"] * 4,
        }
    )


def _features() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.00, 1.00],
            [0.01, 0.99],
        ],
        dtype=float,
    )


def _paired_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "scanner_vendor": ["V1", "V1", "V2", "V2", "V1", "V1", "V2", "V2"],
            "group_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
            "subset": ["pair0"] * 8,
        }
    )


def _paired_features() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.92, 0.08, 0.00, 0.00],
            [0.90, 0.10, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.05, 0.95, 0.00, 0.00],
            [0.08, 0.92, 0.00, 0.00],
            [0.10, 0.90, 0.00, 0.00],
        ],
        dtype=float,
    )


def test_accepted_designs_are_exactly_all_and_paired_2x2() -> None:
    assert EVALUATION_DESIGN_ALL == "all"
    assert set(VALID_EVALUATION_DESIGNS) == {"all", "paired_2x2"}


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_compute_defaults_to_all(metric_cls) -> None:
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=_features(),
        manifest=_manifest(),
        confounder_column="scanner_vendor",
        k_candidates=[1],
        **kwargs,
    )
    assert result.evaluation_design == "all"
    assert result.evaluation_unit == "sample"


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_compute_curve_defaults_to_all(metric_cls) -> None:
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    default_curve = metric_cls.compute_curve(
        features=_features(),
        manifest=_manifest(),
        confounder_column="scanner_vendor",
        k_values=[1],
        **kwargs,
    )
    explicit_curve = metric_cls.compute_curve(
        features=_features(),
        manifest=_manifest(),
        confounder_column="scanner_vendor",
        k_values=[1],
        evaluation_design="all",
        **kwargs,
    )
    assert default_curve == explicit_curve


def test_recommend_tau_defaults_to_all() -> None:
    default_tau = MaRI.recommend_tau(
        _features(),
        _manifest(),
        confounder_column="scanner_vendor",
        k=1,
    )
    explicit_tau = MaRI.recommend_tau(
        _features(),
        _manifest(),
        confounder_column="scanner_vendor",
        k=1,
        evaluation_design="all",
    )
    assert default_tau == pytest.approx(explicit_tau)


def test_croma_compute_defaults_to_all() -> None:
    result = CRoMa.compute(
        features=_features(),
        manifest=_manifest(),
        confounder_column="scanner_vendor",
        m=1,
    )
    assert result.evaluation_design == "all"
    assert result.evaluation_unit == "sample"


def test_result_dataclass_defaults_are_all() -> None:
    assert RobustnessResult.evaluation_design == "all"
    assert RobustnessResult.evaluation_unit == "sample"
    assert CRoMaResult.evaluation_design == "all"
    assert CRoMaResult.evaluation_unit == "sample"


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_all_reproduces_the_full_manifest_golden_value(metric_cls) -> None:
    """The golden numbers are the ones ``dataset_wide`` produced; only the name changed."""
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=_features(),
        manifest=_manifest(),
        confounder_column="scanner_vendor",
        evaluation_design="all",
        k_candidates=[1],
        **kwargs,
    )
    assert result.k == 1
    assert result.value == pytest.approx(1.0)
    assert result.std == pytest.approx(0.0)
    assert result.n_pairs == 1
    assert result.occurrence_source_indices.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_dataset_wide_is_rejected_naming_only_the_two_designs(metric_cls) -> None:
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    with pytest.raises(ValueError) as excinfo:
        metric_cls.compute(
            features=_features(),
            manifest=_manifest(),
            confounder_column="scanner_vendor",
            evaluation_design="dataset_wide",
            k_candidates=[1],
            **kwargs,
        )
    message = str(excinfo.value)
    assert "dataset_wide" not in message
    assert "'all'" in message
    assert "'paired_2x2'" in message


def test_croma_rejects_dataset_wide() -> None:
    with pytest.raises(ValueError, match="paired_2x2"):
        CRoMa.compute(
            features=_features(),
            manifest=_manifest(),
            confounder_column="scanner_vendor",
            evaluation_design="dataset_wide",
            m=1,
        )


def test_explicit_paired_2x2_is_unchanged() -> None:
    result = RI.compute(
        features=_paired_features(),
        manifest=_paired_manifest(),
        confounder_column="scanner_vendor",
        evaluation_design="paired_2x2",
        k_candidates=[1, 3],
    )
    assert result.evaluation_design == "paired_2x2"
    assert result.evaluation_unit == "occurrence"
    assert result.n_pairs == 1


def test_cache_key_identity_distinguishes_all_from_paired_and_from_stale_dataset_wide() -> None:
    from metrics_cache import build_cache_key

    def key_hash(design: str) -> str:
        return build_cache_key(
            artifact_name="ri_summary",
            model="M1",
            input_fingerprint={"manifest_fingerprint": "mf", "embedding_fingerprint": "ef"},
            params={
                "evaluation_design": design,
                "k_values": [5],
                "confounder_column": "center",
            },
        )["key_hash"]

    assert key_hash("all") != key_hash("paired_2x2")
    assert key_hash("all") != key_hash("dataset_wide")


def test_a_cached_summary_that_does_not_name_its_design_is_not_reused() -> None:
    import benchmark as bm

    payload = {
        "k": 5,
        "value": 0.6,
        "std": 0.1,
        "undefined_frac": 0.0,
        "median_value": 0.5,
        "q_alpha": -0.2,
        "ltm_alpha": -0.3,
        "evaluation_unit": "sample",
    }
    assert bm._summary_from_payload(payload) is None
    named = bm._summary_from_payload({**payload, "evaluation_design": "all"})
    assert named is not None
    assert named["evaluation_design"] == "all"


def test_a_cached_summary_without_tail_statistics_is_not_reused() -> None:
    import benchmark as bm

    payload = {
        "k": 5,
        "value": 0.6,
        "std": 0.1,
        "undefined_frac": 0.0,
        "median_value": 0.5,
        "q_alpha": -0.2,
        "ltm_alpha": -0.3,
        "evaluation_design": "all",
        "evaluation_unit": "sample",
    }
    assert bm._summary_from_payload(payload) is not None
    missing_tail = {key: value for key, value in payload.items() if key != "ltm_alpha"}
    assert bm._summary_from_payload(missing_tail) is None


def _write_cli_inputs(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    _manifest().to_csv(manifest_path, index=False)
    embeddings_path = tmp_path / "features.npy"
    np.save(embeddings_path, _features())
    return manifest_path, embeddings_path


@pytest.mark.parametrize("command", ["ri", "mari", "croma"])
def test_cli_defaults_to_all(tmp_path, monkeypatch, capsys, command: str) -> None:
    from croma import cli

    manifest_path, embeddings_path = _write_cli_inputs(tmp_path)
    argv = [
        "croma",
        command,
        "--manifest",
        str(manifest_path),
        "--embeddings",
        str(embeddings_path),
        "--confounder-column",
        "scanner_vendor",
    ]
    if command in ("ri", "mari"):
        argv += ["--k-candidates", "1"]
    else:
        argv += ["--m", "1"]
    monkeypatch.setattr("sys.argv", argv)

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["evaluation_design"] == "all"


def test_cli_rejects_dataset_wide(tmp_path, monkeypatch) -> None:
    from croma import cli

    manifest_path, embeddings_path = _write_cli_inputs(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "croma",
            "ri",
            "--manifest",
            str(manifest_path),
            "--embeddings",
            str(embeddings_path),
            "--confounder-column",
            "scanner_vendor",
            "--evaluation-design",
            "dataset_wide",
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()
