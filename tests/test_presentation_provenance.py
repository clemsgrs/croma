"""Presentation-level provenance contracts for issue #129."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "scripts" / "repro"
BENCH = ROOT / "scripts" / "bench"
FIGURES = REPRO / "figures"
for path in (REPRO, BENCH, FIGURES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _distributions as distributions  # noqa: E402
import croma_pareto_figure  # noqa: E402
import generate_model_tables as model_tables  # noqa: E402
import generate_pareto_float as pareto_float  # noqa: E402
import generate_results_table as results_table  # noqa: E402
from paper_manifest import by_prefix  # noqa: E402
from plotting import style  # noqa: E402


def _tolkach_distributions() -> distributions.Distributions:
    models = [
        distributions.Model(name, 0.2, -0.2, 0.3, False)
        for name in ["RudolfV 2", "RudolfV 2-B", "RudolfV 2-S", "UNI"]
    ]
    models.append(distributions.Model("DINOv2-B", 0.0, -0.4, 0.5, True))
    return distributions.Distributions(tuple(models))


def test_tolkach_presentation_marks_possible_institutional_overlap_conservatively() -> None:
    entry = by_prefix("Tolkach")
    dist = _tolkach_distributions()
    metadata = model_tables.load_metadata().set_index("model")
    rudolf = metadata.loc[["RudolfV 2", "RudolfV 2-B", "RudolfV 2-S"]]

    assert entry.exposure_domain == "charite"
    assert rudolf["institutional_domains"].tolist() == ["charite"] * 3
    exposed = distributions.exposed_models(entry, dist)
    assert exposed == frozenset({"RudolfV 2", "RudolfV 2-B", "RudolfV 2-S"})

    ranked = pd.DataFrame({"model": [model.name for model in dist.pathology]})
    assert results_table.benchmark_exposed(entry, ranked) == set(exposed)

    results_caption = results_table.build_caption(
        entry,
        pd.DataFrame(
            {
                "model": [model.name for model in dist.models],
                "croma": [model.median for model in dist.models],
                "k": [11] * len(dist.models),
                "confounder_display_name": ["Medical Center"] * len(dist.models),
            }
        ),
        set(exposed),
        with_ci=False,
    )
    pareto = pareto_float.build_supp_figure(entry, dist)
    for presentation in (results_caption, pareto):
        assert "possible institutional/source-domain overlap" in presentation
        assert "Exact patient or slide overlap is unknown" in presentation
        assert "does not establish leakage" in presentation
        assert "pretraining leakage" not in presentation


def test_tolkach_table_daggers_each_rudolf_variant(monkeypatch) -> None:
    models = ["RudolfV 2", "RudolfV 2-B", "RudolfV 2-S", "UNI", "DINOv2-B"]
    frame = pd.DataFrame(
        {
            "model": models,
            "bio_knn_bacc": [0.8] * 5,
            "confounder_knn_bacc": [0.6] * 5,
            "ri": [0.2] * 5,
            "mari": [0.3] * 5,
            "delta": [0.1] * 5,
            "croma": [0.2] * 5,
            "croma_frac_neg": [0.1] * 5,
            "croma_ltm_alpha": [-0.2] * 5,
            "support": [100.0] * 5,
            "k": [11] * 5,
            "confounder_display_name": ["Medical Center"] * 5,
        }
    )
    monkeypatch.setattr(results_table, "load_frame", lambda _: frame)

    table = results_table.build_table(by_prefix("Tolkach"), Path("unused.csv"))

    for model in ["RudolfV 2", "RudolfV 2-B", "RudolfV 2-S"]:
        assert f"{model}$^{{\\dagger}}$ &" in table
    assert "UNI$^{\\dagger}$" not in table


def test_tolkach_pareto_renderer_receives_the_rudolf_exposure_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    entry = by_prefix("Tolkach")
    dist = _tolkach_distributions()

    monkeypatch.setattr(croma_pareto_figure, "BENCHMARKS", (entry,))
    monkeypatch.setattr(croma_pareto_figure, "REPO", tmp_path)
    monkeypatch.setattr(croma_pareto_figure, "load", lambda *, entry: dist)
    monkeypatch.setattr(croma_pareto_figure, "_ltm_alpha_pct", lambda entry: 10)
    monkeypatch.setattr(
        croma_pareto_figure.P,
        "plot_croma_pareto",
        lambda rows, out_png, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(croma_pareto_figure.P, "_pdf_export_path", lambda path: path)

    croma_pareto_figure.main()

    assert calls == [
        {
            "exposed": frozenset({"RudolfV 2", "RudolfV 2-B", "RudolfV 2-S"}),
            "ltm_alpha_pct": 10,
        }
    ]


def test_exposure_is_derived_from_domain_tags_not_the_legacy_tcga_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metadata = pd.DataFrame(
        [
            {
                "model": "Tagged model",
                "tcga_exposed": False,
                "corpus_domains": "tcga",
                "institutional_domains": "",
            },
            {
                "model": "Legacy-only model",
                "tcga_exposed": True,
                "corpus_domains": "",
                "institutional_domains": "",
            },
        ]
    )
    metadata_path = tmp_path / "model_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    monkeypatch.setattr(distributions, "METADATA", metadata_path)

    exposed = distributions.metadata_exposed_models(
        by_prefix("TcgaFourByFour"), {"Tagged model", "Legacy-only model"}
    )

    assert exposed == frozenset({"Tagged model"})


def test_committed_tcga_domain_tags_cover_every_disclosed_tcga_model() -> None:
    metadata = model_tables.load_metadata()
    tagged = {
        str(row["model"])
        for _, row in metadata.iterrows()
        if "tcga" in str(row["corpus_domains"]).split(";")
    }
    expected = {
        "H0-mini",
        "Midnight-12k",
        "Prost40M",
        "Phikon",
        "Phikon-v2",
        "mSTAR",
        "GPFM",
        "MUSK",
        "GenBio-PathFM",
        "Mascaret",
        "Phaet",
        "MOOZY",
    }

    assert tagged == expected
    assert set(metadata.loc[metadata["tcga_exposed"], "model"]) == expected


def test_metadata_identity_loader_reads_family_tone_and_order(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        [
            {"model": "Second", "family": "beta", "family_order": 2, "plot_order": 20},
            {"model": "Not plotted", "family": "other", "family_order": 0, "plot_order": ""},
            {"model": "First", "family": "alpha", "family_order": 1, "plot_order": 10},
        ]
    )
    metadata_path = tmp_path / "model_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    families, tones, order = style._load_model_identity(metadata_path)

    assert families == {"First": "alpha", "Second": "beta"}
    assert tones == {"First": 1, "Second": 2}
    assert order == ["First", "Second"]


def test_machine_readable_families_give_new_models_stable_styles_and_insertion_order() -> None:
    metadata = model_tables.load_metadata().set_index("model")
    expected_identity = {
        "RudolfV 2": ("rudolfv2", 0, 23),
        "RudolfV 2-B": ("rudolfv2", 1, 24),
        "RudolfV 2-S": ("rudolfv2", 2, 25),
        "Mascaret": ("waiv", 0, 26),
        "Phaet": ("waiv", 1, 27),
    }
    actual_identity = {
        model: (
            metadata.loc[model, "family"],
            int(metadata.loc[model, "family_order"]),
            int(metadata.loc[model, "plot_order"]),
        )
        for model in expected_identity
    }
    assert actual_identity == expected_identity

    assert style.CANONICAL_MODEL_ORDER == [
        "Virchow2",
        "Virchow",
        "PRISM",
        "UNI2-h",
        "UNI",
        "CONCHv1.5",
        "CONCH",
        "TITAN",
        "Phikon-v2",
        "Phikon",
        "H-optimus-1",
        "H-optimus-0",
        "H0-mini",
        "Prov-GigaPath",
        "Midnight-12k",
        "Hibou-L",
        "Hibou-B",
        "Prost40M",
        "mSTAR",
        "GPFM",
        "MUSK",
        "GenBio-PathFM",
        "RudolfV 2",
        "RudolfV 2-B",
        "RudolfV 2-S",
        "Mascaret",
        "Phaet",
        "DINOv2-B",
    ]
    assert {model: style.MODEL_FAMILY_MAP[model] for model in expected_identity} == {
        "RudolfV 2": "rudolfv2",
        "RudolfV 2-B": "rudolfv2",
        "RudolfV 2-S": "rudolfv2",
        "Mascaret": "waiv",
        "Phaet": "waiv",
    }
    assert {model: style.MODEL_TONE_INDEX[model] for model in expected_identity} == {
        "RudolfV 2": 0,
        "RudolfV 2-B": 1,
        "RudolfV 2-S": 2,
        "Mascaret": 0,
        "Phaet": 1,
    }
    assert len({style.color_for_model(model) for model in ["Mascaret", "Phaet"]}) == 2
    assert (
        len({style.color_for_model(model) for model in ["RudolfV 2", "RudolfV 2-B", "RudolfV 2-S"]})
        == 3
    )
