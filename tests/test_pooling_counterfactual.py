import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
if str(STUDIES) not in sys.path:
    sys.path.insert(0, str(STUDIES))

import pooling_counterfactual as analysis


def _roster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["A", "B", "C", "DINOv2-B"],
            "model": ["A", "B", "C", "DINOv2-B"],
            "panel_order": [1, 2, 3, 4],
            "is_control": [False, False, False, True],
        }
    )


def _cohort(croma: list[float], ltm10: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["A", "B", "C", "DINOv2-B"],
            "croma": croma + [0.0],
            "croma_ltm10": ltm10 + [0.0],
        }
    )


def test_load_roster_requires_the_exact_production_tile_panel(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "model": analysis.EXPECTED_ROSTER,
            "panel": ["tile"] * 26,
            "panel_order": list(range(1, 27)),
            "published_name": [
                "",
                "",
                *([""] * 18),
                "RudolfV-2",
                "RudolfV-2-B",
                "RudolfV-2-S",
                "",
                "",
                "",
            ],
        }
    )
    path = tmp_path / "model_metadata.csv"
    metadata.to_csv(path, index=False)

    roster = analysis.load_roster(path)

    assert roster["model_id"].tolist() == list(analysis.EXPECTED_ROSTER)
    assert roster.loc[roster["model_id"] == "RudolfV 2", "model"].item() == "RudolfV-2"
    assert roster.loc[roster["model_id"] == "DINOv2-B", "is_control"].item()

    metadata.loc[0, "panel_order"] = 2
    metadata.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exact 26-row production roster"):
        analysis.load_roster(path)


def test_public_provenance_freezes_cohorts_operating_points_and_hashes(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    files = {}
    for name in (*analysis.PUBLIC_COHORTS, "cross_benchmark"):
        path = results_root / f"{name}.csv"
        path.write_bytes(f"{name}\n".encode())
        files[f"results/{name}.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "roster": 26,
        "cohorts": {
            cohort: {
                "benchmark": analysis.COHORT_BENCHMARKS[cohort],
                "panel": "tile",
                "k": analysis.FIXED_K[cohort],
                "n_models": 26,
            }
            for cohort in analysis.PUBLIC_COHORTS
        },
        "files": files,
    }
    provenance_path = results_root / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(payload), encoding="utf-8")

    validated, digest = analysis.validate_public_provenance(results_root)

    assert validated == payload
    assert digest == hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    payload["cohorts"]["tcga-4x4"]["k"] = 91
    provenance_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen public cohort contract"):
        analysis.validate_public_provenance(results_root)


def test_rank_scenario_uses_roster_order_for_ties_and_keeps_control_unranked() -> None:
    cohorts = {
        "camelyon": _cohort([3.0, 3.0, 1.0], [1.0, 2.0, 3.0]),
        "tcga-4x4": _cohort([2.0, 3.0, 1.0], [3.0, 2.0, 1.0]),
        "tolkach-esca": _cohort([2.0, 1.0, 3.0], [2.0, 3.0, 1.0]),
    }

    result = analysis.rank_scenario(cohorts, _roster(), scenario="canonical", view="public")
    ranked = result.rankings.set_index("model_id")

    assert ranked.loc["A", ["croma_rank_camelyon", "croma_rank_tcga-4x4"]].tolist() == [1, 2]
    assert ranked.loc["B", "croma_rank_camelyon"] == 2
    assert ranked.loc[
        "A", ["mean_croma_rank", "mean_ltm10_rank", "combined_mean_rank"]
    ].tolist() == [1.667, 2.0, 1.8335]
    assert ranked.loc[
        "B", ["mean_croma_rank", "mean_ltm10_rank", "combined_mean_rank"]
    ].tolist() == [2.0, 1.667, 1.8335]
    assert ranked.loc["A", "headline_order"] == 1
    assert ranked.loc["B", "headline_order"] == 2
    assert ranked.loc["C", "headline_order"] == 3
    assert ranked.loc[["A", "B", "C"], "on_frontier"].tolist() == [True, True, False]
    assert pd.isna(ranked.loc["DINOv2-B", "combined_mean_rank"])
    assert not ranked.loc["DINOv2-B", "ranked"]
    assert result.ties.to_dict("records") == [
        {
            "scenario": "canonical",
            "view": "public",
            "cohort": "camelyon",
            "metric": "croma",
            "value": 3.0,
            "models": "A|B",
            "roster_order": "A|B",
            "resolved_order": "A|B",
        },
        {
            "scenario": "canonical",
            "view": "public",
            "cohort": "cross-cohort",
            "metric": "combined_mean_rank",
            "value": 1.8335,
            "models": "A|B",
            "roster_order": "A|B",
            "resolved_order": "A|B",
        },
    ]


def test_rank_scenario_keeps_identical_points_on_the_min_min_frontier() -> None:
    cohorts = {name: _cohort([2.0, 2.0, 1.0], [2.0, 2.0, 1.0]) for name in analysis.PUBLIC_COHORTS}

    result = analysis.rank_scenario(cohorts, _roster(), scenario="canonical", view="public")

    assert result.rankings.set_index("model_id").loc[["A", "B"], "on_frontier"].tolist() == [
        True,
        False,
    ]
    # method='first' makes the coordinates different even when all raw inputs tie.
    assert len(result.ties) == 6


def test_rank_scenario_reports_mean_axis_ties_and_resolves_headline_by_roster() -> None:
    roster = _roster()
    roster.loc[:2, "model"] = ["Z", "Y", "X"]
    cohorts = {
        "camelyon": _cohort([3.0, 2.0, 1.0], [3.0, 2.0, 1.0]),
        "tcga-4x4": _cohort([2.0, 1.0, 3.0], [2.0, 1.0, 3.0]),
        "tolkach-esca": _cohort([1.0, 3.0, 2.0], [1.0, 3.0, 2.0]),
    }

    result = analysis.rank_scenario(cohorts, roster, scenario="canonical", view="public")

    cross_cohort = result.ties[result.ties["cohort"] == "cross-cohort"]
    assert cross_cohort["metric"].tolist() == [
        "mean_croma_rank",
        "mean_ltm10_rank",
        "combined_mean_rank",
    ]
    assert cross_cohort["models"].tolist() == ["A|B|C"] * 3
    assert result.rankings.sort_values("headline_order")["model_id"].head(3).tolist() == [
        "A",
        "B",
        "C",
    ]


def test_rank_change_classification_covers_public_and_internal_views() -> None:
    canonical = {
        "camelyon": _cohort([3.0, 2.0, 1.0], [3.0, 2.0, 1.0]),
        "tcga-4x4": _cohort([3.0, 2.0, 1.0], [3.0, 2.0, 1.0]),
        "tolkach-esca": _cohort([3.0, 2.0, 1.0], [3.0, 2.0, 1.0]),
    }
    alternative = {
        cohort: frame.assign(
            croma=frame["croma"].replace({3.0: 2.0, 2.0: 3.0}),
        )
        for cohort, frame in canonical.items()
    }
    internal_roster = _roster().iloc[:2].copy()
    internal = {
        scenario: {cohort: frame.iloc[:2].copy() for cohort, frame in cohorts.items()}
        for scenario, cohorts in (("canonical", canonical), ("alternative", alternative))
    }
    results = [
        analysis.rank_scenario(canonical, _roster(), scenario="canonical", view="public"),
        analysis.rank_scenario(alternative, _roster(), scenario="alternative", view="public"),
        analysis.rank_scenario(
            internal["canonical"], internal_roster, scenario="canonical", view="internal"
        ),
        analysis.rank_scenario(
            internal["alternative"], internal_roster, scenario="alternative", view="internal"
        ),
    ]

    changes = analysis.classify_rank_changes(
        pd.concat([result.rankings for result in results], ignore_index=True)
    )

    assert changes.groupby("view").size().to_dict() == {"internal": 2, "public": 3}
    swapped = changes.set_index(["view", "model_id"])
    assert swapped.loc[("public", "A"), "delta_headline_order"] == 1
    assert swapped.loc[("internal", "B"), "delta_headline_order"] == -1


def test_classify_measurements_distinguishes_movement_support_and_zero_crossings() -> None:
    comparisons = pd.DataFrame(
        [
            {
                "benchmark": "pathorob-camelyon",
                "model": "A",
                "fixed_k": 11,
                "canonical_croma": -0.1,
                "alternative_croma": 0.1,
                "canonical_croma_ltm10": 0.2,
                "alternative_croma_ltm10": 0.1,
                "croma_delta_ci_lo": 0.01,
                "croma_delta_ci_hi": 0.3,
                "canonical_ri": 0.4,
                "alternative_ri": 0.5,
                "canonical_mari": 0.3,
                "alternative_mari": 0.2,
                "canonical_support": 0.7,
                "alternative_support": 0.6,
                "canonical_ss_dominated_undefined_frac": 0.2,
                "alternative_ss_dominated_undefined_frac": 0.3,
                "canonical_oo_dominated_undefined_frac": 0.1,
                "alternative_oo_dominated_undefined_frac": 0.1,
                "canonical_mixed_undefined_frac": 0.0,
                "alternative_mixed_undefined_frac": 0.0,
            },
            {
                "benchmark": "pathorob-tcga-4x4",
                "model": "A",
                "fixed_k": 71,
                "canonical_croma": 0.2,
                "alternative_croma": 0.3,
                "canonical_croma_ltm10": -0.2,
                "alternative_croma_ltm10": -0.2,
                "croma_delta_ci_lo": 0.0,
                "croma_delta_ci_hi": 0.2,
                "canonical_ri": 0.4,
                "alternative_ri": 0.4,
                "canonical_mari": 0.3,
                "alternative_mari": 0.3,
                "canonical_support": 0.7,
                "alternative_support": 0.7,
                "canonical_ss_dominated_undefined_frac": 0.2,
                "alternative_ss_dominated_undefined_frac": 0.2,
                "canonical_oo_dominated_undefined_frac": 0.1,
                "alternative_oo_dominated_undefined_frac": 0.1,
                "canonical_mixed_undefined_frac": 0.0,
                "alternative_mixed_undefined_frac": 0.0,
            },
        ]
    )

    classified = analysis.classify_measurements(comparisons).set_index("cohort")

    assert classified.loc["camelyon", "croma_movement"] == "increase"
    assert classified.loc["camelyon", "croma_supported_shift"] == "increase"
    assert classified.loc["camelyon", "croma_sign_crossing"] == "negative-to-positive"
    assert classified.loc["camelyon", "ltm10_movement"] == "decrease"
    assert classified.loc["camelyon", "ltm10_sign_crossing"] == "none"
    assert classified.loc["tcga-4x4", "croma_supported_shift"] == "unsupported"
    assert classified.loc["tcga-4x4", "ltm10_movement"] == "unchanged"

    movements = analysis.build_metric_movements(classified.reset_index())
    camelyon = movements[movements["cohort"] == "camelyon"].set_index("metric")
    assert camelyon.index.tolist() == ["ri", "mari", "croma", "croma_ltm10"]
    assert camelyon.loc[
        "ri", ["canonical_value", "alternative_value", "delta"]
    ].tolist() == pytest.approx([0.4, 0.5, 0.1])
    assert camelyon.loc["mari", "canonical_positive_support"] == 0.7
    assert pd.isna(camelyon.loc["croma", "canonical_positive_support"])
    assert pd.isna(camelyon.loc["croma_ltm10", "canonical_ss_dominated_undefined_frac"])

    wrong_k = comparisons.copy()
    wrong_k.loc[0, "fixed_k"] = 13
    with pytest.raises(ValueError, match="fixed k=11"):
        analysis.classify_measurements(wrong_k)


def test_family_guards_separate_reversals_from_tie_transitions() -> None:
    measurements = pd.DataFrame(
        [
            ["canonical", "camelyon", "parent", 0.4, 0.2],
            ["canonical", "camelyon", "child", 0.3, 0.2],
            ["alternative", "camelyon", "parent", 0.2, 0.2],
            ["alternative", "camelyon", "child", 0.3, 0.2],
            ["canonical", "tcga-4x4", "parent", 0.4, 0.1],
            ["canonical", "tcga-4x4", "child", 0.3, 0.2],
            ["alternative", "tcga-4x4", "parent", 0.4, 0.2],
            ["alternative", "tcga-4x4", "child", 0.3, 0.2],
        ],
        columns=["scenario", "cohort", "model_id", "croma", "croma_ltm10"],
    )
    rankings = pd.DataFrame(
        [
            ["canonical", "parent", 1.5],
            ["canonical", "child", 2.0],
            ["alternative", "parent", 2.0],
            ["alternative", "child", 1.5],
        ],
        columns=["scenario", "model_id", "combined_mean_rank"],
    )

    guards = analysis.classify_family_guards(
        measurements,
        rankings,
        families=(("family", "parent", "child"),),
    )

    assert guards[["scope", "metric", "classification"]].to_dict("records") == [
        {"scope": "camelyon", "metric": "croma", "classification": "reversal"},
        {"scope": "camelyon", "metric": "ltm10", "classification": "unchanged-tie"},
        {"scope": "tcga-4x4", "metric": "croma", "classification": "unchanged"},
        {"scope": "tcga-4x4", "metric": "ltm10", "classification": "tie-transition"},
        {"scope": "cross-cohort", "metric": "combined-rank", "classification": "reversal"},
    ]


@pytest.mark.parametrize(
    ("evidence", "expected", "reason"),
    [
        (
            analysis.RecommendationEvidence(
                supported_croma_deltas=(0.1,),
                public_croma_deltas=(0.1, 0.0, 0.2),
                public_ltm10_deltas=(0.0, 0.1, 0.2),
                sign_crossings=0,
                rank_worsened=False,
                left_frontier=False,
                family_reversals=0,
            ),
            "change",
            "supported-croma-benefit;no-guard-regressions",
        ),
        (
            analysis.RecommendationEvidence(
                supported_croma_deltas=(0.1,),
                public_croma_deltas=(0.1, 0.0, 0.2),
                public_ltm10_deltas=(0.0, -0.01, 0.2),
                sign_crossings=0,
                rank_worsened=False,
                left_frontier=False,
                family_reversals=0,
            ),
            "retain",
            "public-ltm10-decrease",
        ),
        (
            analysis.RecommendationEvidence(
                supported_croma_deltas=(),
                public_croma_deltas=(0.1, 0.1, 0.1),
                public_ltm10_deltas=(0.1, 0.1, 0.1),
                sign_crossings=0,
                rank_worsened=False,
                left_frontier=False,
                family_reversals=0,
            ),
            "retain",
            "no-supported-croma-benefit",
        ),
    ],
)
def test_conservative_recommendation_requires_supported_benefit_and_no_regression(
    evidence: analysis.RecommendationEvidence,
    expected: str,
    reason: str,
) -> None:
    recommendation = analysis.recommendation_from_evidence(evidence)
    assert recommendation == {"recommendation": expected, "reason_codes": reason}


def test_render_impact_summary_leads_with_one_row_per_encoder() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "A",
                "alternative": "cls-only",
                "supported_croma_increases": 2,
                "supported_croma_decreases": 0,
                "public_croma_increases": 2,
                "public_croma_decreases": 1,
                "public_ltm10_increases": 0,
                "public_ltm10_decreases": 3,
                "camelyon_croma_delta": 0.01234,
                "camelyon_croma_supported": True,
                "tcga_2x2_croma_delta": -0.001,
                "tcga_2x2_croma_supported": False,
                "tcga_4x4_croma_delta": 0.02,
                "tcga_4x4_croma_supported": True,
                "tolkach_esca_croma_delta": -0.1,
                "tolkach_esca_croma_supported": True,
                "sign_crossings": 0,
                "rank_change": "2->1",
                "family_reversals": 1,
                "recommendation": "retain",
                "reason_codes": "public-croma-decrease;public-ltm10-decrease;family-reversal",
            }
        ]
    )

    report = analysis.render_report(
        summary=summary,
        public_orders={"canonical": ["B", "A"], "alternative": ["A", "B"]},
        public_frontiers={"canonical": ["B"], "alternative": ["B"]},
        internal_orders={"canonical": ["A", "B"], "alternative": ["A", "B"]},
        internal_frontiers={"canonical": ["A"], "alternative": ["A"]},
    )

    assert report.startswith("# Pooling counterfactual impact\n\n## Impact summary\n")
    assert (
        "| A | cls-only | +0.0123* / -0.0010 / +0.0200* / -0.1000* | +2 / -0 | +0 / -3 | 0 | 2->1 | 1 | retain |"
        in report
    )
    assert "Canonical order: B, A" in report
    assert "Alternative order: A, B" in report
    assert "Internal canonical order: A, B" in report


def test_publish_analysis_bundle_check_is_zero_write_and_preserves_existing_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pooling-sensitivity"
    existing = root / "results" / "comparisons.csv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"sealed #152\n")
    files = {Path("analysis/impact-summary.csv"): b"model,recommendation\nA,retain\n"}

    assert analysis.publish_analysis_bundle(root, files) == "written"
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (existing, root / "analysis/impact-summary.csv")
    }
    assert analysis.publish_analysis_bundle(root, files, check=True) == "checked"
    after = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in before
    }
    assert after == before
    assert existing.read_bytes() == b"sealed #152\n"
    with pytest.raises(ValueError, match="stay below analysis"):
        analysis.publish_analysis_bundle(
            root,
            {Path("analysis/../results/comparisons.csv"): b"corrupt\n"},
            force=True,
        )
    assert existing.read_bytes() == b"sealed #152\n"

    symlink_root = tmp_path / "symlinked-study"
    protected = symlink_root / "results/rankings.csv"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"sealed rankings\n")
    (symlink_root / "analysis").symlink_to("results", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked analysis path"):
        analysis.publish_analysis_bundle(
            symlink_root,
            {Path("analysis/rankings.csv"): b"corrupt\n"},
            force=True,
        )
    assert protected.read_bytes() == b"sealed rankings\n"


def test_build_analysis_bundle_orchestrates_deterministic_synthetic_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_path = tmp_path / "scripts/bench/model_metadata.csv"
    results_root = tmp_path / "results"
    study_root = tmp_path / "output/studies/pooling-sensitivity"
    metadata_path.parent.mkdir(parents=True)
    results_root.mkdir(parents=True)
    (study_root / "results").mkdir(parents=True)
    metadata_path.write_bytes(b"synthetic metadata\n")
    for cohort in analysis.PUBLIC_COHORTS:
        (results_root / f"{cohort}.csv").write_bytes(b"synthetic public result\n")
    cross_benchmark_path = results_root / "cross_benchmark.csv"
    cross_benchmark_path.write_bytes(b"synthetic aggregate\n")

    published = {
        "RudolfV 2": "RudolfV-2",
        "RudolfV 2-B": "RudolfV-2-B",
        "RudolfV 2-S": "RudolfV-2-S",
    }
    roster = pd.DataFrame(
        {
            "model_id": analysis.EXPECTED_ROSTER,
            "model": [published.get(model, model) for model in analysis.EXPECTED_ROSTER],
            "panel_order": range(1, 27),
            "is_control": [model == "DINOv2-B" for model in analysis.EXPECTED_ROSTER],
        }
    )
    canonical = {}
    for cohort_index, cohort in enumerate(analysis.PUBLIC_COHORTS):
        canonical[cohort] = pd.DataFrame(
            {
                "model_id": analysis.EXPECTED_ROSTER,
                "croma": [1.0 - index / 100 - cohort_index / 1000 for index in range(26)],
                "croma_ltm10": [0.5 - index / 100 - cohort_index / 1000 for index in range(26)],
            }
        )

    comparison_rows = []
    for cohort, benchmark in analysis.COHORT_BENCHMARKS.items():
        public_cohort = "tcga-4x4" if cohort == "tcga-2x2" else cohort
        values = canonical[public_cohort].set_index("model_id")
        for model in analysis.STUDY_MODELS:
            row = {
                "benchmark": benchmark,
                "model": model,
                "fixed_k": analysis.FIXED_K[cohort],
                "canonical_representation": "canonical",
                "alternative_representation": analysis.STUDY_ALTERNATIVES[model],
                "croma_delta_ci_lo": -0.1,
                "croma_delta_ci_hi": 0.1,
            }
            for metric in analysis.MOVEMENT_METRICS:
                if metric == "croma":
                    before = float(values.loc[model, "croma"])
                    after = before - 0.001
                elif metric == "croma_ltm10":
                    before = float(values.loc[model, "croma_ltm10"])
                    after = before - 0.002
                elif metric == "support":
                    before = after = 0.8
                elif metric.endswith("undefined_frac"):
                    before = after = 0.0
                else:
                    before = after = 0.7
                row[f"canonical_{metric}"] = before
                row[f"alternative_{metric}"] = after
            comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(study_root / "results/comparisons.csv", index=False)

    monkeypatch.setattr(analysis, "load_roster", lambda _path: roster.copy())
    monkeypatch.setattr(
        analysis,
        "validate_public_provenance",
        lambda _root: ({"roster": 26, "cohorts": {}}, "public-sha"),
    )
    monkeypatch.setattr(
        analysis,
        "_load_public_cohorts",
        lambda _root, _roster: {cohort: frame.copy() for cohort, frame in canonical.items()},
    )
    validated = {}

    def validate_canonical(rankings: pd.DataFrame, path: Path) -> None:
        validated["models"] = len(
            rankings[(rankings["scenario"] == "canonical") & (rankings["view"] == "public")]
        )
        validated["path"] = path

    monkeypatch.setattr(analysis, "_validate_canonical_public_ranking", validate_canonical)
    monkeypatch.setattr(
        analysis,
        "_verify_source_provenance",
        lambda _root: ({"schema_version": 2, "study": "pooling-sensitivity"}, "sealed-sha"),
    )

    bundle = analysis.build_analysis_bundle(
        metadata_path=metadata_path,
        results_root=results_root,
        study_root=study_root,
    )

    assert set(bundle) == {
        Path("analysis/impact-summary.csv"),
        Path("analysis/measurements.csv"),
        Path("analysis/metric-movements.csv"),
        Path("analysis/rankings.csv"),
        Path("analysis/ties.csv"),
        Path("analysis/rank-changes.csv"),
        Path("analysis/family-guards.csv"),
        Path("analysis/recommendations.csv"),
        Path("analysis/summary.json"),
        Path("analysis/report.md"),
        Path("analysis/run-provenance.json"),
    }
    summary = pd.read_csv(pd.io.common.BytesIO(bundle[Path("analysis/impact-summary.csv")]))
    assert summary["model"].tolist() == [
        "Mascaret",
        "Phaet",
        "RudolfV-2",
        "RudolfV-2-B",
        "RudolfV-2-S",
    ]
    assert summary["recommendation"].tolist() == ["retain"] * 5
    assert validated == {"models": 26, "path": cross_benchmark_path}
    provenance = json.loads(bundle[Path("analysis/run-provenance.json")])
    assert provenance["source_study_provenance"]["sha256"] == "sealed-sha"
    assert provenance["public_results_provenance"]["sha256"] == "public-sha"


def test_real_analysis_bundle_has_the_frozen_counterfactual_conclusions() -> None:
    study_root = Path(
        os.environ.get(
            "CROMA_POOLING_STUDY_ROOT",
            ROOT / "output" / "studies" / "pooling-sensitivity",
        )
    )
    if not (study_root / "results/comparisons.csv").exists():
        pytest.skip("local #152 analytical inputs are unavailable")

    bundle = analysis.build_analysis_bundle(
        metadata_path=ROOT / "scripts/bench/model_metadata.csv",
        results_root=ROOT / "results",
        study_root=study_root,
    )
    summary = pd.read_csv(pd.io.common.BytesIO(bundle[Path("analysis/impact-summary.csv")]))
    rankings = pd.read_csv(pd.io.common.BytesIO(bundle[Path("analysis/rankings.csv")]))
    guards = pd.read_csv(pd.io.common.BytesIO(bundle[Path("analysis/family-guards.csv")]))
    provenance = json.loads(bundle[Path("analysis/run-provenance.json")])

    assert summary["model"].tolist() == [
        "Mascaret",
        "Phaet",
        "RudolfV-2",
        "RudolfV-2-B",
        "RudolfV-2-S",
    ]
    assert summary["recommendation"].tolist() == ["retain"] * 5
    public = rankings[(rankings["view"] == "public") & rankings["ranked"]]
    canonical = public[public["scenario"] == "canonical"].sort_values("headline_order")
    alternative = public[public["scenario"] == "alternative"].sort_values("headline_order")
    assert canonical["model"].head(4).tolist() == [
        "Mascaret",
        "RudolfV-2-S",
        "RudolfV-2",
        "RudolfV-2-B",
    ]
    assert alternative["model"].head(4).tolist() == [
        "Mascaret",
        "RudolfV-2",
        "RudolfV-2-S",
        "RudolfV-2-B",
    ]
    assert set(canonical.loc[canonical["on_frontier"], "model"]) == {"Mascaret"}
    assert set(alternative.loc[alternative["on_frontier"], "model"]) == {"Mascaret"}
    assert len(guards[guards["classification"] == "reversal"]) == 3
    assert not (guards["classification"] == "tie-transition").any()
    internal = rankings[(rankings["view"] == "internal") & rankings["ranked"]]
    assert internal[internal["scenario"] == "canonical"].sort_values("headline_order")[
        "model"
    ].tolist() == ["Mascaret", "RudolfV-2-S", "RudolfV-2", "RudolfV-2-B", "Phaet"]
    assert internal[internal["scenario"] == "alternative"].sort_values("headline_order")[
        "model"
    ].tolist() == ["Mascaret", "RudolfV-2-S", "RudolfV-2", "RudolfV-2-B", "Phaet"]
    assert provenance["schema_version"] == 1
    assert (
        provenance["source_study_provenance"]["sha256"]
        == hashlib.sha256((study_root / "run-provenance.json").read_bytes()).hexdigest()
    )
