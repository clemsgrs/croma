import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "tools"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_results as ar


def _m(ratio: float) -> float:
    """Map a legacy distance ratio to the signed CRoMa margin in (-1, 1).

    CRoMa is now ``(d_OS - d_SO) / (d_OS + d_SO) = (r - 1) / (r + 1)``. Test
    fixtures keep readable ratio literals; ``_per_sample_row`` converts them so
    the synthetic per-sample scores are margin-valued (threshold 0).
    """
    return (ratio - 1.0) / (ratio + 1.0)


def _metrics_rows(
    models: list[str],
    *,
    dataset: str = "camelyon",
    evaluation_design: str = "dataset_wide",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, model in enumerate(models):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "evaluation_design": evaluation_design,
                "evaluation_unit": (
                    "sample" if evaluation_design == "dataset_wide" else "occurrence"
                ),
                "ri": 0.86 + 0.002 * idx,
                "mari": 0.85 + 0.002 * idx,
                "croma": 1.18 + 0.01 * idx,
                "croma_q_alpha": 0.82 - 0.01 * idx,
                "croma_ltm_alpha": 0.71 - 0.005 * idx,
            }
        )
    return pd.DataFrame(rows)


def _per_sample_row(
    *,
    model: str,
    sample_id: str,
    label: str,
    confounder: str,
    croma_m1: float,
    evaluation_design: str = "dataset_wide",
    subset: str = "dataset",
    dataset: str = "camelyon",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": model,
        "evaluation_design": evaluation_design,
        "evaluation_unit": ("sample" if evaluation_design == "dataset_wide" else "occurrence"),
        "subset": subset,
        "sample_id": sample_id,
        "slide_id": f"slide-{sample_id}",
        "label": label,
        "confounder": confounder,
        "croma_alpha": 0.25,
        "croma_search": "start=200;growth=2;alpha=0.25",
        # Emit the production headline per-sample column (m=CROMA_HEADLINE_M); the
        # `croma_m1` kwarg is just the test's per-sample value, not the column name.
        ar._SUBGROUP_SCORE_COLUMN: float(_m(croma_m1)),
    }


def _binary_camelyon_like_per_sample_df() -> pd.DataFrame:
    rows = [
        _per_sample_row(
            model="M_fragile",
            sample_id="a_t_r_1",
            label="tumor",
            confounder="RUMC",
            croma_m1=0.40,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="b_t_r_2",
            label="tumor",
            confounder="RUMC",
            croma_m1=0.50,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="c_t_u_3",
            label="tumor",
            confounder="UMCU",
            croma_m1=0.90,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="d_t_u_4",
            label="tumor",
            confounder="UMCU",
            croma_m1=1.60,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="z_n_r_5",
            label="normal",
            confounder="RUMC",
            croma_m1=0.50,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="f_n_r_6",
            label="normal",
            confounder="RUMC",
            croma_m1=1.20,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="g_n_u_7",
            label="normal",
            confounder="UMCU",
            croma_m1=1.30,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="h_n_u_8",
            label="normal",
            confounder="UMCU",
            croma_m1=1.40,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="a_t_r_1",
            label="tumor",
            confounder="RUMC",
            croma_m1=0.70,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="b_t_r_2",
            label="tumor",
            confounder="RUMC",
            croma_m1=1.05,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="c_t_u_3",
            label="tumor",
            confounder="UMCU",
            croma_m1=1.10,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="d_t_u_4",
            label="tumor",
            confounder="UMCU",
            croma_m1=1.20,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="z_n_r_5",
            label="normal",
            confounder="RUMC",
            croma_m1=0.75,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="f_n_r_6",
            label="normal",
            confounder="RUMC",
            croma_m1=1.00,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="g_n_u_7",
            label="normal",
            confounder="UMCU",
            croma_m1=1.15,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="h_n_u_8",
            label="normal",
            confounder="UMCU",
            croma_m1=1.30,
        ),
    ]
    return pd.DataFrame(rows)


def _borderline_fragility_per_sample_df() -> pd.DataFrame:
    rows = [
        _per_sample_row(
            model="M_borderline",
            sample_id="a1",
            label="tumor",
            confounder="RUMC",
            croma_m1=0.40,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="a2",
            label="tumor",
            confounder="RUMC",
            croma_m1=0.50,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="a3",
            label="tumor",
            confounder="RUMC",
            croma_m1=1.10,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="a4",
            label="tumor",
            confounder="RUMC",
            croma_m1=1.20,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="b1",
            label="normal",
            confounder="RUMC",
            croma_m1=0.60,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="b2",
            label="normal",
            confounder="RUMC",
            croma_m1=1.30,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="c1",
            label="tumor",
            confounder="UMCU",
            croma_m1=1.40,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="c2",
            label="tumor",
            confounder="UMCU",
            croma_m1=1.50,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="d1",
            label="normal",
            confounder="UMCU",
            croma_m1=1.60,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="d2",
            label="normal",
            confounder="UMCU",
            croma_m1=1.70,
        ),
    ]
    return pd.DataFrame(rows)


def _sharp_tail_only_per_sample_df() -> pd.DataFrame:
    rows = [
        _per_sample_row(
            model="M_sharp",
            sample_id="a1",
            label="tumor",
            confounder="RUMC",
            croma_m1=0.20,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="a2",
            label="tumor",
            confounder="RUMC",
            croma_m1=0.25,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="a3",
            label="tumor",
            confounder="RUMC",
            croma_m1=1.60,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="a4",
            label="tumor",
            confounder="RUMC",
            croma_m1=1.70,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="b1",
            label="normal",
            confounder="RUMC",
            croma_m1=0.40,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="b2",
            label="normal",
            confounder="RUMC",
            croma_m1=0.80,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="b3",
            label="normal",
            confounder="RUMC",
            croma_m1=0.85,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="b4",
            label="normal",
            confounder="RUMC",
            croma_m1=0.90,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="c1",
            label="tumor",
            confounder="UMCU",
            croma_m1=0.95,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="c2",
            label="tumor",
            confounder="UMCU",
            croma_m1=1.05,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="d1",
            label="normal",
            confounder="UMCU",
            croma_m1=1.10,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="d2",
            label="normal",
            confounder="UMCU",
            croma_m1=1.20,
        ),
    ]
    return pd.DataFrame(rows)


def _tier2_supported_per_sample_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    hidden_pocket_values = [0.70, 0.85, 0.95, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]
    internal_spread_values = [
        1.02,
        1.08,
        1.12,
        1.28,
        1.33,
        1.38,
        1.43,
        1.48,
        1.53,
        1.58,
    ]
    aggravated_values = [0.45, 0.55, 0.65, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    neutral_values = [1.05, 1.10, 1.15, 1.18, 1.20, 1.22, 1.25, 1.28, 1.30, 1.35]

    for idx, value in enumerate(hidden_pocket_values, start=1):
        rows.append(
            _per_sample_row(
                model="M_tier2",
                sample_id=f"hp_{idx}",
                label="tumor",
                confounder="RUMC",
                croma_m1=value,
            )
        )
    for idx, value in enumerate(internal_spread_values, start=1):
        rows.append(
            _per_sample_row(
                model="M_tier2",
                sample_id=f"is_{idx}",
                label="normal",
                confounder="RUMC",
                croma_m1=value,
            )
        )
    for idx, value in enumerate(aggravated_values, start=1):
        rows.append(
            _per_sample_row(
                model="M_tier2",
                sample_id=f"aw_{idx}",
                label="tumor",
                confounder="UMCU",
                croma_m1=value,
            )
        )
    for idx, value in enumerate(neutral_values, start=1):
        rows.append(
            _per_sample_row(
                model="M_tier2",
                sample_id=f"ne_{idx}",
                label="normal",
                confounder="UMCU",
                croma_m1=value,
            )
        )

    return pd.DataFrame(rows)


def test_aggregate_by_model_separates_evaluation_designs() -> None:
    df = pd.DataFrame(
        {
            "model": ["M1", "M1"],
            "evaluation_design": ["dataset_wide", "paired_2x2"],
            "evaluation_unit": ["sample", "occurrence"],
            "ri": [0.8, 0.4],
            "mari": [0.7, 0.3],
        }
    )

    grouped = ar._aggregate_by_model(df)

    assert set(grouped["model"]) == {
        "M1 [dataset_wide;sample]",
        "M1 [paired_2x2;occurrence]",
    }
    assert grouped.sort_values("ri")["ri"].tolist() == [0.4, 0.8]


def test_build_croma_subgroup_analysis_highlights_tumor_fragility() -> None:
    subgroup_df, context_df = ar._build_croma_subgroup_analysis(
        _binary_camelyon_like_per_sample_df()
    )

    fragile_context = context_df[
        (context_df["model"] == "M_fragile") & (context_df["context_id"] == "dataset")
    ].iloc[0]
    assert fragile_context["skip_reason"] == ""
    assert int(fragile_context["tail_size"]) == 2

    tumor_row = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["context_id"] == "dataset")
        & (subgroup_df["scope"] == "label")
        & (subgroup_df["label"] == "tumor")
    ].iloc[0]
    normal_row = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["context_id"] == "dataset")
        & (subgroup_df["scope"] == "label")
        & (subgroup_df["label"] == "normal")
    ].iloc[0]

    tumor_m = np.array([_m(v) for v in (0.40, 0.50, 0.90, 1.60)])
    normal_m = np.array([_m(v) for v in (0.50, 1.20, 1.30, 1.40)])
    assert float(tumor_row["mean_croma"]) == pytest.approx(float(tumor_m.mean()))
    assert float(tumor_row["rest_mean_croma"]) == pytest.approx(float(normal_m.mean()))
    assert float(tumor_row["mean_croma_delta_vs_rest"]) == pytest.approx(
        float(tumor_m.mean() - normal_m.mean())
    )
    assert float(tumor_row["tail_prevalence"]) == pytest.approx(0.50)
    assert float(tumor_row["tail_share"]) == pytest.approx(1.0)
    assert float(tumor_row["group_frac"]) == pytest.approx(0.50)
    assert float(tumor_row["rest_croma_neg_frac"]) == pytest.approx(0.25)
    assert float(tumor_row["croma_neg_frac_delta_vs_rest"]) == pytest.approx(0.50)
    assert float(normal_row["tail_prevalence"]) == pytest.approx(0.0)

    markdown = ar._render_croma_subgroup_markdown(subgroup_df, context_df)
    assert "tumor" in markdown
    assert "RUMC" in markdown
    assert "tail_enriched" in markdown
    assert "2.0x" in markdown
    assert "no_rest_tail" in markdown


def test_subgroup_analysis_reports_primary_and_supporting_scopes() -> None:
    subgroup_df, _context_df = ar._build_croma_subgroup_analysis(
        _binary_camelyon_like_per_sample_df()
    )

    assert set(subgroup_df["scope"]) == {"stratum", "label", "confounder"}

    fragile_rows = subgroup_df[subgroup_df["model"] == "M_fragile"]
    stratum_top = fragile_rows[fragile_rows["scope"] == "stratum"].sort_values(
        ["tail_prevalence", "mean_croma"], ascending=[False, True]
    )
    assert tuple(stratum_top.iloc[0][["label", "confounder"]]) == ("tumor", "RUMC")


def test_exact_tail_membership_uses_alpha_and_sample_id_tiebreak() -> None:
    subgroup_df, context_df = ar._build_croma_subgroup_analysis(
        _binary_camelyon_like_per_sample_df()
    )

    fragile_context = context_df[
        (context_df["model"] == "M_fragile") & (context_df["context_id"] == "dataset")
    ].iloc[0]
    assert int(fragile_context["tail_size"]) == 2

    normal_rumc = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["context_id"] == "dataset")
        & (subgroup_df["scope"] == "stratum")
        & (subgroup_df["label"] == "normal")
        & (subgroup_df["confounder"] == "RUMC")
    ].iloc[0]
    tumor_rumc = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["context_id"] == "dataset")
        & (subgroup_df["scope"] == "stratum")
        & (subgroup_df["label"] == "tumor")
        & (subgroup_df["confounder"] == "RUMC")
    ].iloc[0]

    assert float(tumor_rumc["tail_count"]) == pytest.approx(2.0)
    assert float(normal_rumc["tail_count"]) == pytest.approx(0.0)


def test_croma_neg_frac_and_tail_prevalence_stay_distinct() -> None:
    subgroup_df, _context_df = ar._build_croma_subgroup_analysis(
        _binary_camelyon_like_per_sample_df()
    )

    tumor_row = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["scope"] == "label")
        & (subgroup_df["label"] == "tumor")
    ].iloc[0]

    assert float(tumor_row["croma_neg_frac"]) == pytest.approx(0.75)
    assert float(tumor_row["tail_prevalence"]) == pytest.approx(0.50)


def test_subgroup_rows_include_tier_metrics_and_statuses() -> None:
    subgroup_df, context_df = ar._build_croma_subgroup_analysis(
        _binary_camelyon_like_per_sample_df()
    )

    assert len(context_df) == 2
    assert {
        "subgroup_name",
        "rest_median_croma",
        "median_croma_delta_vs_rest",
        "tier1_status",
        "croma_neg_count",
        "subgroup_q_alpha",
        "subgroup_ltm_alpha",
        "internal_tail_drop",
        "tier2_status",
        "tail_severity_label",
        "tier3_status",
    }.issubset(subgroup_df.columns)

    fragile_tumor = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["scope"] == "label")
        & (subgroup_df["label"] == "tumor")
    ].iloc[0]
    fragile_tumor_m = np.array([_m(v) for v in (0.40, 0.50, 0.90, 1.60)])
    fragile_normal_m = np.array([_m(v) for v in (0.50, 1.20, 1.30, 1.40)])
    assert float(fragile_tumor["rest_median_croma"]) == pytest.approx(
        float(np.median(fragile_normal_m))
    )
    assert float(fragile_tumor["median_croma_delta_vs_rest"]) == pytest.approx(
        float(np.median(fragile_tumor_m) - np.median(fragile_normal_m))
    )
    assert fragile_tumor["tier1_status"] == "broad_weakness"
    assert fragile_tumor["tail_severity_label"] == "no_rest_tail"
    assert fragile_tumor["tier3_status"] == "tail_enriched"

    borderline_umcu = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["scope"] == "confounder")
        & (subgroup_df["confounder"] == "UMCU")
    ]
    assert len(borderline_umcu) == 1

    borderline_tumor = ar._build_croma_subgroup_analysis(_borderline_fragility_per_sample_df())[0]
    borderline_tumor_row = borderline_tumor[
        (borderline_tumor["scope"] == "label") & (borderline_tumor["label"] == "tumor")
    ].iloc[0]
    borderline_tumor_m = np.array([_m(v) for v in (0.40, 0.50, 1.10, 1.20, 1.40, 1.50)])
    borderline_normal_m = np.array([_m(v) for v in (0.60, 1.30, 1.60, 1.70)])
    assert float(borderline_tumor_row["median_croma"]) == pytest.approx(
        float(np.median(borderline_tumor_m))
    )
    assert float(borderline_tumor_row["rest_median_croma"]) == pytest.approx(
        float(np.median(borderline_normal_m))
    )
    assert borderline_tumor_row["tier1_status"] == "relative_weakness"

    tier2_df, _ = ar._build_croma_subgroup_analysis(_tier2_supported_per_sample_df())
    hidden_row = tier2_df[
        (tier2_df["model"] == "M_tier2")
        & (tier2_df["scope"] == "stratum")
        & (tier2_df["label"] == "tumor")
        & (tier2_df["confounder"] == "RUMC")
    ].iloc[0]
    aggravated_row = tier2_df[
        (tier2_df["model"] == "M_tier2")
        & (tier2_df["scope"] == "stratum")
        & (tier2_df["label"] == "tumor")
        & (tier2_df["confounder"] == "UMCU")
    ].iloc[0]

    # Replicate the builder's subgroup LTM/drop (compute_tail_metrics at alpha=0.25)
    # on the margin-valued hidden-pocket stratum (tumor, RUMC).
    hp_m = np.array([_m(v) for v in (0.70, 0.85, 0.95, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50)])
    hp_q = np.percentile(hp_m, 25)
    hp_ltm = float(hp_m[hp_m <= hp_q].mean())
    assert float(hidden_row["subgroup_ltm_alpha"]) == pytest.approx(hp_ltm)
    assert float(hidden_row["internal_tail_drop"]) == pytest.approx(float(np.median(hp_m) - hp_ltm))
    assert float(hidden_row["croma_neg_frac"]) == pytest.approx(0.3)
    assert int(hidden_row["croma_neg_count"]) == 3
    assert hidden_row["tier2_status"] == "hidden_pocket"
    assert aggravated_row["tier2_status"] == "aggravated_weakness"


def test_markdown_suppresses_borderline_tail_overrepresentation_below_twofold() -> None:
    subgroup_df, context_df = ar._build_croma_subgroup_analysis(
        _borderline_fragility_per_sample_df()
    )

    tumor_rumc = subgroup_df[
        (subgroup_df["scope"] == "stratum")
        & (subgroup_df["label"] == "tumor")
        & (subgroup_df["confounder"] == "RUMC")
    ].iloc[0]

    assert float(tumor_rumc["tail_prevalence"]) == pytest.approx(0.50)
    assert float(tumor_rumc["context_tail_prevalence"]) == pytest.approx(0.30)
    assert float(tumor_rumc["tail_prevalence_ratio"]) == pytest.approx(5.0 / 3.0)

    markdown = ar._render_croma_subgroup_markdown(subgroup_df, context_df)

    assert "## M_borderline" in markdown
    assert "#### Tail-Specific Fragility" in markdown
    assert "relative_weakness" in markdown
    assert "tail_severe" in markdown
    assert "1.7x" in markdown
    assert "insufficient_support" in markdown


def test_tier2_requires_support_and_breadth_beyond_median_vs_ltm() -> None:
    # CRoMa is the signed margin; robustness threshold is 0, the robust-median floor
    # ~0.07 and the LTM ceiling ~-0.05 (ratio-era 1.15/0.90 mapped through (r-1)/(r+1)).
    assert (
        ar._tier2_status(
            n_samples=8,
            median_croma=0.30,
            subgroup_ltm_alpha=-0.30,
            internal_tail_drop=0.60,
            croma_neg_frac=0.25,
            croma_neg_count=3,
        )
        == "insufficient_support"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_croma=0.13,
            subgroup_ltm_alpha=-0.10,
            internal_tail_drop=0.60,
            croma_neg_frac=1.0 / 12.0,
            croma_neg_count=1,
        )
        == "neutral"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_croma=0.04,
            subgroup_ltm_alpha=-0.10,
            internal_tail_drop=0.40,
            croma_neg_frac=0.25,
            croma_neg_count=3,
        )
        == "neutral"
    )


def test_tier2_status_distinguishes_hidden_spread_and_aggravated_weakness() -> None:
    assert (
        ar._tier2_status(
            n_samples=12,
            median_croma=0.20,
            subgroup_ltm_alpha=-0.10,
            internal_tail_drop=0.45,
            croma_neg_frac=0.25,
            croma_neg_count=3,
        )
        == "hidden_pocket"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_croma=0.25,
            subgroup_ltm_alpha=0.10,
            internal_tail_drop=0.25,
            croma_neg_frac=0.0,
            croma_neg_count=0,
        )
        == "internal_spread"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_croma=-0.20,
            subgroup_ltm_alpha=-0.30,
            internal_tail_drop=0.25,
            croma_neg_frac=0.40,
            croma_neg_count=5,
        )
        == "aggravated_weakness"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_croma=0.15,
            subgroup_ltm_alpha=-0.10,
            internal_tail_drop=0.08,
            croma_neg_frac=0.25,
            croma_neg_count=3,
        )
        == "neutral"
    )


def test_tier1_status_distinguishes_broad_relative_and_aggravated_weakness() -> None:
    assert (
        ar._tier1_status(
            n_samples=4,
            median_croma=-0.10,
            rest_median_croma=0.10,
            median_delta=-0.25,
            croma_neg_delta=0.10,
        )
        == "broad_weakness"
    )
    assert (
        ar._tier1_status(
            n_samples=4,
            median_croma=0.07,
            rest_median_croma=0.20,
            median_delta=-0.20,
            croma_neg_delta=0.10,
        )
        == "relative_weakness"
    )
    assert (
        ar._tier1_status(
            n_samples=4,
            median_croma=-0.15,
            rest_median_croma=-0.08,
            median_delta=-0.15,
            croma_neg_delta=0.10,
        )
        == "aggravated_weakness"
    )
    assert (
        ar._tier1_status(
            n_samples=4,
            median_croma=0.10,
            rest_median_croma=-0.10,
            median_delta=-0.10,
            croma_neg_delta=0.10,
        )
        == "neutral"
    )


def test_markdown_renders_three_tier_tables_per_context() -> None:
    subgroup_df, context_df = ar._build_croma_subgroup_analysis(_tier2_supported_per_sample_df())

    markdown = ar._render_croma_subgroup_markdown(subgroup_df, context_df)

    assert "#### Broad Subgroup Weakness" in markdown
    assert "#### Hidden Subgroup Pockets" in markdown
    assert "#### Tail-Specific Fragility" in markdown
    assert (
        "| Scope | Subgroup | Status | Median CRoMa | Rest Median | Median Delta | CRoMa<0 Frac | Rest CRoMa<0 Frac | CRoMa<0 Delta |"
        in markdown
    )
    assert (
        "| Scope | Subgroup | Status | N | CRoMa<0 Frac | CRoMa<0 Count | Median CRoMa | Subgroup LTM@alpha | Drop |"
        in markdown
    )
    assert (
        "| Scope | Subgroup | Status | Tail Prevalence | Overall Tail Prev | Ratio | Tail Mean CRoMa | Rest Tail Mean | Severity |"
        in markdown
    )
    assert (
        "| stratum | tumor / RUMC | hidden_pocket | 10 | 0.300 | 3 | 0.121 | -0.094 | 0.215 |"
        in markdown
    )
    assert (
        "| stratum | tumor / UMCU | aggravated_weakness | 10 | 0.700 | 7 | -0.067 | -0.294 | 0.227 |"
        in markdown
    )
    assert (
        "| stratum | normal / RUMC | internal_spread | 10 | 0.000 | 0 | 0.151 | 0.035 | 0.116 |"
        in markdown
    )


def test_tail_tier_distinguishes_enriched_and_severe_cases_independently() -> None:
    subgroup_df, context_df = ar._build_croma_subgroup_analysis(_sharp_tail_only_per_sample_df())

    markdown = ar._render_croma_subgroup_markdown(subgroup_df, context_df)
    assert (
        "| stratum | tumor / RUMC | tail_enriched_and_severe | 0.500 | 0.250 | 2.0x | -0.633 | -0.429 | more severe |"
        in markdown
    )
    assert (
        "| label | tumor | tail_severe | 0.333 | 0.250 | 1.3x | -0.633 | -0.429 | more severe |"
        in markdown
    )


def test_multiclass_dataset_wide_context_is_skipped() -> None:
    per_sample_df = pd.DataFrame(
        [
            _per_sample_row(
                model="M1",
                sample_id="a1",
                label="A",
                confounder="X",
                croma_m1=0.8,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="a2",
                label="A",
                confounder="Y",
                croma_m1=1.2,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="b1",
                label="B",
                confounder="X",
                croma_m1=0.9,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="b2",
                label="B",
                confounder="Y",
                croma_m1=1.3,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="c1",
                label="C",
                confounder="X",
                croma_m1=0.95,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="c2",
                label="C",
                confounder="Y",
                croma_m1=1.4,
                dataset="tcga",
            ),
        ]
    )

    subgroup_df, context_df = ar._build_croma_subgroup_analysis(per_sample_df)

    assert subgroup_df.empty
    skipped = context_df.iloc[0]
    assert bool(skipped["skipped"]) is True
    assert "heterogeneous biological boundaries" in skipped["skip_reason"]

    markdown = ar._render_croma_subgroup_markdown(subgroup_df, context_df)
    assert "Skipped" in markdown
    assert "paired runs" in markdown


def test_paired_contexts_are_analyzed_independently() -> None:
    per_sample_df = pd.DataFrame(
        [
            _per_sample_row(
                model="M1",
                sample_id="ab_tx_1",
                label="A",
                confounder="X",
                croma_m1=0.40,
                evaluation_design="paired_2x2",
                subset="A+B__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ab_ty_2",
                label="A",
                confounder="Y",
                croma_m1=0.60,
                evaluation_design="paired_2x2",
                subset="A+B__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ab_bx_3",
                label="B",
                confounder="X",
                croma_m1=1.20,
                evaluation_design="paired_2x2",
                subset="A+B__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ab_by_4",
                label="B",
                confounder="Y",
                croma_m1=1.30,
                evaluation_design="paired_2x2",
                subset="A+B__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ac_tx_1",
                label="A",
                confounder="X",
                croma_m1=1.40,
                evaluation_design="paired_2x2",
                subset="A+C__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ac_ty_2",
                label="A",
                confounder="Y",
                croma_m1=1.30,
                evaluation_design="paired_2x2",
                subset="A+C__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ac_cx_3",
                label="C",
                confounder="X",
                croma_m1=0.50,
                evaluation_design="paired_2x2",
                subset="A+C__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ac_cy_4",
                label="C",
                confounder="Y",
                croma_m1=0.60,
                evaluation_design="paired_2x2",
                subset="A+C__X_Y",
                dataset="tcga",
            ),
        ]
    )

    subgroup_df, context_df = ar._build_croma_subgroup_analysis(per_sample_df)

    assert set(context_df["context_id"]) == {"A+B__X_Y", "A+C__X_Y"}
    ab_rows = subgroup_df[
        (subgroup_df["context_id"] == "A+B__X_Y") & (subgroup_df["scope"] != "confounder")
    ]
    ac_rows = subgroup_df[
        (subgroup_df["context_id"] == "A+C__X_Y") & (subgroup_df["scope"] != "confounder")
    ]
    assert set(ab_rows["label"]) == {"A", "B"}
    assert set(ac_rows["label"]) == {"A", "C"}


def test_low_support_groups_remain_in_markdown_with_insufficient_support_status() -> None:
    per_sample_df = pd.DataFrame(
        [
            _per_sample_row(
                model="M1",
                sample_id="a1",
                label="tumor",
                confounder="RUMC",
                croma_m1=0.30,
            ),
            _per_sample_row(
                model="M1",
                sample_id="a2",
                label="tumor",
                confounder="RUMC",
                croma_m1=0.40,
            ),
            _per_sample_row(
                model="M1",
                sample_id="a3",
                label="normal",
                confounder="UMCU",
                croma_m1=1.10,
            ),
            _per_sample_row(
                model="M1",
                sample_id="a4",
                label="normal",
                confounder="UMCU",
                croma_m1=1.20,
            ),
            _per_sample_row(
                model="M1",
                sample_id="a5",
                label="normal",
                confounder="CWZ",
                croma_m1=0.35,
            ),
        ]
    )

    subgroup_df, context_df = ar._build_croma_subgroup_analysis(per_sample_df)
    cwz_row = subgroup_df[
        (subgroup_df["scope"] == "stratum")
        & (subgroup_df["label"] == "normal")
        & (subgroup_df["confounder"] == "CWZ")
    ].iloc[0]
    assert int(cwz_row["n_samples"]) == 1

    markdown = ar._render_croma_subgroup_markdown(subgroup_df, context_df)
    assert "| stratum | normal / CWZ | insufficient_support |" in markdown
    assert "| confounder | CWZ | insufficient_support |" in markdown


def test_main_writes_model_specific_croma_subgroup_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    per_sample_csv = tmp_path / "per_sample_metrics.csv"
    out_dir = tmp_path / "analysis"
    _metrics_rows(["M_fragile", "M_stable"]).to_csv(metrics_csv, index=False)
    _binary_camelyon_like_per_sample_df().to_csv(per_sample_csv, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_results.py",
            "--metrics-csv",
            str(metrics_csv),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert ar.main() == 0

    subgroup_df = pd.read_csv(out_dir / "model_specific_croma_subgroups.csv")
    markdown = (out_dir / "model_specific_croma_subgroups.md").read_text(encoding="utf-8")

    assert set(subgroup_df["scope"]) == {"stratum", "label", "confounder"}
    assert "tumor" in markdown
    assert "Broad Subgroup Weakness" in markdown


def test_model_action_flags_use_only_lower_coverage_risk_threshold() -> None:
    df_model = pd.DataFrame(
        {
            "model": ["M1", "M2"],
            "ri": [0.80, 0.82],
            "mari": [0.78, 0.81],
            "croma": [1.10, 1.12],
            "ri_undefined_frac": [0.26, 0.05],
            "mari_undefined_frac": [0.24, 0.31],
        }
    )

    flags = ar._model_action_flags(df_model=df_model)

    assert set(flags["flag"]) == {"coverage_risk"}
    assert set(flags["model"]) == {"M1", "M2"}
    assert all(float(v) == pytest.approx(0.25) for v in flags["threshold"])
    assert set(round(float(v), 3) for v in flags["value"]) == {0.26, 0.31}


def test_model_action_flags_keep_only_coverage_embedding_and_ltm_tail_flags() -> None:
    df_model = pd.DataFrame(
        {
            "model": ["M1"],
            "ri": [0.80],
            "mari": [0.79],
            "croma": [1.10],
            "croma_q_alpha": [0.90],
            "croma_ltm_alpha": [0.70],
            "ri_undefined_frac": [0.30],
            "mari_undefined_frac": [0.28],
            "ri_ss_dominated_undefined_frac": [0.24],
            "mari_ss_dominated_undefined_frac": [0.26],
            "ri_oo_dominated_undefined_frac": [0.12],
            "mari_oo_dominated_undefined_frac": [0.11],
        }
    )

    flags = ar._model_action_flags(df_model=df_model)

    assert set(flags["flag"]) == {
        "coverage_risk",
        "poor_embedding",
        "tail_gap_ltm_high",
    }
    assert "tail_gap_q_high" not in set(flags["flag"])
    assert not any(str(flag).startswith("entangled_clusters_") for flag in flags["flag"])
    assert not any(str(flag).startswith("ss_dominated_undefined_") for flag in flags["flag"])
    coverage_flag = flags[flags["flag"] == "coverage_risk"].iloc[0]
    poor_embedding_flag = flags[flags["flag"] == "poor_embedding"].iloc[0]
    assert float(coverage_flag["value"]) == pytest.approx(0.30)
    assert float(poor_embedding_flag["value"]) == pytest.approx(0.12)
