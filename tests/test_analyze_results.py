import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_results as ar


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
                "ccmr": 1.18 + 0.01 * idx,
                "ccmr_q_alpha": 0.82 - 0.01 * idx,
                "ccmr_ltm_alpha": 0.71 - 0.005 * idx,
            }
        )
    return pd.DataFrame(rows)


def _per_sample_row(
    *,
    model: str,
    sample_id: str,
    label: str,
    confounder: str,
    ccmr_m1: float,
    evaluation_design: str = "dataset_wide",
    subset: str = "dataset",
    dataset: str = "camelyon",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": model,
        "evaluation_design": evaluation_design,
        "evaluation_unit": (
            "sample" if evaluation_design == "dataset_wide" else "occurrence"
        ),
        "subset": subset,
        "sample_id": sample_id,
        "slide_id": f"slide-{sample_id}",
        "label": label,
        "confounder": confounder,
        "ccmr_alpha": 0.25,
        "ccmr_search": "start=200;growth=2;alpha=0.25",
        "ccmr_m1": float(ccmr_m1),
    }


def _binary_camelyon_like_per_sample_df() -> pd.DataFrame:
    rows = [
        _per_sample_row(
            model="M_fragile",
            sample_id="a_t_r_1",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=0.40,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="b_t_r_2",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=0.50,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="c_t_u_3",
            label="tumor",
            confounder="UMCU",
            ccmr_m1=0.90,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="d_t_u_4",
            label="tumor",
            confounder="UMCU",
            ccmr_m1=1.60,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="z_n_r_5",
            label="normal",
            confounder="RUMC",
            ccmr_m1=0.50,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="f_n_r_6",
            label="normal",
            confounder="RUMC",
            ccmr_m1=1.20,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="g_n_u_7",
            label="normal",
            confounder="UMCU",
            ccmr_m1=1.30,
        ),
        _per_sample_row(
            model="M_fragile",
            sample_id="h_n_u_8",
            label="normal",
            confounder="UMCU",
            ccmr_m1=1.40,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="a_t_r_1",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=0.70,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="b_t_r_2",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=1.05,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="c_t_u_3",
            label="tumor",
            confounder="UMCU",
            ccmr_m1=1.10,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="d_t_u_4",
            label="tumor",
            confounder="UMCU",
            ccmr_m1=1.20,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="z_n_r_5",
            label="normal",
            confounder="RUMC",
            ccmr_m1=0.75,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="f_n_r_6",
            label="normal",
            confounder="RUMC",
            ccmr_m1=1.00,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="g_n_u_7",
            label="normal",
            confounder="UMCU",
            ccmr_m1=1.15,
        ),
        _per_sample_row(
            model="M_stable",
            sample_id="h_n_u_8",
            label="normal",
            confounder="UMCU",
            ccmr_m1=1.30,
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
            ccmr_m1=0.40,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="a2",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=0.50,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="a3",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=1.10,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="a4",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=1.20,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="b1",
            label="normal",
            confounder="RUMC",
            ccmr_m1=0.60,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="b2",
            label="normal",
            confounder="RUMC",
            ccmr_m1=1.30,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="c1",
            label="tumor",
            confounder="UMCU",
            ccmr_m1=1.40,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="c2",
            label="tumor",
            confounder="UMCU",
            ccmr_m1=1.50,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="d1",
            label="normal",
            confounder="UMCU",
            ccmr_m1=1.60,
        ),
        _per_sample_row(
            model="M_borderline",
            sample_id="d2",
            label="normal",
            confounder="UMCU",
            ccmr_m1=1.70,
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
            ccmr_m1=0.20,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="a2",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=0.25,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="a3",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=1.60,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="a4",
            label="tumor",
            confounder="RUMC",
            ccmr_m1=1.70,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="b1",
            label="normal",
            confounder="RUMC",
            ccmr_m1=0.40,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="b2",
            label="normal",
            confounder="RUMC",
            ccmr_m1=0.80,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="b3",
            label="normal",
            confounder="RUMC",
            ccmr_m1=0.85,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="b4",
            label="normal",
            confounder="RUMC",
            ccmr_m1=0.90,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="c1",
            label="tumor",
            confounder="UMCU",
            ccmr_m1=0.95,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="c2",
            label="tumor",
            confounder="UMCU",
            ccmr_m1=1.05,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="d1",
            label="normal",
            confounder="UMCU",
            ccmr_m1=1.10,
        ),
        _per_sample_row(
            model="M_sharp",
            sample_id="d2",
            label="normal",
            confounder="UMCU",
            ccmr_m1=1.20,
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
                ccmr_m1=value,
            )
        )
    for idx, value in enumerate(internal_spread_values, start=1):
        rows.append(
            _per_sample_row(
                model="M_tier2",
                sample_id=f"is_{idx}",
                label="normal",
                confounder="RUMC",
                ccmr_m1=value,
            )
        )
    for idx, value in enumerate(aggravated_values, start=1):
        rows.append(
            _per_sample_row(
                model="M_tier2",
                sample_id=f"aw_{idx}",
                label="tumor",
                confounder="UMCU",
                ccmr_m1=value,
            )
        )
    for idx, value in enumerate(neutral_values, start=1):
        rows.append(
            _per_sample_row(
                model="M_tier2",
                sample_id=f"ne_{idx}",
                label="normal",
                confounder="UMCU",
                ccmr_m1=value,
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


def test_k_sweep_sensitivity_separates_evaluation_designs() -> None:
    df = pd.DataFrame(
        {
            "model": ["M1", "M1", "M1", "M1"],
            "evaluation_design": [
                "dataset_wide",
                "dataset_wide",
                "paired_2x2",
                "paired_2x2",
            ],
            "evaluation_unit": ["sample", "sample", "occurrence", "occurrence"],
            "k": [1, 3, 1, 3],
            "ri": [0.8, 0.7, 0.5, 0.2],
            "mari": [0.75, 0.7, 0.45, 0.1],
        }
    )

    grouped = ar._k_sweep_sensitivity(df)

    assert set(grouped["model"]) == {
        "M1 [dataset_wide;sample]",
        "M1 [paired_2x2;occurrence]",
    }
    paired_row = grouped[grouped["model"] == "M1 [paired_2x2;occurrence]"].iloc[0]
    assert float(paired_row["ri_range"]) == 0.3


def test_build_ccmr_subgroup_analysis_highlights_tumor_fragility() -> None:
    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(
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

    assert float(tumor_row["mean_ccmr"]) == pytest.approx(0.85)
    assert float(tumor_row["rest_mean_ccmr"]) == pytest.approx(1.10)
    assert float(tumor_row["mean_ccmr_delta_vs_rest"]) == pytest.approx(-0.25)
    assert float(tumor_row["tail_prevalence"]) == pytest.approx(0.50)
    assert float(tumor_row["tail_share"]) == pytest.approx(1.0)
    assert float(tumor_row["group_frac"]) == pytest.approx(0.50)
    assert float(tumor_row["rest_ccmr_lt1_frac"]) == pytest.approx(0.25)
    assert float(tumor_row["ccmr_lt1_frac_delta_vs_rest"]) == pytest.approx(0.50)
    assert float(normal_row["tail_prevalence"]) == pytest.approx(0.0)

    markdown = ar._render_ccmr_subgroup_markdown(subgroup_df, context_df)
    assert "tumor" in markdown
    assert "RUMC" in markdown
    assert "tail_enriched" in markdown
    assert "2.0x" in markdown
    assert "no_rest_tail" in markdown


def test_subgroup_analysis_reports_primary_and_supporting_scopes() -> None:
    subgroup_df, _context_df = ar._build_ccmr_subgroup_analysis(
        _binary_camelyon_like_per_sample_df()
    )

    assert set(subgroup_df["scope"]) == {"stratum", "label", "confounder"}

    fragile_rows = subgroup_df[subgroup_df["model"] == "M_fragile"]
    stratum_top = fragile_rows[fragile_rows["scope"] == "stratum"].sort_values(
        ["tail_prevalence", "mean_ccmr"], ascending=[False, True]
    )
    assert tuple(stratum_top.iloc[0][["label", "confounder"]]) == ("tumor", "RUMC")


def test_exact_tail_membership_uses_alpha_and_sample_id_tiebreak() -> None:
    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(
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


def test_ccmr_lt1_frac_and_tail_prevalence_stay_distinct() -> None:
    subgroup_df, _context_df = ar._build_ccmr_subgroup_analysis(
        _binary_camelyon_like_per_sample_df()
    )

    tumor_row = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["scope"] == "label")
        & (subgroup_df["label"] == "tumor")
    ].iloc[0]

    assert float(tumor_row["ccmr_lt1_frac"]) == pytest.approx(0.75)
    assert float(tumor_row["tail_prevalence"]) == pytest.approx(0.50)


def test_subgroup_rows_include_tier_metrics_and_statuses() -> None:
    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(
        _binary_camelyon_like_per_sample_df()
    )

    assert len(context_df) == 2
    assert {
        "subgroup_name",
        "rest_median_ccmr",
        "median_ccmr_delta_vs_rest",
        "tier1_status",
        "ccmr_lt1_count",
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
    assert float(fragile_tumor["rest_median_ccmr"]) == pytest.approx(1.25)
    assert float(fragile_tumor["median_ccmr_delta_vs_rest"]) == pytest.approx(-0.55)
    assert fragile_tumor["tier1_status"] == "broad_weakness"
    assert fragile_tumor["tail_severity_label"] == "no_rest_tail"
    assert fragile_tumor["tier3_status"] == "tail_enriched"

    borderline_umcu = subgroup_df[
        (subgroup_df["model"] == "M_fragile")
        & (subgroup_df["scope"] == "confounder")
        & (subgroup_df["confounder"] == "UMCU")
    ]
    assert len(borderline_umcu) == 1

    borderline_tumor = ar._build_ccmr_subgroup_analysis(
        _borderline_fragility_per_sample_df()
    )[0]
    borderline_tumor_row = borderline_tumor[
        (borderline_tumor["scope"] == "label") & (borderline_tumor["label"] == "tumor")
    ].iloc[0]
    assert float(borderline_tumor_row["median_ccmr"]) == pytest.approx(1.15)
    assert float(borderline_tumor_row["rest_median_ccmr"]) == pytest.approx(1.45)
    assert borderline_tumor_row["tier1_status"] == "relative_weakness"

    tier2_df, _ = ar._build_ccmr_subgroup_analysis(_tier2_supported_per_sample_df())
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

    assert float(hidden_row["subgroup_ltm_alpha"]) == pytest.approx(0.8333333333)
    assert float(hidden_row["internal_tail_drop"]) == pytest.approx(0.4416666667)
    assert float(hidden_row["ccmr_lt1_frac"]) == pytest.approx(0.3)
    assert int(hidden_row["ccmr_lt1_count"]) == 3
    assert hidden_row["tier2_status"] == "hidden_pocket"
    assert aggravated_row["tier2_status"] == "aggravated_weakness"


def test_markdown_suppresses_borderline_tail_overrepresentation_below_twofold() -> None:
    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(
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

    markdown = ar._render_ccmr_subgroup_markdown(subgroup_df, context_df)

    assert "## M_borderline" in markdown
    assert "#### Tail-Specific Fragility" in markdown
    assert "relative_weakness" in markdown
    assert "tail_severe" in markdown
    assert "1.7x" in markdown
    assert "insufficient_support" in markdown


def test_tier2_requires_support_and_breadth_beyond_median_vs_ltm() -> None:
    assert (
        ar._tier2_status(
            n_samples=8,
            median_ccmr=1.30,
            subgroup_ltm_alpha=0.70,
            internal_tail_drop=0.60,
            ccmr_lt1_frac=0.25,
            ccmr_lt1_count=3,
        )
        == "insufficient_support"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_ccmr=1.30,
            subgroup_ltm_alpha=0.70,
            internal_tail_drop=0.60,
            ccmr_lt1_frac=1.0 / 12.0,
            ccmr_lt1_count=1,
        )
        == "neutral"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_ccmr=1.10,
            subgroup_ltm_alpha=0.70,
            internal_tail_drop=0.40,
            ccmr_lt1_frac=0.25,
            ccmr_lt1_count=3,
        )
        == "neutral"
    )


def test_tier2_status_distinguishes_hidden_spread_and_aggravated_weakness() -> None:
    assert (
        ar._tier2_status(
            n_samples=12,
            median_ccmr=1.25,
            subgroup_ltm_alpha=0.80,
            internal_tail_drop=0.45,
            ccmr_lt1_frac=0.25,
            ccmr_lt1_count=3,
        )
        == "hidden_pocket"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_ccmr=1.35,
            subgroup_ltm_alpha=1.10,
            internal_tail_drop=0.25,
            ccmr_lt1_frac=0.0,
            ccmr_lt1_count=0,
        )
        == "internal_spread"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_ccmr=0.80,
            subgroup_ltm_alpha=0.55,
            internal_tail_drop=0.25,
            ccmr_lt1_frac=0.40,
            ccmr_lt1_count=5,
        )
        == "aggravated_weakness"
    )
    assert (
        ar._tier2_status(
            n_samples=12,
            median_ccmr=1.20,
            subgroup_ltm_alpha=0.95,
            internal_tail_drop=0.10,
            ccmr_lt1_frac=0.25,
            ccmr_lt1_count=3,
        )
        == "neutral"
    )


def test_tier1_status_distinguishes_broad_relative_and_aggravated_weakness() -> None:
    assert (
        ar._tier1_status(
            n_samples=4,
            median_ccmr=0.85,
            rest_median_ccmr=1.10,
            median_delta=-0.25,
            ccmr_lt1_delta=0.10,
        )
        == "broad_weakness"
    )
    assert (
        ar._tier1_status(
            n_samples=4,
            median_ccmr=1.15,
            rest_median_ccmr=1.35,
            median_delta=-0.20,
            ccmr_lt1_delta=0.10,
        )
        == "relative_weakness"
    )
    assert (
        ar._tier1_status(
            n_samples=4,
            median_ccmr=0.75,
            rest_median_ccmr=0.90,
            median_delta=-0.15,
            ccmr_lt1_delta=0.10,
        )
        == "aggravated_weakness"
    )
    assert (
        ar._tier1_status(
            n_samples=4,
            median_ccmr=1.10,
            rest_median_ccmr=0.90,
            median_delta=-0.10,
            ccmr_lt1_delta=0.10,
        )
        == "neutral"
    )


def test_markdown_renders_three_tier_tables_per_context() -> None:
    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(
        _tier2_supported_per_sample_df()
    )

    markdown = ar._render_ccmr_subgroup_markdown(subgroup_df, context_df)

    assert "#### Broad Subgroup Weakness" in markdown
    assert "#### Hidden Subgroup Pockets" in markdown
    assert "#### Tail-Specific Fragility" in markdown
    assert (
        "| Scope | Subgroup | Status | Median CCMR | Rest Median | Median Delta | CCMR<1 Frac | Rest CCMR<1 Frac | CCMR<1 Delta |"
        in markdown
    )
    assert (
        "| Scope | Subgroup | Status | N | CCMR<1 Frac | CCMR<1 Count | Median CCMR | Subgroup LTM@alpha | Drop |"
        in markdown
    )
    assert (
        "| Scope | Subgroup | Status | Tail Prevalence | Overall Tail Prev | Ratio | Tail Mean CCMR | Rest Tail Mean | Severity |"
        in markdown
    )
    assert (
        "| stratum | tumor / RUMC | hidden_pocket | 10 | 0.300 | 3 | 1.275 | 0.833 | 0.442 |"
        in markdown
    )
    assert (
        "| stratum | tumor / UMCU | aggravated_weakness | 10 | 0.700 | 7 | 0.875 | 0.550 | 0.325 |"
        in markdown
    )
    assert (
        "| stratum | normal / RUMC | internal_spread | 10 | 0.000 | 0 | 1.355 | 1.073 | 0.282 |"
        in markdown
    )


def test_tail_tier_distinguishes_enriched_and_severe_cases_independently() -> None:
    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(
        _sharp_tail_only_per_sample_df()
    )

    markdown = ar._render_ccmr_subgroup_markdown(subgroup_df, context_df)
    assert (
        "| stratum | tumor / RUMC | tail_enriched_and_severe | 0.500 | 0.250 | 2.0x | 0.225 | 0.400 | more severe |"
        in markdown
    )
    assert (
        "| label | tumor | tail_severe | 0.333 | 0.250 | 1.3x | 0.225 | 0.400 | more severe |"
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
                ccmr_m1=0.8,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="a2",
                label="A",
                confounder="Y",
                ccmr_m1=1.2,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="b1",
                label="B",
                confounder="X",
                ccmr_m1=0.9,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="b2",
                label="B",
                confounder="Y",
                ccmr_m1=1.3,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="c1",
                label="C",
                confounder="X",
                ccmr_m1=0.95,
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="c2",
                label="C",
                confounder="Y",
                ccmr_m1=1.4,
                dataset="tcga",
            ),
        ]
    )

    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(per_sample_df)

    assert subgroup_df.empty
    skipped = context_df.iloc[0]
    assert bool(skipped["skipped"]) is True
    assert "heterogeneous biological boundaries" in skipped["skip_reason"]

    markdown = ar._render_ccmr_subgroup_markdown(subgroup_df, context_df)
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
                ccmr_m1=0.40,
                evaluation_design="paired_2x2",
                subset="A+B__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ab_ty_2",
                label="A",
                confounder="Y",
                ccmr_m1=0.60,
                evaluation_design="paired_2x2",
                subset="A+B__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ab_bx_3",
                label="B",
                confounder="X",
                ccmr_m1=1.20,
                evaluation_design="paired_2x2",
                subset="A+B__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ab_by_4",
                label="B",
                confounder="Y",
                ccmr_m1=1.30,
                evaluation_design="paired_2x2",
                subset="A+B__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ac_tx_1",
                label="A",
                confounder="X",
                ccmr_m1=1.40,
                evaluation_design="paired_2x2",
                subset="A+C__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ac_ty_2",
                label="A",
                confounder="Y",
                ccmr_m1=1.30,
                evaluation_design="paired_2x2",
                subset="A+C__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ac_cx_3",
                label="C",
                confounder="X",
                ccmr_m1=0.50,
                evaluation_design="paired_2x2",
                subset="A+C__X_Y",
                dataset="tcga",
            ),
            _per_sample_row(
                model="M1",
                sample_id="ac_cy_4",
                label="C",
                confounder="Y",
                ccmr_m1=0.60,
                evaluation_design="paired_2x2",
                subset="A+C__X_Y",
                dataset="tcga",
            ),
        ]
    )

    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(per_sample_df)

    assert set(context_df["context_id"]) == {"A+B__X_Y", "A+C__X_Y"}
    ab_rows = subgroup_df[
        (subgroup_df["context_id"] == "A+B__X_Y")
        & (subgroup_df["scope"] != "confounder")
    ]
    ac_rows = subgroup_df[
        (subgroup_df["context_id"] == "A+C__X_Y")
        & (subgroup_df["scope"] != "confounder")
    ]
    assert set(ab_rows["label"]) == {"A", "B"}
    assert set(ac_rows["label"]) == {"A", "C"}


def test_low_support_groups_remain_in_markdown_with_insufficient_support_status() -> (
    None
):
    per_sample_df = pd.DataFrame(
        [
            _per_sample_row(
                model="M1",
                sample_id="a1",
                label="tumor",
                confounder="RUMC",
                ccmr_m1=0.30,
            ),
            _per_sample_row(
                model="M1",
                sample_id="a2",
                label="tumor",
                confounder="RUMC",
                ccmr_m1=0.40,
            ),
            _per_sample_row(
                model="M1",
                sample_id="a3",
                label="normal",
                confounder="UMCU",
                ccmr_m1=1.10,
            ),
            _per_sample_row(
                model="M1",
                sample_id="a4",
                label="normal",
                confounder="UMCU",
                ccmr_m1=1.20,
            ),
            _per_sample_row(
                model="M1",
                sample_id="a5",
                label="normal",
                confounder="CWZ",
                ccmr_m1=0.35,
            ),
        ]
    )

    subgroup_df, context_df = ar._build_ccmr_subgroup_analysis(per_sample_df)
    cwz_row = subgroup_df[
        (subgroup_df["scope"] == "stratum")
        & (subgroup_df["label"] == "normal")
        & (subgroup_df["confounder"] == "CWZ")
    ].iloc[0]
    assert int(cwz_row["n_samples"]) == 1

    markdown = ar._render_ccmr_subgroup_markdown(subgroup_df, context_df)
    assert "| stratum | normal / CWZ | insufficient_support |" in markdown
    assert "| confounder | CWZ | insufficient_support |" in markdown


def test_main_writes_model_specific_ccmr_subgroup_outputs(
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

    subgroup_df = pd.read_csv(out_dir / "model_specific_ccmr_subgroups.csv")
    markdown = (out_dir / "model_specific_ccmr_subgroups.md").read_text(
        encoding="utf-8"
    )

    assert set(subgroup_df["scope"]) == {"stratum", "label", "confounder"}
    assert "tumor" in markdown
    assert "Broad Subgroup Weakness" in markdown


def test_model_action_flags_use_only_lower_coverage_risk_threshold() -> None:
    df_model = pd.DataFrame(
        {
            "model": ["M1", "M2"],
            "ri": [0.80, 0.82],
            "mari": [0.78, 0.81],
            "ccmr": [1.10, 1.12],
            "ri_undefined_frac": [0.26, 0.05],
            "mari_undefined_frac": [0.24, 0.31],
        }
    )

    flags = ar._model_action_flags(
        df_model=df_model,
        delta_df=pd.DataFrame(),
        k_sensitivity_df=pd.DataFrame(),
        ccmr_m_sensitivity_df=pd.DataFrame(),
    )

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
            "ccmr": [1.10],
            "ccmr_q_alpha": [0.90],
            "ccmr_ltm_alpha": [0.70],
            "ri_undefined_frac": [0.30],
            "mari_undefined_frac": [0.28],
            "ri_ss_dominated_undefined_frac": [0.24],
            "mari_ss_dominated_undefined_frac": [0.26],
            "ri_oo_dominated_undefined_frac": [0.12],
            "mari_oo_dominated_undefined_frac": [0.11],
        }
    )

    flags = ar._model_action_flags(
        df_model=df_model,
        delta_df=pd.DataFrame(),
        k_sensitivity_df=pd.DataFrame(),
        ccmr_m_sensitivity_df=pd.DataFrame(),
    )

    assert set(flags["flag"]) == {
        "coverage_risk",
        "poor_embedding",
        "tail_gap_ltm_high",
    }
    assert "tail_gap_q_high" not in set(flags["flag"])
    assert not any(
        str(flag).startswith("entangled_clusters_") for flag in flags["flag"]
    )
    assert not any(
        str(flag).startswith("ss_dominated_undefined_") for flag in flags["flag"]
    )
    coverage_flag = flags[flags["flag"] == "coverage_risk"].iloc[0]
    poor_embedding_flag = flags[flags["flag"] == "poor_embedding"].iloc[0]
    assert float(coverage_flag["value"]) == pytest.approx(0.30)
    assert float(poor_embedding_flag["value"]) == pytest.approx(0.12)


def test_report_adds_coverage_section_and_filters_coverage_and_rank_shift_from_additional_flags(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "analysis_report.md"
    df_raw = _metrics_rows(["M1", "M2"])
    df_model = pd.DataFrame(
        {
            "model": ["M1", "M2"],
            "ri": [0.80, 0.82],
            "mari": [0.78, 0.81],
            "ccmr": [1.10, 1.12],
            "ri_undefined_frac": [0.30, 0.10],
            "mari_undefined_frac": [0.28, 0.10],
        }
    )
    top_df = pd.DataFrame(
        {
            "metric": ["ri"],
            "rank_position": [1],
            "model": ["M1"],
            "value": [0.80],
        }
    )
    delta_df = pd.DataFrame(
        {
            "model": ["M1"],
            "metric_a": ["ri"],
            "metric_b": ["mari"],
            "pair": ["ri_vs_mari"],
            "rank_a": [3.0],
            "rank_b": [1.0],
            "improvement_delta": [2.0],
            "improvement_delta_signed": ["+2"],
            "direction": ["improvement"],
            "abs_improvement_delta": [2.0],
        }
    )
    pearson = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["ri", "mari"],
        columns=["ri", "mari"],
    )
    action_flags_df = pd.DataFrame(
        [
            {
                "model": "M1",
                "flag": "coverage_risk",
                "severity": "high",
                "value": 0.30,
                "threshold": 0.25,
                "detail": "Undefined coverage is high.",
            },
            {
                "model": "M1",
                "flag": "rank_shift_ri_vs_mari",
                "severity": "high",
                "value": 2.0,
                "threshold": 2.0,
                "detail": "Rank shift.",
            },
            {
                "model": "M1",
                "flag": "poor_embedding",
                "severity": "high",
                "value": 0.12,
                "threshold": 0.10,
                "detail": "Poor embedding.",
            },
        ]
    )

    ar._write_report(
        out_path=out_path,
        input_csv=tmp_path / "metrics.csv",
        df_raw=df_raw,
        df_model=df_model,
        top_metrics=["ri"],
        corr_metrics=["ri", "mari"],
        rank_metrics=["ri", "mari"],
        rank_reference="ri",
        top_df=top_df,
        delta_df=delta_df,
        pearson=pearson,
        spearman=pearson,
        top_k=1,
        action_flags_df=action_flags_df,
        k_sensitivity_df=pd.DataFrame(),
        ccmr_m_sensitivity_df=pd.DataFrame(),
    )

    report = out_path.read_text(encoding="utf-8")

    assert "## Spearman Correlations" in report
    assert "## Coverage Risk" in report
    assert "| Model | Undefined Frac | Coverage Risk |" in report
    assert "| M1 | 0.300 | yes |" in report
    assert "| M2 | 0.100 | no |" in report

    additional_section = report.split(
        "## Additional Insights and Action Flags", maxsplit=1
    )[1]
    additional_section = additional_section.split("## K-Sweep Sensitivity", maxsplit=1)[
        0
    ]
    assert "`coverage_risk`" not in additional_section
    assert "`rank_shift_ri_vs_mari`" not in additional_section
    assert "`poor_embedding`" in additional_section
