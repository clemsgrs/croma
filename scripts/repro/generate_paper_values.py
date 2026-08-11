"""Emit a LaTeX macro file of derived per-benchmark scalars cited inline in the paper.

Inline prose numbers (the pooled-CRoMa span, counts of confounder-dominant models,
the best lower-tail mean, ...) were historically hand-typed, which let them drift
from the auto-generated result tables (e.g. a span upper bound truncated 0.195->0.19
while ``generate_results_table.py`` rounded the same value to 0.20). This script
derives those scalars from the same ``metrics.csv`` the tables are built from and
writes them as ``\\newcommand`` macros, so the prose can cite ``\\CamelyonCromaSpan``
instead of a literal. Run it whenever the faithful metrics are regenerated.

Scale safety (see the ratio-vs-margin provenance hazard): CRoMa is stored as the
signed margin in ``(-1, 1)`` in the canonical dirs, but some older dirs hold the raw
ratio ``d_OS / d_SO`` in ``(0, inf)``. Each benchmark's scale is auto-detected and a
ratio column is transformed to the margin via ``(r - 1) / (r + 1)`` before any
statistic is computed, so the emitted values are always on the paper's margin scale.

Usage:
  python scripts/repro/generate_paper_values.py
  python scripts/repro/generate_paper_values.py --out paper/generated_values.tex
  python scripts/repro/generate_paper_values.py --scale ratio   # force-transform inputs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from _paper_tables import (
    CaptionClaimError,
    bare_num as _apd_bare,
    ci_bracket as _ci,
    croma_as_margin,
    detect_croma_scale as _detect_scale,
    num_math as _num,
    pct_round as _pct,
    to_margin as _to_margin,
)
from paper_manifest import TABLES, by_prefix
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting.style import published_models  # noqa: E402

# Which run backs each macro family is the manifest's business, not this script's. It used
# to be two lists here (a median-k one and a k-star "faithful" one behind a USE_MEDIAN_K
# flag) plus a third in reproduce_faithful.py, and they disagreed: the tables were rendered
# from k-star while these macros were computed from median-k. See ADR-0010.
BENCHMARKS: list[tuple[str, str]] = [(t.prefix, t.metrics_rel) for t in TABLES]

# SS-shell (local-entanglement) scalars for concern 6. Sourced from the typed-neighbour
# -rank experiment summary, which carries per-model CRoMa, SS-shell exit depth, and the
# fixed-k SS-pocket prevalence (fraction with no typed neighbour among the k nearest).
SS_SHELL_SUMMARY = (
    "Camelyon",
    f"{by_prefix('Camelyon').studies_rel}/typed_neighbor_rank_summary.csv",
)
SS_POCKET_K = 10  # reference neighbourhood for the prevalence quoted in prose

# Uncertainty scalars for concerns 3 (UQ) and 6 (redundancy). Sourced from the
# bootstrap_uncertainty experiment: per-model pooled-median CRoMa CIs + rank stability
# (CSV) and the cross-model Spearman correlations with bootstrap CIs (JSON).
UNCERTAINTY_SUMMARY = (
    "Camelyon",
    f"{by_prefix('Camelyon').run_rel}/results/bootstrap_uncertainty.json",
    f"{by_prefix('Camelyon').run_rel}/results/bootstrap_uncertainty.csv",
)

# Downstream-validation scalars. Correlations are reported per benchmark only; PCaBiop
# is descriptive because it contains four slide encoders.
APD_CORRELATION_CSV = "output/studies/apd/apd_correlation.csv"
APD_STUDY_CSV = "output/studies/apd/apd.csv"
APD_TARGET_MACRO = {
    "nipd_id": "NipdId",
    "nipd_ood": "NipdOod",
    "apd_id": "ApdId",
    "apd_ood": "ApdOod",
}
APD_METRIC_MACRO = {"croma": "Croma", "ri": "Ri", "mari": "Mari"}
APD_SCOPE_MACRO = {
    "camelyon": "Camelyon",
    "tcga_4x4": "Tcga",
    "tolkach": "Tolkach",
    "pcabiop": "Pcabiop",
}
APD_RANGE_BENCHMARKS = list(APD_SCOPE_MACRO)

# Confounder-probe collapse. `confounder_knn_bacc` is a k-free scalar the pipeline already
# computes: how decodable the confounder is from the frozen representation. Its rank
# correlation with each reported statistic establishes that every *pooled* robustness score
# -- RI, MaRI, the CRoMa median, and F(0) -- is a near-monotone transform of confounder
# decodability, while LTM escapes wherever the tail decouples from the median. Computed on
# the three headline PathoROB benchmarks; the natural-image control is excluded, since it is
# a floor rather than a ranked competitor.
PROBE_BENCHMARKS = ["Camelyon", "Tolkach", "TcgaFourByFour"]
PROBE_POOLED_TARGETS = [("ri", "Ri"), ("mari", "Mari"), ("croma", "Croma"), ("f0", "FZero")]
PROBE_TAIL_TARGET = ("croma_ltm_alpha", "Ltm")

# The probe-saturation example: two models whose confounder decodability is indistinguishable
# at the ceiling, yet whose CRoMa medians are far apart. Prose names both models, so their
# identity is pinned here rather than derived (e.g. as "the top two by confounder bacc",
# which would silently re-point at a different pair on a re-run).
SATURATION_PAIR = ("Camelyon", "Hibou-B", "Hibou-L")

# The pretraining-provenance exception (Sec 3.4): the TCGA-only encoder that leads both
# TCGA-containing benchmarks. Same pinning rule as SATURATION_PAIR -- prose names the model
# and its twin, so both are fixed here and every ordinal claim the paragraph makes is
# asserted below rather than assumed.
#
# `twin` is the encoder nearest to `model` in probe space once the two k-NN accuracies are
# differenced: it is the TCGA analogue of the Camelyon saturation pair, showing that the
# pooled probes rank-predict CRoMa without setting its scale. Pinned, but cross-checked to
# still *be* the nearest -- the paragraph rotted once already by asserting an ordering that
# had silently inverted (the run moved from 8 medical centres to PathoROB's 4 in-domain
# ones, and with it every TCGA probe accuracy).
PROVENANCE_PREFIX = "TcgaFourByFour"
PROVENANCE_MODEL = "Midnight-12k"
PROVENANCE_TWIN = "CONCHv1.5"
PROVENANCE_TOP_BIO_DEPTH = 5
PROVENANCE_MIN_FOLD = 2.0  # the twin's CRoMa must be at least this many times smaller
MODEL_METADATA_CSV = Path(__file__).resolve().parent.parent / "bench" / "model_metadata.csv"


_CARDINALS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
              "ten", "eleven", "twelve"]


def _cardinal(n: int) -> str:
    """Small counts spelled out; larger ones stay numerals. ``0`` becomes ``none``."""
    return _CARDINALS[n] if n < len(_CARDINALS) else str(n)


def _span(lo: float, hi: float, decimals: int = 2) -> str:
    r"""Closed interval ``$[lo, hi]$`` for a min--max span or value range in prose.

    Mirrors ``ci_bracket`` (the bracketed CI idiom the tables use) so spans and confidence
    intervals share one look, but routes each endpoint through ``bare_num`` so a ``-0.00``
    boundary (e.g. Prostate's CRoMa max) prints as ``0.00`` rather than a spurious minus.
    Correlation ranges read as ``$\rho\in$[lo, hi]`` / ``$|\rho|$ in [lo, hi]`` in prose.
    """
    return rf"$[{_apd_bare(lo, decimals)}, {_apd_bare(hi, decimals)}]$"


def _macros_for(prefix: str, df: pd.DataFrame, scale_override: str) -> tuple[list[str], str]:
    raw = df["croma"].astype(float)
    scale = scale_override if scale_override != "auto" else _detect_scale(raw)
    croma = _to_margin(raw, scale)
    ltm = df["croma_ltm_alpha"].astype(float) if "croma_ltm_alpha" in df else None

    lo, hi = float(croma.min()), float(croma.max())
    # The confounder-dominant *count* is a cross-model statistic and prose already prints it
    # over ``\...RankedNModels``, so it must be counted over the ranked panel too. It matched
    # only because the control happens to be biology-dominant on every benchmark; the day it
    # is not, "7 of 20" would silently become a numerator out of 21. Span/min/max stay over
    # the full panel, where the control is a reported floor rather than a competitor.
    from plotting.style import CONTROL_MODEL as _control

    ranked_df = df[df["model"] != _control]
    ranked_croma = croma[df["model"] != _control]
    n_negative = int((ranked_croma < 0.0).sum())
    # Each scalar: (macro-suffix, body). Add a line here to expose a new value.
    specs = [
        ("CromaSpan", _span(lo, hi)),
        ("CromaMin", _num(lo)),
        ("CromaMax", _num(hi)),
        ("CromaConfounderDominant", str(n_negative)),
        # Spelled out, for prose that must read naturally when the count is zero ("and none
        # on Tolkach-ESCA", not "and 0 on Tolkach-ESCA"). Same guarded number either way.
        ("CromaConfounderDominantWord", _cardinal(n_negative)),
        ("NModels", str(int(len(croma)))),
    ]
    if ltm is not None:
        specs.append(("CromaLtmMax", _num(float(ltm.max()))))  # best (least-negative) tail

    # Confounder-/biology-kNN balanced-accuracy range: prose cites how near-perfectly decodable
    # each signal is for *every* model (e.g. Camelyon confounder "[0.92, 1.00]"), a k-free
    # diagnostic. Computed over the ranked panel, like every other cross-model statistic here:
    # the natural-image control is the bio-kNN floor on all four tile benchmarks (and the
    # confounder floor on Camelyon/Tolkach), so including it would report the control's number
    # as the panel minimum for prose that reads "across all encoders". It is a reported floor,
    # not a competitor, so it is excluded here as from the counts, correlations and support range.
    if "confounder_knn_bacc" in ranked_df:
        cb = ranked_df["confounder_knn_bacc"].astype(float)
        specs.append(("ConfBaccRange", _span(float(cb.min()), float(cb.max()))))
    if "bio_knn_bacc" in ranked_df:
        bb = ranked_df["bio_knn_bacc"].astype(float)
        specs.append(("BioBaccRange", _span(float(bb.min()), float(bb.max()))))

    # Cross-metric rank agreement. These were once emitted by the bootstrap block, which made
    # three macros the live results section cites hostage to an optional experiment artifact:
    # when the bootstrap summary is absent the generator warns and skips, and the paper stops
    # compiling. They are plain Spearman rhos between columns of this very frame -- only their
    # CIs ever needed the bootstrap -- so they belong here, where the source always exists.
    # Cross-model correlations exclude the natural-image control (see plotstyle.CONTROL_MODEL).
    if {"ri", "mari"} <= set(df.columns):
        from plotting.style import CONTROL_MODEL
        from scipy.stats import spearmanr

        ranked = df[df["model"] != CONTROL_MODEL]
        ranked_croma = _to_margin(ranked["croma"].astype(float), scale)
        for suffix, left, right in [
            ("CromaVsRiRho", ranked_croma, ranked["ri"]),
            ("CromaVsMariRho", ranked_croma, ranked["mari"]),
            ("RiVsMariRho", ranked["ri"], ranked["mari"]),
        ]:
            specs.append((suffix, _num(float(spearmanr(left, right).statistic))))
        delta = (ranked["mari"].astype(float) - ranked["ri"].astype(float)).abs().max()
        specs.append(("MariRiMaxAbsDelta", _num(float(delta))))
        specs.append(("RankedNModels", str(int(len(ranked)))))
        # Support range over the ranked panel. The control is excluded here and not merely for
        # consistency: it has by far the thinnest structure of either kind, so few of its
        # anchors are SS-dominated and its support is an outlier (Camelyon 68% against a
        # pathology panel of 10--46%). Including it would blunt the very point the range makes.
        if "ri_undefined_frac" in ranked.columns:
            support = 1.0 - ranked["ri_undefined_frac"].astype(float)
            specs.append(("SupportRange", rf"$[{support.min() * 100:.0f}, {support.max() * 100:.0f}]\%$"))

    # Leader bundle: the model with the highest (least-negative) CRoMa. Lets prose cite the
    # leader's own scalars (name, CRoMa, biological k-NN accuracy, support) without drift.
    leader = df.loc[croma.idxmax()]
    specs.append(("BestModel", str(leader["model"])))
    specs.append(("BestCroma", _num(hi)))  # == CromaMax, named for prose readability
    if "bio_knn_bacc" in df:
        specs.append(("BestBioBacc", _num(float(leader["bio_knn_bacc"]), decimals=3)))
    if "ri_undefined_frac" in df:
        support = 1.0 - float(leader["ri_undefined_frac"])
        specs.append(("BestSupport", rf"{support * 100:.1f}\%"))

    lines = [rf"\newcommand{{\{prefix}{suffix}}}{{{body}}}" for suffix, body in specs]
    return lines, scale


def _ss_shell_macros(prefix: str, df: pd.DataFrame) -> list[str]:
    """SS-shell local-entanglement scalars (concern 6) from the typed-rank summary.

    Quotes the prevalence at the highest-CRoMa (leader) and least-entangled models to
    make the "even CRoMa leaders are locally SS-saturated" point, plus the rank
    correlations that establish "related, not redundant" rather than independence.
    """
    from scipy.stats import spearmanr

    from plotting.style import CONTROL_MODEL

    pocket_col = f"ss_pocket_frac_k{SS_POCKET_K}"
    # Ranked panel: the leader claim and both rank correlations compare models, so the
    # natural-image control -- a floor, not a competitor -- is excluded (see CONTEXT.md).
    sub = df[df["croma"].notna() & (df["model"] != CONTROL_MODEL)].copy()
    leader = sub.loc[sub["croma"].idxmax()]
    least = sub.loc[sub[pocket_col].idxmin()]
    depth_rho, _ = spearmanr(sub["croma"], sub["ss_depth_med"])
    pocket_rho, _ = spearmanr(sub["croma"], sub[pocket_col])
    specs = [
        ("SsPocketK", str(SS_POCKET_K)),
        ("SsLeaderModel", str(leader["model"])),
        ("SsLeaderPocketFrac", _pct(float(leader[pocket_col]))),
        ("SsLeastPocketFrac", _pct(float(least[pocket_col]))),
        ("SsDepthCromaRho", _num(float(depth_rho))),
        ("SsPocketCromaRho", _num(float(pocket_rho))),
    ]
    return [rf"\newcommand{{\{prefix}{suffix}}}{{{body}}}" for suffix, body in specs]


def _uncertainty_macros(prefix: str, summary: dict, df: pd.DataFrame) -> list[str]:
    """Bootstrap UQ + redundancy scalars (concerns 3 and 6).

    Exposes the cross-model rank correlations with CIs (CRoMa vs RI/MaRI, and RI vs
    MaRI to make the ``CRoMa is the least rank-redundant of the three'' point), the
    headline-leader CRoMa with its CI, the size of the top rank-tie cluster (models
    whose 95% rank interval includes rank 1), and the closest adjacent-pair win
    probability (the coin-flip that shows nearby point estimates are statistical
    ties). All sourced from the bootstrap experiment so prose cannot drift.
    """
    corr = summary["correlations"]
    leader = df.sort_values("croma", ascending=False).iloc[0]
    top_tie_n = int((df["rank_lo"] == 1).sum())
    adj = summary.get("adjacent_pair_win", [])
    closest = min(adj, key=lambda d: abs(d["p_higher_beats_lower"] - 0.5)) if adj else None
    # The rho point estimates live in _macros_for, which reads the always-present metrics.csv;
    # emitting them here too would be a duplicate \newcommand whenever this block does run.
    specs = [
        ("CromaVsRiCi", _ci(float(corr["croma_vs_ri"]["lo"]), float(corr["croma_vs_ri"]["hi"]))),
        ("CromaVsMariCi", _ci(float(corr["croma_vs_mari"]["lo"]), float(corr["croma_vs_mari"]["hi"]))),
        ("CromaLeaderModel", str(leader["model"])),
        ("CromaLeaderVal", _num(float(leader["croma"]))),
        ("CromaLeaderCi", _ci(float(leader["croma_lo"]), float(leader["croma_hi"]))),
        ("CromaTopTieN", str(top_tie_n)),
        ("CromaNBoot", str(int(summary["n_boot"]))),
    ]
    if closest is not None:
        specs.append(("CromaClosestPairWinProb", _num(float(closest["p_higher_beats_lower"]))))
    return [rf"\newcommand{{\{prefix}{suffix}}}{{{body}}}" for suffix, body in specs]


def _pvalue(pvalue: float, floor: float = 0.01) -> str:
    """Math-mode p-value; anything under ``floor`` is reported as an inequality."""
    return rf"${{<}}{floor:g}$" if pvalue < floor else f"${pvalue:.2f}$"


def _frac_negative(sample_path: str) -> float:
    """F(0): the confounder-dominant fraction, read off a model's per-sample CRoMa.

    Closed boundary (an exact zero is confounder-dominant) over the defined occurrences
    only -- the definition ``CRoMaResult.f0`` computes inside the library.
    """
    import numpy as np

    values = pd.Series(np.load(sample_path).astype(float))
    margin = _to_margin(values, _detect_scale(values))
    defined = margin[np.isfinite(margin)]
    return float((defined <= 0.0).mean())


def _probe_macros(prefix: str, df: pd.DataFrame) -> tuple[list[str], list[float]]:
    """Rank correlations between confounder decodability and each robustness statistic.

    Returns the macro lines plus the |rho| of the *pooled* targets, so ``build`` can quote a
    single across-benchmark range for them.
    """
    from plotting.style import CONTROL_MODEL
    from scipy.stats import spearmanr

    df = df[df["model"] != CONTROL_MODEL].copy()
    df["f0"] = df["croma_samples_path"].map(_frac_negative)
    probe = df["confounder_knn_bacc"].astype(float)

    lines, pooled = [], []
    for column, suffix in [*PROBE_POOLED_TARGETS, PROBE_TAIL_TARGET]:
        rho, pvalue = spearmanr(probe, df[column].astype(float))
        lines.append(rf"\newcommand{{\{prefix}Probe{suffix}Rho}}{{{_num(float(rho), decimals=2)}}}")
        if (column, suffix) == PROBE_TAIL_TARGET:
            lines.append(rf"\newcommand{{\{prefix}Probe{suffix}P}}{{{_pvalue(float(pvalue))}}}")
        else:
            pooled.append(abs(float(rho)))

    # The biological probe, for contrast: it explains far less of the CRoMa spread than the
    # confounder probe does, which is what licenses "not explained by biological accuracy".
    bio_rho, _ = spearmanr(df["bio_knn_bacc"].astype(float), df["croma"].astype(float))
    lines.append(rf"\newcommand{{\{prefix}BioProbeCromaRho}}{{{_num(float(bio_rho), decimals=2)}}}")
    lines.append(rf"\newcommand{{\{prefix}ProbeNModels}}{{{len(df)}}}")

    sat_prefix, low, high = SATURATION_PAIR
    if prefix == sat_prefix and {low, high} <= set(df["model"]):
        pair = df.set_index("model").loc[[low, high]]
        bacc_gap = float(pair["confounder_knn_bacc"].diff().iloc[-1])
        croma_gap = float(pair["croma"].diff().iloc[-1])
        lines += [
            rf"\newcommand{{\{prefix}SaturationBaccGap}}{{{_num(abs(bacc_gap), decimals=4)}}}",
            rf"\newcommand{{\{prefix}SaturationCromaGap}}{{{_num(abs(croma_gap))}}}",
            rf"\newcommand{{\{prefix}SaturationCromaLow}}{{{_num(float(pair.loc[low, 'croma']))}}}",
            rf"\newcommand{{\{prefix}SaturationCromaHigh}}{{{_num(float(pair.loc[high, 'croma']))}}}",
        ]
    return lines, pooled


def _provenance_macros(df: pd.DataFrame) -> list[str]:
    """Scalars for the pretraining-provenance paragraph (Sec 3.4), with its claims asserted.

    The paragraph argues that neither pooled probe explains ``PROVENANCE_MODEL``'s lead: it
    has the panel's *best* biology (so the margin is not bought by surrendering class
    separation) yet is *not* the least centre-decodable encoder (so it is not bought by
    invariance alone). Its declared robustness-targeted fine-tune may improve centre
    invariance without invalidating that parent-model argument. What distinguishes the
    leader is the joint position, and the twin shows the probes fix a ranking but not a scale.

    Every one of those is an ordinal claim over a frame that moves whenever the benchmark is
    re-run, so each is checked here. The previous hand-typed version asserted the exact
    opposite ("the lowest biological accuracy among the leading encoders", "the least
    decodable of all 16") and nothing caught it.
    """
    from plotting.style import CONTROL_MODEL

    d = df[df["model"] != CONTROL_MODEL].copy()
    d["croma"] = croma_as_margin(d["croma"])
    d["gap"] = d["bio_knn_bacc"].astype(float) - d["confounder_knn_bacc"].astype(float)
    d = d.set_index("model")

    model, twin = PROVENANCE_MODEL, PROVENANCE_TWIN
    metadata = pd.read_csv(MODEL_METADATA_CSV, keep_default_na=False, na_values=[])
    fine_tunes = metadata[
        (metadata["parent_model"] == model)
        & (metadata["variant_role"] == "robustness-finetune")
    ]["model"].tolist()
    if len(fine_tunes) != 1:
        raise CaptionClaimError(
            f"Sec 3.4 expects one robustness-targeted child of {model} in "
            f"{MODEL_METADATA_CSV}; got {fine_tunes}."
        )
    fine_tune = str(fine_tunes[0])
    for name in (model, twin):
        if name not in d.index:
            raise CaptionClaimError(f"Sec 3.4 names {name!r}, absent from the ranked panel.")
    others = d.drop(index=model)

    if d["croma"].idxmax() != model:
        raise CaptionClaimError(
            f"Sec 3.4 calls {model} the benchmark's CRoMa leader; it is {d['croma'].idxmax()}."
        )
    if d["bio_knn_bacc"].idxmax() != model:
        raise CaptionClaimError(
            f"Sec 3.4 claims {model} has the panel's highest biological k-NN accuracy; "
            f"{d['bio_knn_bacc'].idxmax()} does."
        )
    least_conf = d["confounder_knn_bacc"].idxmin()
    if least_conf == model:
        raise CaptionClaimError(
            f"Sec 3.4 argues {model}'s lead is not bought by centre-invariance alone, which "
            "rests on it NOT being the least centre-decodable encoder. It now is."
        )
    top_bio = d["bio_knn_bacc"].nlargest(PROVENANCE_TOP_BIO_DEPTH).index
    top_bio_least_conf = d.loc[top_bio, "confounder_knn_bacc"].idxmin()
    allowed = {model}
    if fine_tune in d.index:
        allowed.add(fine_tune)
    if top_bio_least_conf not in allowed:
        raise CaptionClaimError(
            "Sec 3.4's original 'alone resists centre decoding' guard now treats "
            f"{fine_tune} explicitly as {model}'s "
            "robustness-targeted child; neither family member is now least "
            f"centre-decodable among the top {PROVENANCE_TOP_BIO_DEPTH} biological "
            f"encoders ({top_bio_least_conf} is)."
        )
    nearest = (others["gap"] - d.loc[model, "gap"]).abs().idxmin()
    if nearest != twin:
        raise CaptionClaimError(
            f"Sec 3.4 names {twin} as {model}'s nearest neighbour in differenced-probe "
            f"space; it is now {nearest}."
        )
    fold = float(d.loc[model, "croma"] / d.loc[twin, "croma"])
    if fold < PROVENANCE_MIN_FOLD:
        raise CaptionClaimError(
            f"Sec 3.4 rests on {model} out-scoring its probe-space twin {twin} by a wide "
            f"margin; the ratio is now {fold:.2f}x, under {PROVENANCE_MIN_FOLD}x."
        )

    next_bio = others["bio_knn_bacc"].idxmax()
    specs = [
        ("Model", model),
        ("BioBacc", _num(float(d.loc[model, "bio_knn_bacc"]), decimals=3)),
        ("ConfBacc", _num(float(d.loc[model, "confounder_knn_bacc"]), decimals=3)),
        ("Gap", _num(float(d.loc[model, "gap"]), decimals=3)),
        ("LeastConfModel", str(least_conf)),
        ("LeastConfBacc", _num(float(d.loc[least_conf, "confounder_knn_bacc"]), decimals=3)),
        ("TopBioDepth", str(PROVENANCE_TOP_BIO_DEPTH)),
        ("NextBioModel", str(next_bio)),
        ("NextBioBacc", _num(float(d.loc[next_bio, "bio_knn_bacc"]), decimals=3)),
        ("NextBioConfBacc", _num(float(d.loc[next_bio, "confounder_knn_bacc"]), decimals=3)),
        ("TwinModel", twin),
        ("TwinGap", _num(float(d.loc[twin, "gap"]), decimals=3)),
        ("TwinCroma", _num(float(d.loc[twin, "croma"]))),
        ("RunnerUpModel", str(others["croma"].idxmax())),
        ("RunnerUpCroma", _num(float(others["croma"].max()))),
        ("CromaFold", f"${fold:.1f}\\times$"),
    ]
    if fine_tune in d.index:
        specs.extend(
            [
                ("FineTuneModel", fine_tune),
                ("FineTuneParent", model),
                (
                    "FineTuneBioBacc",
                    _num(float(d.loc[fine_tune, "bio_knn_bacc"]), decimals=3),
                ),
                (
                    "FineTuneConfBacc",
                    _num(float(d.loc[fine_tune, "confounder_knn_bacc"]), decimals=3),
                ),
                ("FineTuneCroma", _num(float(d.loc[fine_tune, "croma"]))),
            ]
        )
    return [rf"\newcommand{{\Provenance{suffix}}}{{{body}}}" for suffix, body in specs]


def _provenance_overlap_macros() -> list[str]:
    """The Tolkach-ESCA half of Sec 3.4: within-benchmark localisation to the TCGA cohort.

    Two claims, both asserted: ``PROVENANCE_MODEL`` has the largest TCGA boost, and it still
    leads once the TCGA cohort is dropped -- the sentence that makes the lead "amplified,
    rather than created" by pretraining overlap. The prose used to cite the *whole-benchmark*
    CRoMa (``\\TolkachCromaMax``, $0.58$) for the second claim, which is a different quantity
    from the TCGA-free subset median ($0.60$) it purported to report.

    Note the two runner-ups are different models and must not be conflated: the runner-up by
    boost (how much the TCGA cohort flatters a model) is not the runner-up by subset CRoMa
    (who else is robust once TCGA is gone).
    """
    from _overlap import rows as overlap_rows

    r = overlap_rows(include_control=False).set_index("model")
    model = PROVENANCE_MODEL
    if r["boost"].idxmax() != model:
        raise CaptionClaimError(
            f"Sec 3.4 credits {model} with the largest TCGA boost; {r['boost'].idxmax()} has it."
        )
    if r["rest"].idxmax() != model:
        raise CaptionClaimError(
            f"Sec 3.4 says {model}'s lead survives dropping the TCGA cohort; on the remaining "
            f"three it is now behind {r['rest'].idxmax()}."
        )
    boost_up = r.drop(index=model)["boost"].idxmax()
    rest_up = r.drop(index=model)["rest"].idxmax()
    specs = [
        ("TolkachRest", _num(float(r.loc[model, "rest"]))),
        ("TolkachTcga", _num(float(r.loc[model, "tcga"]))),
        ("TolkachBoost", f"${r.loc[model, 'boost']:.1f}\\times$"),
        ("TolkachBoostRunnerUpModel", str(boost_up)),
        ("TolkachBoostRunnerUp", f"${r.loc[boost_up, 'boost']:.2f}\\times$"),
        ("TolkachRestRunnerUpModel", str(rest_up)),
        ("TolkachRestRunnerUp", _num(float(r.loc[rest_up, "rest"]))),
    ]
    return [rf"\newcommand{{\Provenance{suffix}}}{{{body}}}" for suffix, body in specs]


def _apd_macros(df: pd.DataFrame) -> list[str]:
    """Per-benchmark Spearman cells for nIPD and the APD continuity analysis."""
    from _apd import Apd

    apd = Apd(corr=df)
    lines = [
        rf"\newcommand{{\ApdNModels}}{{{apd.n_models}}}",
        rf"\newcommand{{\PcabiopApdNModels}}{{{apd.n('pcabiop')}}}",
    ]
    for target, t_suffix in APD_TARGET_MACRO.items():
        for metric, m_suffix in APD_METRIC_MACRO.items():
            scoped = (df[(df["target"] == target) & (df["metric"] == metric)]
                      .set_index("scope")["spearman"])
            for scope, s_suffix in APD_SCOPE_MACRO.items():
                if scope not in scoped.index:
                    continue
                macro = f"{t_suffix}{m_suffix}{s_suffix}"
                lines.append(rf"\newcommand{{\{macro}}}{{{_apd_bare(float(scoped[scope]))}}}")
            bench = [float(scoped[b]) for b in APD_RANGE_BENCHMARKS if b in scoped.index]
            if bench:
                lo, hi = _apd_bare(min(bench)), _apd_bare(max(bench))
                lines.append(rf"\newcommand{{\{t_suffix}{m_suffix}Range}}{{$[{lo}, {hi}]$}}")
    return lines


def _pcabiop_nipd_macros(df: pd.DataFrame) -> list[str]:
    """Model-level PCaBiop/PAR values cited in the external-validation paragraph."""
    rows = df[df["dataset"] == "pcabiop"].set_index("model")
    required = {"MOOZY", "PRISM", "PRISM2", "Prov-GigaPath", "TITAN"}
    if set(rows.index) != required:
        raise CaptionClaimError(
            f"PCaBiop nIPD paragraph expects {sorted(required)}, got {sorted(rows.index)}"
        )

    model_suffix = {
        "MOOZY": "Moozy",
        "PRISM": "Prism",
        "PRISM2": "PrismTwo",
        "Prov-GigaPath": "ProvGigaPath",
        "TITAN": "Titan",
    }
    lines = []
    for model, suffix in model_suffix.items():
        for column, metric in (("nipd_id", "NipdId"), ("nipd_ood", "NipdOod")):
            lines.append(
                rf"\newcommand{{\Pcabiop{suffix}{metric}}}"
                rf"{{{float(rows.loc[model, column]) * 100:.1f}\%}}"
            )

    prov = rows.loc["Prov-GigaPath"]
    lines.extend(
        [
            rf"\newcommand{{\PcabiopProvGigaPathOodBaseline}}"
            rf"{{{float(prov['ood_baseline']):.3f}}}",
            rf"\newcommand{{\PcabiopProvGigaPathApdOod}}"
            rf"{{{float(prov['apd_ood']) * 100:.1f}\%}}",
        ]
    )
    return lines


# The benchmarks the headline "rankings are consistent across datasets" claim ranges over.
# Deliberately ONE benchmark per source cohort. `TcgaTwoByTwo` is omitted, not forgotten:
# it and `TcgaFourByFour` are two views of TCGA, and their CRoMa rankings agree at rho=0.99
# -- far above any genuinely cross-cohort pair (next highest 0.94). Averaging both in would
# let a within-cohort pair masquerade as evidence of cross-cohort transfer and lift the mean
# from 0.90 to 0.92. The claim is about transfer between cohorts, so only cross-cohort pairs
# may enter it.
CROSS_COHORT_BENCHMARKS = ["Camelyon", "Tolkach", "TcgaFourByFour"]
SLIDE_PANEL_BENCHMARK = "Panda"
EXPANDED_ENCODERS = frozenset(
    # Published spellings: the frames this set is compared against are restyled at load.
    {"Mascaret", "Phaet", "RudolfV-2", "RudolfV-2-B", "RudolfV-2-S"}
)


def _historical_rank_pareto(current: "RankPareto") -> "RankPareto":
    """Re-rank the current panel after removing exactly the five issue #133 additions."""
    from _rank_pareto import RankPareto

    missing = EXPANDED_ENCODERS - set(current.models)
    if missing:
        raise CaptionClaimError(
            f"Discussion frontier guard is missing issue #133 encoders: {sorted(missing)}"
        )

    prior_models = [model for model in current.models if model not in EXPANDED_ENCODERS]
    return RankPareto(
        medians=current.medians.loc[prior_models],
        median_ranks=current.median_ranks.loc[prior_models]
        .rank(ascending=True, method="first")
        .astype(int),
        tail_ranks=current.tail_ranks.loc[prior_models]
        .rank(ascending=True, method="first")
        .astype(int),
        exposed=frozenset(current.exposed & set(prior_models)),
        adversarial=current.adversarial,
    )


def _rank_frontier_change_macros(
    historical: "RankPareto", current: "RankPareto"
) -> list[str]:
    """Guard the Discussion claim from two explicit rank-Pareto inputs.

    This comparison is deliberately I/O-free so committed tests can exercise the exact
    claim without the git-ignored benchmark tree. Both inputs use ``RankPareto.frontier``,
    the same property that rings the generated figure.
    """
    current_frontier = frozenset(current.frontier)
    prior_frontier = frozenset(historical.frontier)
    if not current_frontier or not prior_frontier:
        raise CaptionClaimError("Discussion frontier guard produced an empty frontier.")
    if current_frontier == prior_frontier:
        raise CaptionClaimError(
            "Discussion says frontier membership changed after issue #133, but the "
            f"historical and expanded sets are both {sorted(current_frontier)}."
        )

    return [
        rf"\newcommand{{\TilePriorRankFrontierNModels}}{{{len(prior_frontier)}}}",
        rf"\newcommand{{\TileRankFrontierNModels}}{{{len(current_frontier)}}}",
        rf"\newcommand{{\TileRankFrontierModels}}{{{', '.join(sorted(current_frontier))}}}",
    ]


def _load_rank_frontier_change_macros() -> list[str]:
    """Load live benchmark ranks for the canonical local paper build."""
    from _rank_pareto import load as load_rank_pareto

    current = load_rank_pareto()
    return _rank_frontier_change_macros(_historical_rank_pareto(current), current)


def _cross_cohort_macros(croma: dict[str, pd.Series], panel_sizes: dict[str, int]) -> list[str]:
    """Panel sizes and the cross-cohort CRoMa rank-agreement cited by the abstract and intro.

    Correlations are cross-model, so the natural-image control is excluded; the tile-panel
    count reported alongside them is therefore the ranked panel, not the full 26.
    """
    import itertools

    from plotting.style import CONTROL_MODEL
    from scipy.stats import spearmanr

    present = [b for b in CROSS_COHORT_BENCHMARKS if b in croma]
    if len(present) < 2:
        print("warning: too few cohorts for cross-cohort rho, skipping", file=sys.stderr)
        return []

    shared = sorted(set.intersection(*(set(croma[b].index) for b in present)) - {CONTROL_MODEL})
    rhos = [
        float(spearmanr(croma[a][shared], croma[b][shared]).statistic)
        for a, b in itertools.combinations(present, 2)
    ]
    mean_rho = sum(rhos) / len(rhos)

    specs = [
        ("CrossCohortRhoMean", _num(mean_rho)),
        ("CrossCohortRhoRange", _span(min(rhos), max(rhos))),
        ("CrossCohortNBenchmarks", str(len(present))),
        ("TileRankedNModels", str(len(shared))),
    ]
    if SLIDE_PANEL_BENCHMARK in panel_sizes:
        specs.append(("SlideNModels", str(panel_sizes[SLIDE_PANEL_BENCHMARK])))

    print(
        f"{'cross-cohort':16s}        -> {len(specs)} macros "
        f"(n={len(shared)}, {len(rhos)} pairs over {present})",
        file=sys.stderr,
    )
    return ["% Cross-cohort CRoMa rank agreement + panel sizes (abstract, introduction)"] + [
        rf"\newcommand{{\{suffix}}}{{{body}}}" for suffix, body in specs
    ]


def build(benchmarks: list[tuple[str, str]], root: Path, scale_override: str) -> str:
    out = [
        "% AUTO-GENERATED by scripts/repro/generate_paper_values.py -- do not edit by hand.",
        "% Re-run after regenerating the faithful metrics; cite e.g. \\CamelyonCromaSpan in prose.",
    ]
    pooled_probe_rhos: list[float] = []
    croma_by_prefix: dict[str, pd.Series] = {}
    panel_sizes: dict[str, int] = {}
    for prefix, rel in benchmarks:
        path = root / rel
        if not path.exists():
            print(f"warning: missing {path}, skipping {prefix}", file=sys.stderr)
            continue
        # The runs store registry identities; every macro built from this frame is a
        # published surface, so the restyle happens at the load.
        df = published_models(pd.read_csv(path))
        lines, scale = _macros_for(prefix, df, scale_override)
        out.append(f"% {prefix}: {rel} (scale={scale})")
        out.extend(lines)
        print(f"{prefix:16s} scale={scale:6s} -> {len(lines)} macros", file=sys.stderr)

        croma_by_prefix[prefix] = pd.Series(
            _to_margin(df["croma"].astype(float), scale).to_numpy(), index=df["model"]
        )
        panel_sizes[prefix] = len(df)

        if prefix in PROBE_BENCHMARKS:
            probe_lines, pooled = _probe_macros(prefix, df)
            pooled_probe_rhos.extend(pooled)
            out.append(f"% {prefix} confounder-probe collapse (Sec 3.3)")
            out.extend(probe_lines)
            print(f"{prefix + ' probe':16s}        -> {len(probe_lines)} macros", file=sys.stderr)

        # Unlike the optional study blocks below, this one never warns-and-skips: its source
        # is the benchmark's own metrics.csv, which we have just read. A missing prefix here
        # means the manifest no longer carries the benchmark Sec 3.4 is written about.
        if prefix == PROVENANCE_PREFIX:
            prov_lines = _provenance_macros(df)
            out.append(f"% {prefix} pretraining-provenance exception (Sec 3.4)")
            out.extend(prov_lines)
            print(f"{'provenance':16s}        -> {len(prov_lines)} macros", file=sys.stderr)

    if pooled_probe_rhos:
        lo, hi = min(pooled_probe_rhos), max(pooled_probe_rhos)
        out.append("% Across-benchmark |rho| range for the pooled scores vs the confounder probe")
        out.append(rf"\newcommand{{\ProbePooledRhoRange}}{{{_span(lo, hi)}}}")

    out.extend(_cross_cohort_macros(croma_by_prefix, panel_sizes))
    out.append("% Expanded-panel frontier guard (Discussion)")
    out.extend(_load_rank_frontier_change_macros())

    ss_prefix, ss_rel = SS_SHELL_SUMMARY
    ss_path = root / ss_rel
    if ss_path.exists():
        ss_lines = _ss_shell_macros(ss_prefix, published_models(pd.read_csv(ss_path)))
        out.append(f"% {ss_prefix} SS-shell (concern 6): {ss_rel}")
        out.extend(ss_lines)
        print(f"{ss_prefix + ' SS-shell':16s}        -> {len(ss_lines)} macros", file=sys.stderr)
    else:
        print(f"warning: missing {ss_path}, skipping SS-shell macros", file=sys.stderr)

    unc_prefix, unc_json, unc_csv = UNCERTAINTY_SUMMARY
    json_path, csv_path = root / unc_json, root / unc_csv
    if json_path.exists() and csv_path.exists():
        import json as _json

        summary = _json.loads(json_path.read_text())
        unc_lines = _uncertainty_macros(unc_prefix, summary, published_models(pd.read_csv(csv_path)))
        out.append(f"% {unc_prefix} uncertainty (concerns 3 & 6): {unc_json}")
        out.extend(unc_lines)
        print(f"{unc_prefix + ' uncertainty':16s}     -> {len(unc_lines)} macros", file=sys.stderr)
    else:
        print(f"warning: missing {json_path} or {csv_path}, skipping UQ macros", file=sys.stderr)

    from _overlap import PER_SAMPLE as OVERLAP_PER_SAMPLE

    if OVERLAP_PER_SAMPLE.exists():
        ov_lines = _provenance_overlap_macros()
        out.append("% Pretraining-overlap scalars cited by Sec 3.4")
        out.extend(ov_lines)
        print(f"{'provenance/overlap':16s}    -> {len(ov_lines)} macros", file=sys.stderr)
    else:
        print(f"warning: missing {OVERLAP_PER_SAMPLE}, skipping overlap macros", file=sys.stderr)

    apd_path = root / APD_CORRELATION_CSV
    apd_study_path = root / APD_STUDY_CSV
    if apd_path.exists() and apd_study_path.exists():
        corr = pd.read_csv(apd_path)
        apd_lines = _apd_macros(corr) + _pcabiop_nipd_macros(pd.read_csv(apd_study_path))
        out.append(f"% APD downstream validation: {APD_CORRELATION_CSV}")
        out.extend(apd_lines)
        print(f"{'APD validation':16s}        -> {len(apd_lines)} macros", file=sys.stderr)
    else:
        print(
            f"warning: missing {apd_path} or {apd_study_path}, skipping APD macros",
            file=sys.stderr,
        )
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Repo root for resolving metrics paths.")
    p.add_argument("--out", type=Path, default=Path("paper/generated_values.tex"))
    p.add_argument("--scale", choices=["auto", "margin", "ratio"], default="auto",
                   help="Input CRoMa scale; 'auto' detects per benchmark (default).")
    args = p.parse_args()

    tex = build(BENCHMARKS, args.root, args.scale)
    args.out.write_text(tex)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
