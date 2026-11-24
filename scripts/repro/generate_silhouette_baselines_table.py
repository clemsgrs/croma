"""Build ``tab:silhouette-baselines``: CRoMa against batch-ASW and iLISI.

Reads the study ``scripts/studies/silhouette_baselines.py`` writes under
``output/studies/silhouette-baselines/`` and renders, per cohort, the two agreement
correlations with median CRoMa, the three correlations with in-domain nIPD, and the two
head-to-head differences. The caption gives the number of encoders at the iLISI floor on each
cohort and explains how they are ranked. The estimates, intervals, p and q values come
from the study, along with the implementation version, kNN, perplexity, resample and
permutation counts.

A missing study artifact fails the build; there is no skip. The caption's claims are
checked before it is printed: the natural-image control is in the study's join and out of
every correlation, the CRoMa row is the rho ``tab:apd-correlation`` reports for the same
cohort, the q values are Benjamini--Hochberg over exactly the family the caption names,
and where the floor tie moves no rank the untied rows equal the reported ones.

Usage:
  python scripts/repro/generate_silhouette_baselines_table.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "src", HERE, REPO / "scripts" / "studies"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _paper_tables import CROMA_HEADLINE_M, CaptionClaimError, bare_num  # noqa: E402
from croma.metrics.bootstrap import benjamini_hochberg  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402
import silhouette_baselines as study  # noqa: E402

TABLE_OUT = Path("paper/sections/supp/silhouette_baselines.tex")
APD_CORRELATION_CSV = REPO / "output/studies/apd/apd_correlation.csv"
_HEADER = "% Table values exported by scripts/repro/generate_silhouette_baselines_table.py."
_REMEDY = "run scripts/studies/silhouette_baselines.py (in an environment with croma[repro])"

#: How each metric is named in the table. A baseline added to the study without a name
#: here fails the render rather than printing its column key.
DISPLAY = {"croma": r"\code{CRoMa}", "batch_asw": "batch-ASW", "ilisi": "iLISI"}
#: Float-equality tolerance for the CRoMa row against tab:apd-correlation's.
_RHO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Baselines:
    """The study's three artifacts plus the downstream table it must agree with."""

    per_encoder: pd.DataFrame
    table: pd.DataFrame
    provenance: dict
    apd_correlation: pd.DataFrame

    @property
    def cohorts(self) -> list[str]:
        return list(pd.unique(self.table["dataset"]))

    def row(self, dataset: str, kind: str, metric: str) -> pd.Series:
        t = self.table
        match = t[(t["dataset"] == dataset) & (t["kind"] == kind) & (t["metric"] == metric)]
        if len(match) != 1:
            raise KeyError(f"expected one ({dataset}, {kind}, {metric}) row, found {len(match)}")
        return match.iloc[0]

    @property
    def n_ranked(self) -> int:
        counts = set(self.table["n"].astype(int))
        if len(counts) != 1:
            raise CaptionClaimError(f"cohorts were ranked over differing panels {sorted(counts)}")
        return counts.pop()

    def assert_control_excluded(self) -> None:
        """The control is in each cohort's join and outside each of its correlations."""
        for dataset in self.cohorts:
            joined = self.per_encoder[self.per_encoder["dataset"] == dataset]
            if study.CONTROL_MODEL not in set(joined["model"]):
                raise CaptionClaimError(
                    f"{study.CONTROL_MODEL} is missing from the {dataset} join, so its "
                    "exclusion cannot be verified"
                )
            ns = set(self.table.loc[self.table["dataset"] == dataset, "n"].astype(int))
            if ns != {len(joined) - 1}:
                raise CaptionClaimError(
                    f"the caption says the natural-image control is excluded, but {dataset}'s "
                    f"correlations ran over {sorted(ns)} of {len(joined)} joined encoders"
                )

    def assert_croma_matches_apd_correlation(self) -> None:
        """CRoMa's downstream row is the one tab:apd-correlation prints for the cohort."""
        apd = self.apd_correlation
        for dataset in self.cohorts:
            ours = self.row(dataset, "downstream", "croma")
            theirs = apd[
                (apd["target"] == study.TARGET)
                & (apd["metric"] == "croma")
                & (apd["scope"] == dataset)
            ]
            if len(theirs) != 1:
                raise CaptionClaimError(f"tab:apd-correlation has no CRoMa row for {dataset}")
            theirs = theirs.iloc[0]
            if int(theirs["n"]) != int(ours["n"]) or not np.isclose(
                float(theirs["spearman"]), float(ours["estimate"]), rtol=0, atol=_RHO_TOLERANCE
            ):
                raise CaptionClaimError(
                    f"{dataset}: rho(CRoMa, nIPD_ID) is {ours['estimate']:.6f} over n={ours['n']} "
                    f"here but {theirs['spearman']:.6f} over n={theirs['n']} in "
                    "tab:apd-correlation; one of the two studies is stale"
                )

    def ilisi_floor_counts(self) -> list[int]:
        """Per cohort, in table order: ranked encoders whose iLISI is on the floor."""
        counts = []
        for dataset in self.cohorts:
            joined = self.per_encoder[self.per_encoder["dataset"] == dataset]
            ranked = joined[joined["ranked"].astype(bool)]
            counts.append(int(study.at_ilisi_floor(ranked["ilisi"]).sum()))
        return counts

    def assert_untied_ilisi_rows_consistent(self) -> None:
        """Check that published and untied estimates agree wherever tying leaves ranks unchanged."""
        columns = ["estimate", "ci_lo", "ci_hi"]
        for dataset in self.cohorts:
            joined = self.per_encoder[self.per_encoder["dataset"] == dataset]
            returned = joined.loc[joined["ranked"].astype(bool), "ilisi"].astype(float)
            tied = returned.mask(study.at_ilisi_floor(returned), 0.0)
            untied = {kind: self.row(dataset, k, "ilisi") for kind, k in study.AS_RETURNED.items()}
            if not returned.rank().equals(tied.rank()):
                continue
            for kind, row in untied.items():
                reported = self.row(dataset, kind, "ilisi")
                if not np.allclose(
                    row[columns].to_numpy(dtype=float),
                    reported[columns].to_numpy(dtype=float),
                    rtol=0,
                    atol=1e-12,
                ):
                    raise CaptionClaimError(
                        f"{dataset}: tying the iLISI floor moves no rank, yet the "
                        f"{study.AS_RETURNED[kind]} row differs from the {kind} row; the study "
                        "table is stale"
                    )

    def downstream_family(self) -> pd.DataFrame:
        """The Benjamini--Hochberg family the caption names, verified against its q values."""
        family = self.table[self.table["kind"] == "downstream"]
        expected = benjamini_hochberg(family["p_value"].to_numpy(dtype=float))
        if not np.allclose(family["q_value"].to_numpy(dtype=float), expected, rtol=0, atol=1e-12):
            raise CaptionClaimError(
                "the q values are not Benjamini--Hochberg over the table's downstream rows"
            )
        others = self.table[self.table["kind"] != "downstream"]
        if others[["p_value", "q_value"]].notna().any().any():
            raise CaptionClaimError("only downstream rows may carry p and q values")
        return family


def _read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is absent; {_REMEDY}")
    return pd.read_csv(path)


def paths_under(root: Path) -> dict[str, Path]:
    """``load``'s inputs at their places under ``root`` rather than under this checkout.

    The keys are ``load``'s parameter names. ``generate_paper_values.py --root`` reads
    every other block from the tree it is given, and this one must read the same tree.
    """
    defaults = {
        "per_encoder_csv": study.PER_ENCODER_CSV,
        "correlation_csv": study.CORRELATION_CSV,
        "provenance_json": study.PROVENANCE_JSON,
        "apd_correlation_csv": APD_CORRELATION_CSV,
    }
    return {name: Path(root) / path.relative_to(REPO) for name, path in defaults.items()}


def load(
    per_encoder_csv: Path = study.PER_ENCODER_CSV,
    correlation_csv: Path = study.CORRELATION_CSV,
    provenance_json: Path = study.PROVENANCE_JSON,
    apd_correlation_csv: Path = APD_CORRELATION_CSV,
) -> Baselines:
    """Read the study's artifacts; any missing one raises, naming how to produce it."""
    per_encoder = _read_csv(per_encoder_csv)
    table = _read_csv(correlation_csv)
    if not Path(provenance_json).exists():
        raise FileNotFoundError(f"{provenance_json} is absent; {_REMEDY}")
    if not Path(apd_correlation_csv).exists():
        raise FileNotFoundError(
            f"{apd_correlation_csv} is absent; run scripts/studies/apd/apd_croma_correlation.py"
        )
    return Baselines(
        per_encoder=per_encoder,
        table=table,
        provenance=json.loads(Path(provenance_json).read_text()),
        apd_correlation=pd.read_csv(apd_correlation_csv),
    )


def _estimate(value: float) -> str:
    return f"${bare_num(float(value))}$"


def _interval(lo: float, hi: float) -> str:
    return f"$[{bare_num(float(lo))}, {bare_num(float(hi))}]$"


def _probability(value: float) -> str:
    return r"${<}0.001$" if value < 0.001 else f"${value:.3f}$"


def _thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def _cohort_label(dataset: str) -> str:
    return by_benchmark(study.DATASETS[dataset]["benchmark"]).short_name


def _cohort_block(baselines: Baselines, dataset: str) -> list[str]:
    def line(measure: str, metric: str, row: pd.Series, with_p: bool) -> str:
        cells = [measure, metric, _estimate(row["estimate"]), _interval(row["ci_lo"], row["ci_hi"])]
        cells += [_probability(row["p_value"]), _probability(row["q_value"])] if with_p else ["--", "--"]
        return " & ".join(cells) + r" \\"

    nipd = r"\code{nIPD}$_\mathrm{ID}$"
    lines = [rf"\multicolumn{{6}}{{l}}{{\textbf{{{_cohort_label(dataset)}}}}} \\"]
    for i, metric in enumerate(study.BASELINES):
        measure = r"Agreement with \code{CRoMa}" if i == 0 else ""
        lines.append(line(measure, DISPLAY[metric], baselines.row(dataset, "agreement", metric), False))
    for i, metric in enumerate((study.CROMA, *study.BASELINES)):
        measure = rf"$\rho$ with {nipd}" if i == 0 else ""
        lines.append(line(measure, DISPLAY[metric], baselines.row(dataset, "downstream", metric), True))
    for i, metric in enumerate(study.BASELINES):
        measure = r"Head-to-head $\Delta\rho$" if i == 0 else ""
        label = rf"{DISPLAY['croma']} $-$ {DISPLAY[metric]}"
        lines.append(line(measure, label, baselines.row(dataset, "head_to_head", metric), False))
    return lines


def _ilisi_floor_summary(baselines: Baselines) -> str:
    names = [_cohort_label(dataset) for dataset in baselines.cohorts]
    counts = baselines.ilisi_floor_counts()
    first = counts[0]
    parts = [f"On {names[0]}, ${first}$ of ${baselines.n_ranked}$ encoders had a median "
             "rescaled iLISI of zero to numerical precision"]
    if first > 1:
        parts.append(" and were assigned the same rank. At this floor, iLISI could not "
                     "distinguish among these encoders.")
    else:
        parts.append(".")
    rest = []
    for name, count in zip(names[1:], counts[1:]):
        if count == 0:
            rest.append(f"no encoder reached the floor on {name}")
        elif count == 1:
            rest.append(f"one did on {name}" if rest else f"one encoder reached the floor on {name}")
        else:
            rest.append(f"${count}$ encoders reached the floor on {name}")
    if rest:
        statement = " and ".join(rest)
        parts.append(" " + statement[0].upper() + statement[1:] + ".")
    return "".join(parts)


def _caption(baselines: Baselines) -> str:
    prov = baselines.provenance
    inference = prov["inference"]
    family = baselines.downstream_family()
    level = f"{float(inference['level']) * 100:g}"
    return (
        r"\caption{\textbf{\code{CRoMa} against two batch-mixing baselines from single-cell "
        r"genomics.} batch-ASW is the silhouette of medical-centre labels computed within each "
        r"biological class, $1-|s|$ averaged per class and then over classes "
        r"\cite{luecken2022scib}; iLISI is the median inverse Simpson's index of centre labels in "
        r"a perplexity-weighted neighbourhood, rescaled to $[0, 1]$ "
        r"\cite{korsunsky2019harmony,luecken2022scib}. Both are computed with "
        rf"{prov['implementation']}~v{prov['implementation_version']} "
        r"(\texttt{silhouette\_batch}, \texttt{ilisi\_knn}) with the biological class as the "
        r"label and the medical centre as the batch, on L2-normalised features, so that "
        r"Euclidean distance is a monotone function of the cosine distance \code{CRoMa} uses; "
        rf"iLISI reads the exact $k{{=}}{int(prov['n_neighbors'])}$ nearest-neighbour graph at "
        rf"perplexity~${int(prov['perplexity'])}$. Higher means better mixing for both. "
        f"{_ilisi_floor_summary(baselines)} Unlike "
        r"\code{CRoMa}, which excludes candidates from the anchor's own slide or case, both "
        r"baselines include them, as their published definitions do. Spearman $\rho$ is "
        rf"computed across the ${baselines.n_ranked}$ ranked tile encoders of each cohort "
        r"(natural-image control excluded), against median "
        rf"\code{{CRoMa}}$(m{{=}}{int(CROMA_HEADLINE_M)})$ for agreement and against in-domain "
        r"\code{nIPD} (Table~\ref{tab:apd-correlation}) downstream. Brackets are "
        rf"{level}\% percentile intervals over ${_thousands(inference['n_boot'])}$ encoder "
        r"resamples, one draw shared by every correlation in a cohort, so the head-to-head "
        r"differences $\Delta\rho = \rho(\mcode{CRoMa}, \mcode{nIPD}_\mathrm{ID}) - "
        r"\rho(\text{baseline}, \mcode{nIPD}_\mathrm{ID})$ are paired. $p$ is a two-sided "
        rf"permutation $p$ over ${_thousands(inference['n_permutations'])}$ permutations; $q$ "
        rf"applies the Benjamini--Hochberg correction across the ${len(family)}$ downstream "
        r"associations in this table, a family separate from Table~\ref{tab:apd-correlation}'s. "
        r"Agreement and head-to-head rows carry intervals only."
        "}"
    )


def build_table(baselines: Baselines) -> str:
    baselines.assert_control_excluded()
    baselines.assert_croma_matches_apd_correlation()
    baselines.assert_untied_ilisi_rows_consistent()
    caption = _caption(baselines)
    body: list[str] = []
    for dataset in baselines.cohorts:
        body += [r"\hline", *_cohort_block(baselines, dataset)]
    level = f"{float(baselines.provenance['inference']['level']) * 100:g}"
    return "\n".join(
        [
            _HEADER,
            r"\begin{table}[!htbp]",
            r"\noindent",
            r"For a tumour tile, \code{CRoMa} asks whether tumour tiles from another centre are "
            r"closer than normal tiles from its own centre. Batch-ASW measures how well centres mix"
            r" within each biological class. iLISI measures the effective number of centres "
            r"represented among a tile's neighbours, regardless of class. To test whether centre "
            r"mixing alone was as informative about susceptibility to confounded training, we "
            r"compared how strongly each score correlated with in-domain performance loss "
            r"(\code{nIPD}) across encoders.",
            r"\par\medskip",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llcccc}",
            r"\hline",
            rf"Measure & Metric & Estimate & {level}\% CI & $p$ & $q$ \\",
            *body,
            r"\hline",
            r"\end{tabular}",
            caption,
            r"\label{tab:silhouette-baselines}",
            r"\end{table}",
        ]
    ) + "\n"


def main() -> None:
    TABLE_OUT.write_text(build_table(load()))
    print(f"wrote {TABLE_OUT}")


if __name__ == "__main__":
    main()
