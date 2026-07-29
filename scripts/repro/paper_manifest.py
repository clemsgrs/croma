"""The single source of truth for which benchmark run backs which paper artifact.

Every generator that writes into ``paper/sections/`` reads this module: the results-table
generator to know what to render and where, and ``generate_paper_values.py`` to know which
``metrics.csv`` each ``\\<Prefix>...`` macro family is computed from. See ADR-0010.

Before this module the same mapping lived in three hand-maintained lists that had drifted
apart -- ``generate_paper_values.py``'s ``_MEDIAN``/``_FAITHFUL`` and
``reproduce_faithful.py``'s ``CFG``. They disagreed about which protocol the paper reports
(``CFG`` was pinned to ``k-star`` while the paper reported ``median-k``), and ``CFG`` omitted
two of the five live tables outright. One list, read by everyone, is the fix.

Two invariants this module exists to enforce:

* **The model roster is never declared.** It is whatever the run's ``metrics.csv`` contains.
  A roster constant is a second source of truth for the contents of a run, and the last one
  silently pinned the paper to 16 models for three months after the panel grew to 21.
* **The protocol is a property of the benchmark, not of the invocation.** The slide panel
  stays at ``k-star`` because with four models the dataset median-k is degenerate; the tile
  panel uses ``median-k``. That reasoning belongs next to the row it governs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultsTable:
    """One benchmark run and the paper artifacts derived from it.

    ``prefix`` names the macro family in ``generated_values.tex`` (LaTeX command names take
    letters only, so digits are spelled out: ``TcgaFourByFour``). ``out_tex`` is ``None`` for
    a run whose numbers reach the paper only as macros, or whose float is assembled from
    several runs and so cannot be a whole-file render.
    """

    prefix: str
    benchmark: str
    protocol: str
    display_name: str = ""
    #: The short benchmark name used in float and table *titles* ("Camelyon", "TCGA-4x4").
    #: ``display_name`` keeps the "PathoROB " prefix so ``_cross_benchmark.short_label`` can
    #: still derive the math form ("TCGA ($4\times4$)") the m-sweep body and APD operating-point
    #: prose want; captions that head a float use this dashed form instead.
    short_name: str = ""
    label: str = ""
    out_tex: str | None = None
    model_type: str = "tile-level"
    #: Whether this benchmark's table daggers its TCGA-exposed encoders. True only for the two
    #: TCGA benchmarks, whose scored cohorts are entirely TCGA-drawn; Camelyon and Tolkach-ESCA
    #: are scored TCGA-free, so they mark nothing. The dagger set is read from model_metadata.csv.
    tcga_dagger: bool = False
    #: The column-defining table. Its caption spells the columns out; every other caption
    #: refers back to it. Exactly one entry may set this.
    primary: bool = False
    #: Render bootstrap CIs beside each CRoMa cell. Off: the CIs were deliberately dropped
    #: from the rendered tables, and the supplement they cite is commented out of supp.tex
    #: (turning this on without re-enabling that \input leaves a dangling \ref). This used
    #: to be inferred from whether ``bootstrap_uncertainty.csv`` happened to sit next to the
    #: run's ``metrics.csv``, so merely running a study rewrote all five main tables.
    with_ci: bool = False

    @property
    def run_rel(self) -> str:
        """The run directory: everything computed from this benchmark lives under it."""
        return f"output/metrics/{self.protocol}/{self.benchmark}"

    @property
    def metrics_rel(self) -> str:
        return f"{self.run_rel}/results/metrics.csv"

    @property
    def per_sample_rel(self) -> str:
        return f"{self.run_rel}/results/per_sample_metrics.csv"

    @property
    def studies_rel(self) -> str:
        """Where a *study* over this run writes. Studies read the run, so they inherit
        its protocol; a study that hard-codes one is a second source of truth."""
        return f"{self.run_rel}/studies"


#: Ordered as the paper presents them: the primary table first, then the remaining tile
#: benchmarks, then the slide panel.
TABLES: list[ResultsTable] = [
    ResultsTable(
        prefix="Camelyon",
        benchmark="pathorob-camelyon",
        protocol="median-k",
        display_name="PathoROB Camelyon",
        short_name="Camelyon",
        label="tab:main-results",
        out_tex="paper/sections/results_table.tex",
        primary=True,
    ),
    ResultsTable(
        prefix="TcgaTwoByTwo",
        benchmark="pathorob-tcga-2x2",
        protocol="median-k",
        display_name=r"PathoROB TCGA ($2\times2$)",
        short_name="TCGA-2x2",
        label="tab:main-results-tcga",
        out_tex="paper/sections/results_table_tcga.tex",
        tcga_dagger=True,
    ),
    ResultsTable(
        prefix="TcgaFourByFour",
        benchmark="pathorob-tcga-4x4",
        protocol="median-k",
        display_name=r"PathoROB TCGA ($4\times4$)",
        short_name="TCGA-4x4",
        label="tab:main-results-tcga4x4",
        out_tex="paper/sections/results_table_tcga4x4.tex",
        tcga_dagger=True,
    ),
    ResultsTable(
        prefix="Tolkach",
        benchmark="pathorob-tolkach-esca",
        protocol="median-k",
        display_name="PathoROB Tolkach-ESCA",
        short_name="Tolkach-ESCA",
        label="tab:main-results-tolkach",
        out_tex="paper/sections/results_table_tolkach.tex",
    ),
    # Prostate reaches the paper only as macros: its results table was cut from the manuscript,
    # so it owns no .tex file (out_tex is None) and drops out of rendered().
    ResultsTable(
        prefix="Prostate",
        benchmark="prostate",
        protocol="median-k",
        display_name="prostate-shift-binary (KI+RUMC)",
        short_name="prostate-shift-binary",
        label="tab:main-results-prostate",
    ),
    # The slide panel stays at k-star in every mode, matching what supp_panda.tex reports and
    # argues for: with four models the dataset median-k collapses to a tiny k (3 on the sparse
    # grid), starving RI/MaRI of support. Both panels are assembled into one two-panel float
    # (supp/table_panda_isup.tex), so neither is a whole-file render; their numbers reach the paper
    # as macros. Generating that float is the remaining hand-authored results body.
    ResultsTable(
        prefix="Panda",
        benchmark="panda",
        protocol="k-star",
        display_name="PANDA (cancer detection)",
        label="tab:main-results-panda",
        model_type="slide-level",
    ),
    ResultsTable(
        prefix="PandaIsup",
        benchmark="panda-isup",
        protocol="k-star",
        display_name="PANDA (ISUP grading)",
        label="tab:main-results-panda-isup",
        model_type="slide-level",
    ),
]


def rendered() -> list[ResultsTable]:
    """The entries that own a whole ``.tex`` file under ``paper/sections/``."""
    return [t for t in TABLES if t.out_tex]


def primary() -> ResultsTable:
    """The column-defining table every other caption refers back to."""
    (entry,) = [t for t in TABLES if t.primary]
    return entry


def by_prefix(prefix: str) -> ResultsTable:
    (entry,) = [t for t in TABLES if t.prefix == prefix]
    return entry


def by_benchmark(benchmark: str) -> ResultsTable:
    """The run a study should read for ``benchmark``, at the protocol the paper reports.

    Studies used to name the protocol themselves. Every one of them said ``k-star``, and
    when the tile panel was re-run at ``median-k`` the old runs were archived, so all five
    call sites silently pointed at a directory that no longer existed -- their generators
    skipped, and the macros they feed vanished from the paper. Ask here instead.
    """
    (entry,) = [t for t in TABLES if t.benchmark == benchmark]
    return entry
