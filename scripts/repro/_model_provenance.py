"""Resolve benchmark exposure from structured model provenance.

Membership is a domain intersection: a model is exposed when the benchmark's scored domain
appears in either its disclosed corpus domains or its conservative institutional domains. The
legacy ``tcga_exposed`` column remains a rendered summary fact, but presentation code does not
use it as a parallel membership switch (ADR-0005).
"""

from __future__ import annotations

import pandas as pd


def _domains(cell: object) -> set[str]:
    """Parse one semicolon-separated provenance-domain cell."""
    if not isinstance(cell, str) or not cell.strip():
        return set()
    return {token.strip() for token in cell.split(";") if token.strip()}


def exposed_models_for_domain(
    metadata: pd.DataFrame, domain: str, roster: set[str]
) -> frozenset[str]:
    """Return roster members whose corpus or institution intersects ``domain``."""
    if not domain:
        return frozenset()
    marked = {
        str(row["model"])
        for _, row in metadata.iterrows()
        if domain
        in (_domains(row.get("corpus_domains")) | _domains(row.get("institutional_domains")))
    }
    return frozenset(marked & roster)
