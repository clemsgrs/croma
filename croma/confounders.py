from __future__ import annotations

import re

CANONICAL_CONFOUNDER_COLUMN = "confounder"


def normalize_confounder_column_name(value: object) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("confounder_column must be a non-empty string")
    return name


def infer_confounder_display_name(confounder_column: object) -> str:
    name = normalize_confounder_column_name(confounder_column)
    lowered = name.lower()
    if lowered == "medical_center":
        return "Medical Center"
    if lowered == CANONICAL_CONFOUNDER_COLUMN:
        return "Confounder"

    words = [part for part in re.split(r"[_\s]+", name) if part]
    if not words:
        return "Confounder"
    return " ".join(word[:1].upper() + word[1:] for word in words)
