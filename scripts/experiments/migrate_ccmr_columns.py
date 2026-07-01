"""One-time, idempotent migration of legacy ``ccmr*`` result columns to ``croma*``.

The metric formerly known as ``CCMR`` (Cross-confounder Margin Ratio) was renamed
to ``CRoMa`` (Cross-confounder Robustness Margin). The rename changes **no numeric
value** -- it is a pure identifier change -- so previously computed result CSVs are
still valid; only their column *names* are stale.

This utility walks a directory, finds result CSVs, and renames every ``ccmr*``
column header to its ``croma*`` counterpart **in place, preserving every value
byte-for-byte** (no recompute). Only the header line is rewritten; the data rows
are copied through untouched, so the migration cannot perturb any stored number.

It is safe to run repeatedly: a CSV whose columns are already ``croma*`` (or which
has no ``ccmr*`` columns at all) is left byte-for-byte unchanged.

The canonical set of legacy headline columns handled is::

    ccmr, ccmr_std, ccmr_m, ccmr_undefined_frac, ccmr_k_start, ccmr_k_final,
    ccmr_retries, ccmr_alpha, ccmr_q_alpha, ccmr_ltm_alpha, ccmr_auc, ccmr_min,
    ccmr_delta, ccmr_samples_path, ccmr_search

Any additional per-radius columns of the form ``ccmr_m<k>`` (e.g. ``ccmr_m5``) are
handled by the same leading-prefix rule, so no headline or sweep column is missed.

Usage::

    python scripts/experiments/migrate_ccmr_columns.py <directory>
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

# Canonical legacy headline columns (documentation / reference). The renaming rule
# below is prefix-based so it also covers per-radius ``ccmr_m<k>`` columns.
LEGACY_HEADLINE_COLUMNS: tuple[str, ...] = (
    "ccmr",
    "ccmr_std",
    "ccmr_m",
    "ccmr_undefined_frac",
    "ccmr_k_start",
    "ccmr_k_final",
    "ccmr_retries",
    "ccmr_alpha",
    "ccmr_q_alpha",
    "ccmr_ltm_alpha",
    "ccmr_auc",
    "ccmr_min",
    "ccmr_delta",
    "ccmr_samples_path",
    "ccmr_search",
)

_OLD_PREFIX = "ccmr"
_NEW_PREFIX = "croma"


def rename_column(name: str) -> str:
    """Map a single legacy column name to its ``croma*`` counterpart.

    A column is renamed iff it is exactly ``ccmr`` or begins with ``ccmr_``; the
    leading ``ccmr`` is replaced with ``croma`` and the remainder is preserved.
    Every other column name is returned unchanged.
    """
    if name == _OLD_PREFIX or name.startswith(_OLD_PREFIX + "_"):
        return _NEW_PREFIX + name[len(_OLD_PREFIX) :]
    return name


def _split_terminator(line: str) -> tuple[str, str]:
    """Split a header line into (content, line-terminator)."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def migrate_csv(path: Path) -> bool:
    """Rename legacy ``ccmr*`` header columns in ``path`` in place.

    Only the header line is rewritten; all data rows are preserved byte-for-byte.
    Returns ``True`` if the file was modified, ``False`` if it was already migrated
    (or contained no ``ccmr*`` columns) and left untouched.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        header_line = f.readline()
        rest = f.read()

    if not header_line:
        return False

    content, terminator = _split_terminator(header_line)
    header = next(csv.reader([content]))
    new_header = [rename_column(c) for c in header]
    if new_header == header:
        return False

    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(new_header)
    new_header_line = buf.getvalue() + terminator

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(new_header_line)
        f.write(rest)
    return True


def migrate_directory(directory: Path) -> list[Path]:
    """Migrate every ``*.csv`` under ``directory`` recursively.

    Returns the list of files that were actually modified.
    """
    changed: list[Path] = []
    for path in sorted(directory.rglob("*.csv")):
        if path.is_file() and migrate_csv(path):
            changed.append(path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename legacy ccmr* result columns to croma* in place."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to walk for result CSVs (recursively).",
    )
    args = parser.parse_args()

    directory = args.directory
    if not directory.is_dir():
        raise SystemExit(f"not a directory: {directory}")

    changed = migrate_directory(directory)
    if changed:
        print(f"migrated {len(changed)} CSV file(s):")
        for path in changed:
            print(f"  {path}")
    else:
        print("no CSV files needed migration (already croma* or no ccmr* columns)")


if __name__ == "__main__":
    main()
