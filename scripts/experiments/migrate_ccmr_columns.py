"""One-time, idempotent migration of legacy ``ccmr*`` result columns to ``croma*``.

The metric formerly known as ``CCMR`` (Cross-confounder Margin Ratio) was renamed
to ``CRoMa`` (Cross-confounder Robustness Margin). The rename changes **no numeric
value** -- it is a pure identifier change -- so previously computed result CSVs are
still valid; only their column *names* are stale.

This utility walks a directory, finds result CSVs, and renames every ``ccmr*``
column header to its ``croma*`` counterpart **in place, preserving every value
byte-for-byte** (no recompute). Only the header line is rewritten; the data rows
are copied through untouched, so the migration cannot perturb any stored number.

**One-time exception beyond the original header-only design.** The rename is a
pure identifier change, so besides column *headers* the same ``ccmr`` token also
survives in a handful of analysis CSVs as **cell values** (metric labels like
``ccmr``/``ccmr_m5`` and comparison keys like ``ccmr_vs_ri``) and in a few CSV
**filenames** (e.g. ``model_specific_ccmr_subgroups.csv``). To let the already
``croma``-aware paper value/table generators read these CSVs, this migration was
widened -- as an accepted, documented exception (see
``docs/adr/0001-rename-ccmr-to-croma.md``) -- to also:

- rewrite **cell values** that are exactly the ``ccmr`` token or begin with
  ``ccmr_`` (the same leading-token rule used for headers), leaving every other
  cell -- including values that merely *contain* ``ccmr`` as a substring, such as
  filesystem paths -- byte-for-byte unchanged; and
- rename affected **CSV filenames**, replacing the ``ccmr`` token in the basename
  with ``croma``.

This is still a pure identifier change: **no value is ever recomputed or
perturbed**, only the stale ``ccmr`` identifier is rewritten.

It is safe to run repeatedly: a CSV whose columns/cells are already ``croma*`` (or
which has no ``ccmr`` token at all) and whose filename carries no ``ccmr`` token is
left byte-for-byte unchanged.

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
import re
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

# Token-boundary match for the ``ccmr`` identifier inside a filename: the token
# must not be flanked by an alphanumeric character, so ``_ccmr_`` and ``ccmr_``
# match but ``accmrx`` (ccmr embedded in a longer word) does not.
_FILENAME_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])" + _OLD_PREFIX + r"(?![A-Za-z0-9])")


def _rename_leading_token(text: str) -> str:
    """Rewrite a leading ``ccmr`` identifier token to ``croma``.

    The shared rule used by both the header pass and the cell-value pass: ``text``
    is rewritten iff it is exactly ``ccmr`` or begins with ``ccmr_``; the leading
    ``ccmr`` becomes ``croma`` and the remainder is preserved. Any other string --
    including one that merely *contains* ``ccmr`` as a substring -- is returned
    unchanged.
    """
    if text == _OLD_PREFIX or text.startswith(_OLD_PREFIX + "_"):
        return _NEW_PREFIX + text[len(_OLD_PREFIX) :]
    return text


def rename_column(name: str) -> str:
    """Map a single legacy column name to its ``croma*`` counterpart.

    A column is renamed iff it is exactly ``ccmr`` or begins with ``ccmr_``; the
    leading ``ccmr`` is replaced with ``croma`` and the remainder is preserved.
    Every other column name is returned unchanged.
    """
    return _rename_leading_token(name)


def rename_cell(value: str) -> str:
    """Map a single legacy cell value to its ``croma`` counterpart.

    Uses the same conservative leading-token rule as :func:`rename_column`: a cell
    whose entire value is exactly ``ccmr`` or begins with ``ccmr_`` has its leading
    ``ccmr`` rewritten to ``croma``; every other cell (including values that merely
    contain ``ccmr`` as a substring, e.g. a filesystem path) is left unchanged.
    """
    return _rename_leading_token(value)


def rename_filename(name: str) -> str:
    """Rewrite the ``ccmr`` token in a CSV basename to ``croma``.

    The token is matched on non-alphanumeric boundaries, so ``ccmr`` appearing as a
    delimited token (``model_specific_ccmr_subgroups.csv``, ``ccmr_report.csv``) is
    rewritten while ``ccmr`` embedded in a longer word (``accmrx.csv``) and names
    with no ``ccmr`` token are returned unchanged.
    """
    return _FILENAME_TOKEN_RE.sub(_NEW_PREFIX, name)


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


def migrate_csv_values(path: Path) -> bool:
    """Rewrite legacy ``ccmr`` identifier *cell values* in ``path`` in place.

    Only data-row cells are considered (the header line is left to
    :func:`migrate_csv`). A cell is rewritten iff its entire value is exactly
    ``ccmr`` or begins with ``ccmr_`` (see :func:`rename_cell`); every other cell
    is preserved. Data lines that contain no rewritten cell are copied through
    byte-for-byte, so unrelated rows and unnecessary quoting are untouched.

    Returns ``True`` if the file was modified, ``False`` if it was already migrated
    (or contained no matching cells) and left untouched.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        lines = f.readlines()

    if len(lines) <= 1:
        return False

    changed = False
    new_lines = [lines[0]]  # header handled by migrate_csv, left untouched here
    for line in lines[1:]:
        content, terminator = _split_terminator(line)
        if content == "":
            new_lines.append(line)
            continue
        row = next(csv.reader([content]))
        new_row = [rename_cell(c) for c in row]
        if new_row == row:
            new_lines.append(line)
            continue
        buf = io.StringIO()
        csv.writer(buf, lineterminator="").writerow(new_row)
        new_lines.append(buf.getvalue() + terminator)
        changed = True

    if not changed:
        return False

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines(new_lines)
    return True


def migrate_filename(path: Path) -> Path | None:
    """Rename ``path`` so its basename's ``ccmr`` token becomes ``croma``.

    Returns the new :class:`~pathlib.Path` if the file was renamed, or ``None`` if
    the basename carried no ``ccmr`` token and the file was left in place.
    """
    new_name = rename_filename(path.name)
    if new_name == path.name:
        return None
    new_path = path.with_name(new_name)
    path.rename(new_path)
    return new_path


def migrate_directory(directory: Path) -> list[Path]:
    """Migrate every ``*.csv`` under ``directory`` recursively.

    Applies the header rename, the cell-value rename, and the filename rename to
    each CSV. Returns the list of files that were actually modified (reported under
    their final, possibly renamed, path).
    """
    changed: list[Path] = []
    for path in sorted(directory.rglob("*.csv")):
        if not path.is_file():
            continue
        touched = False
        if migrate_csv(path):
            touched = True
        if migrate_csv_values(path):
            touched = True
        new_path = migrate_filename(path)
        if new_path is not None:
            touched = True
            path = new_path
        if touched:
            changed.append(path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename legacy ccmr result identifiers to croma in place: column "
            "headers, known cell values, and affected CSV filenames."
        )
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
        print("no CSV files needed migration (already croma, or no ccmr identifier)")


if __name__ == "__main__":
    main()
