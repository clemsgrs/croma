"""Compare two ``results/`` trees by registry identity: did anything but a name change?

A published-name change (RudolfV 2 -> RudolfV-2, Mettle -> Mettle-1 (Pre-release)) has to
leave every published number where it was. A plain ``git diff`` cannot say so: the renamed
rows move, and the checksums in PROVENANCE.json change with the bytes. This joins the two
trees on the registry identity behind each printed name -- the same resolution the plot
style layer uses -- so a renamed model is one row on both sides, then reports the largest
change over every numeric field, the renames it saw, and any other difference.

Run from the repository root::

    python scripts/tools/diff_results.py main             # main's results/ vs the working tree
    python scripts/tools/diff_results.py /path/to/old/results

Exits 1 unless the two trees differ by names only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from plotting.style import registry_model_name  # noqa: E402

#: PROVENANCE.json fields every export rewrites whatever the numbers: the export date,
#: and the checksums of files whose bytes changed -- which a rename always does.
VOLATILE_PROVENANCE = ("exported", "files")

#: Columns that, with the model's registry identity, key one row of a published CSV.
ROW_KEYS = ("cohort", "regime")


@dataclass
class Comparison:
    max_abs_delta: float = 0.0
    n_numeric: int = 0
    problems: list[str] = field(default_factory=list)
    renamed: set[tuple[str, str]] = field(default_factory=set)

    @property
    def name_only(self) -> bool:
        return self.max_abs_delta == 0.0 and not self.problems

    def number(self, before: float, after: float, where: str) -> None:
        self.n_numeric += 1
        if math.isnan(before) or math.isnan(after):
            if not (math.isnan(before) and math.isnan(after)):
                self.problems.append(f"{where}: {before!r} -> {after!r}")
            return
        self.max_abs_delta = max(self.max_abs_delta, abs(after - before))

    def name(self, before: str, after: str, where: str) -> None:
        if registry_model_name(before) != registry_model_name(after):
            self.problems.append(f"{where}: model {before!r} -> {after!r}")
        elif before != after:
            self.renamed.add((before, after))


def compare(before: Mapping[str, str], after: Mapping[str, str]) -> Comparison:
    """Compare two trees given as ``{path: file text}``."""
    result = Comparison()
    for path in sorted(set(before) | set(after)):
        if path not in before or path not in after:
            result.problems.append(f"{path}: only in {'before' if path in before else 'after'}")
        elif path.endswith(".csv"):
            _compare_csv(before[path], after[path], path, result)
        elif path.endswith(".json"):
            old, new = json.loads(before[path]), json.loads(after[path])
            if path.endswith("PROVENANCE.json"):
                for key in VOLATILE_PROVENANCE:
                    old.pop(key, None)
                    new.pop(key, None)
            _compare_json(old, new, path, result)
        elif before[path] != after[path]:
            result.problems.append(f"{path}: differs and is neither CSV nor JSON")
    return result


def _keyed_rows(text: str, path: str, result: Comparison) -> tuple[list[str], dict]:
    reader = csv.DictReader(io.StringIO(text))
    rows: dict[tuple, dict[str, str]] = {}
    for row in reader:
        key = (registry_model_name(row["model"]),) + tuple(
            row[c] for c in ROW_KEYS if c in row
        )
        if key in rows:
            result.problems.append(f"{path}: two rows keyed {key}")
        rows[key] = row
    return list(reader.fieldnames or []), rows


def _compare_csv(before: str, after: str, path: str, result: Comparison) -> None:
    old_columns, old = _keyed_rows(before, path, result)
    new_columns, new = _keyed_rows(after, path, result)
    if old_columns != new_columns:
        result.problems.append(f"{path}: columns {old_columns} -> {new_columns}")
    for key in sorted(set(old) | set(new)):
        if key not in old or key not in new:
            result.problems.append(f"{path}: row {key} only {'before' if key in old else 'after'}")
            continue
        for column in old_columns:
            a, b = old[key].get(column, ""), new[key].get(column, "")
            where = f"{path}:{'/'.join(key)}:{column}"
            if column == "model":
                result.name(a, b, where)
            elif _is_number(a) and _is_number(b):
                result.number(float(a), float(b), where)
            elif a != b:
                result.problems.append(f"{where}: {a!r} -> {b!r}")


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _compare_json(before: object, after: object, where: str, result: Comparison) -> None:
    if isinstance(before, dict) and isinstance(after, dict):
        # Dicts keyed by model name (the distribution payload, PROVENANCE's per-model k)
        # are joined on registry identity, like the CSV rows.
        old = {registry_model_name(k): k for k in before}
        new = {registry_model_name(k): k for k in after}
        for identity in sorted(set(old) | set(new)):
            if identity not in old or identity not in new:
                side = "before" if identity in old else "after"
                result.problems.append(f"{where}: key {identity!r} only {side}")
                continue
            a, b = old[identity], new[identity]
            result.name(a, b, f"{where}:{identity}")
            inner = f"{where}.{identity}"
            if identity == "model" and isinstance(before[a], str) and isinstance(after[b], str):
                result.name(before[a], after[b], inner)
            else:
                _compare_json(before[a], after[b], inner, result)
    elif isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            result.problems.append(f"{where}: length {len(before)} -> {len(after)}")
        for i, (a, b) in enumerate(zip(before, after)):
            _compare_json(a, b, f"{where}[{i}]", result)
    elif _is_json_number(before) and _is_json_number(after):
        result.number(float(before), float(after), where)
    elif before != after:
        result.problems.append(f"{where}: {before!r} -> {after!r}")


def _is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_tree(source: str) -> dict[str, str]:
    """``{results/<name>: text}`` from a directory holding the files, or a git ref."""
    directory = Path(source)
    if directory.is_dir():
        return {
            f"results/{p.name}": p.read_text(encoding="utf-8")
            for p in sorted(directory.iterdir())
            if p.is_file()
        }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout

    names = git("ls-tree", "--name-only", f"{source}:results").split()
    return {f"results/{name}": git("show", f"{source}:results/{name}") for name in names}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("before", help="a git ref, or a directory holding the old results/ files")
    parser.add_argument(
        "--after", default=str(ROOT / "results"), help="the new results/ directory or git ref"
    )
    args = parser.parse_args(argv)

    before, after = read_tree(args.before), read_tree(args.after)
    result = compare(before, after)
    print(f"compared {len(set(before) | set(after))} files: {args.before} -> {args.after}")
    print(f"ignored PROVENANCE.json fields every export rewrites: {', '.join(VOLATILE_PROVENANCE)}")
    for old, new in sorted(result.renamed):
        print(f"renamed  {old} -> {new}")
    print(f"numeric fields compared: {result.n_numeric}")
    print(f"max |delta|: {result.max_abs_delta:g}")
    for problem in result.problems:
        print(f"DIFFERS  {problem}")
    print("name only: " + ("yes" if result.name_only else "NO"))
    return 0 if result.name_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
