import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, TextIO


def parse_k_candidates(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("k-candidates must include at least one integer")
    return values


def safe_model_name(model: str) -> str:
    return str(model).replace("/", "_").replace(":", "_")


def k_candidates_signature(k_candidates: list[int] | tuple[int, ...]) -> str:
    uniq = sorted({int(k) for k in k_candidates})
    if not uniq:
        raise ValueError("k_candidates must include at least one integer")
    if min(uniq) <= 0:
        raise ValueError("k_candidates must be strictly positive")
    return ",".join(str(int(k)) for k in uniq)


def excluded_centers_signature(excluded_centers: list[str] | tuple[str, ...] | None) -> str:
    if not excluded_centers:
        return ""
    uniq = sorted({str(center).strip() for center in excluded_centers if str(center).strip()})
    return ",".join(uniq)


def ccrr_search_signature(
    *,
    start_k: int,
    k_growth_factor: float,
    alpha: float,
) -> str:
    return (
        f"start={int(start_k)};"
        f"growth={float(k_growth_factor):.8g};"
        f"alpha={float(alpha):.8g}"
    )


class StreamingMetricsWriter:
    def __init__(self, *, csv_path: Path, json_path: Path) -> None:
        self.csv_path = csv_path
        self.json_path = json_path
        self._csv_file: TextIO | None = None
        self._json_file: TextIO | None = None
        self._csv_writer: csv.DictWriter | None = None
        self._fieldnames: list[str] | None = None
        self._has_rows = False
        self._is_closed = False

    @property
    def has_rows(self) -> bool:
        return self._has_rows

    def write_rows(self, rows: Iterable[Mapping[str, object]]) -> int:
        count = 0
        for row in rows:
            normalized = dict(row)
            self._ensure_open(normalized)
            assert self._csv_writer is not None
            assert self._json_file is not None
            self._csv_writer.writerow(normalized)
            if self._has_rows:
                self._json_file.write(",\n")
            else:
                self._json_file.write("[\n")
            self._json_file.write(json.dumps(normalized, allow_nan=True))
            self._has_rows = True
            count += 1
        return count

    def close(self) -> None:
        if self._is_closed:
            return
        if self._json_file is not None:
            if self._has_rows:
                self._json_file.write("\n]\n")
            else:
                self._json_file.write("[]\n")
            self._json_file.close()
            self._json_file = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        self._csv_writer = None
        self._fieldnames = None
        self._is_closed = True

    def _ensure_open(self, row: Mapping[str, object]) -> None:
        if self._is_closed:
            raise RuntimeError("cannot write rows after closing StreamingMetricsWriter")
        if self._csv_writer is not None and self._json_file is not None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = list(row.keys())
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._json_file = self.json_path.open("w", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames, extrasaction="ignore")
        self._csv_writer.writeheader()


def save_metrics(rows: Iterable[Mapping[str, object]], csv_path: Path, json_path: Path) -> None:
    writer = StreamingMetricsWriter(csv_path=csv_path, json_path=json_path)
    writer.write_rows(rows)
    writer.close()
