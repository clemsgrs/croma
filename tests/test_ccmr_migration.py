import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "scripts" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import migrate_ccmr_columns as mig

# The full canonical set of legacy headline columns plus a per-radius sweep column
# and two columns that must be left untouched.
_OLD_TO_NEW = {
    "ccmr": "croma",
    "ccmr_std": "croma_std",
    "ccmr_m": "croma_m",
    "ccmr_undefined_frac": "croma_undefined_frac",
    "ccmr_k_start": "croma_k_start",
    "ccmr_k_final": "croma_k_final",
    "ccmr_retries": "croma_retries",
    "ccmr_alpha": "croma_alpha",
    "ccmr_q_alpha": "croma_q_alpha",
    "ccmr_ltm_alpha": "croma_ltm_alpha",
    "ccmr_auc": "croma_auc",
    "ccmr_min": "croma_min",
    "ccmr_delta": "croma_delta",
    "ccmr_samples_path": "croma_samples_path",
    "ccmr_search": "croma_search",
    "ccmr_m5": "croma_m5",
}
_UNTOUCHED = {"model": "resnet", "ri": 0.42, "mari": 0.77}


def _write_legacy_csv(path: Path) -> dict:
    """Write a synthetic CSV with legacy columns and known values."""
    row = {}
    row.update(_UNTOUCHED)
    known_values = {
        "ccmr": 0.123456789,
        "ccmr_std": 0.05,
        "ccmr_m": 5,
        "ccmr_undefined_frac": 0.01,
        "ccmr_k_start": 200,
        "ccmr_k_final": 400,
        "ccmr_retries": 2,
        "ccmr_alpha": 0.10,
        "ccmr_q_alpha": -0.2,
        "ccmr_ltm_alpha": -0.35,
        "ccmr_auc": 0.66,
        "ccmr_min": -0.9,
        "ccmr_delta": 0.03,
        "ccmr_samples_path": "output/samples/croma_resnet.csv",
        "ccmr_search": "start=200;growth=2;alpha=0.1",
        "ccmr_m5": 0.111,
    }
    row.update(known_values)
    df = pd.DataFrame([row, row])
    df.to_csv(path, index=False)
    return known_values


def test_migration_renames_all_ccmr_columns_and_preserves_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    known_values = _write_legacy_csv(csv_path)
    before = pd.read_csv(csv_path)

    changed = mig.migrate_csv(csv_path)
    assert changed is True

    after = pd.read_csv(csv_path)

    # Every ccmr* column renamed to its croma* counterpart; none of the old names left.
    for old, new in _OLD_TO_NEW.items():
        assert old not in after.columns
        assert new in after.columns

    # No other column changed name.
    for col in _UNTOUCHED:
        assert col in after.columns

    # Column set is exactly the renamed old columns plus the untouched ones.
    expected_cols = set(_OLD_TO_NEW.values()) | set(_UNTOUCHED)
    assert set(after.columns) == expected_cols

    # Every value preserved exactly (compare old-named series to new-named series).
    for old, new in _OLD_TO_NEW.items():
        pd.testing.assert_series_equal(
            before[old].reset_index(drop=True),
            after[new].reset_index(drop=True),
            check_names=False,
        )
    for col in _UNTOUCHED:
        pd.testing.assert_series_equal(
            before[col].reset_index(drop=True),
            after[col].reset_index(drop=True),
            check_names=False,
        )
    # spot-check a couple of exact scalar values survived
    assert after["croma"].iloc[0] == known_values["ccmr"]
    assert after["croma_samples_path"].iloc[0] == known_values["ccmr_samples_path"]


def test_migration_is_idempotent(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_legacy_csv(csv_path)

    assert mig.migrate_csv(csv_path) is True
    first = csv_path.read_bytes()

    # Second run must be a no-op and leave the file byte-for-byte identical.
    assert mig.migrate_csv(csv_path) is False
    assert csv_path.read_bytes() == first


def test_migration_leaves_data_rows_byte_for_byte(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_legacy_csv(csv_path)

    original_lines = csv_path.read_text(encoding="utf-8").splitlines()
    mig.migrate_csv(csv_path)
    migrated_lines = csv_path.read_text(encoding="utf-8").splitlines()

    # Only the header (line 0) changed; every data row is identical.
    assert original_lines[0] != migrated_lines[0]
    assert original_lines[1:] == migrated_lines[1:]


def test_migrate_directory_reports_changed_files(tmp_path: Path) -> None:
    legacy = tmp_path / "a" / "metrics.csv"
    legacy.parent.mkdir(parents=True)
    _write_legacy_csv(legacy)

    unrelated = tmp_path / "b" / "other.csv"
    unrelated.parent.mkdir(parents=True)
    pd.DataFrame([{"model": "x", "ri": 1.0}]).to_csv(unrelated, index=False)

    changed = mig.migrate_directory(tmp_path)
    assert legacy in changed
    assert unrelated not in changed  # no ccmr* columns -> untouched


# --------------------------------------------------------------------------- #
# Value-level cell migration                                                   #
# --------------------------------------------------------------------------- #


def _write_analysis_csv(path: Path) -> None:
    """Synthetic analysis CSV whose *cells* carry the metric identifier.

    Mirrors files like model_ranks / correlation_* / top_models_by_metric, where
    the metric name appears as a comparison key (``ccmr_vs_ri``) or a metric label
    (``ccmr``, ``ccmr_m5``). One column holds a filesystem path that merely
    *contains* ``ccmr`` as a substring and must be left untouched.
    """
    df = pd.DataFrame(
        [
            {"metric": "ccmr", "pair": "ccmr_vs_ri", "rho": 0.91, "src": "output/ccmr/a.csv"},
            {"metric": "ccmr_m5", "pair": "ccmr_vs_mari", "rho": 0.82, "src": "output/accmr.csv"},
            {"metric": "ri", "pair": "mari_vs_ri", "rho": 0.55, "src": "output/ri/b.csv"},
        ]
    )
    df.to_csv(path, index=False)


def test_migrate_csv_values_renames_metric_label_and_key_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "model_ranks.csv"
    _write_analysis_csv(csv_path)

    changed = mig.migrate_csv_values(csv_path)
    assert changed is True

    after = pd.read_csv(csv_path)
    # metric-label column: whole-cell token and ccmr_<suffix> both rewritten.
    assert list(after["metric"]) == ["croma", "croma_m5", "ri"]
    # comparison-key column: leading ccmr token rewritten, non-ccmr key preserved.
    assert list(after["pair"]) == ["croma_vs_ri", "croma_vs_mari", "mari_vs_ri"]


def test_migrate_csv_values_preserves_nontarget_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "model_ranks.csv"
    _write_analysis_csv(csv_path)
    before = pd.read_csv(csv_path)

    mig.migrate_csv_values(csv_path)
    after = pd.read_csv(csv_path)

    # A path that merely *contains* ccmr as a substring is untouched (not a whole
    # cell token, and does not begin with ``ccmr_``).
    assert list(after["src"]) == list(before["src"])
    # Unrelated numeric column untouched byte-for-byte.
    pd.testing.assert_series_equal(before["rho"], after["rho"], check_names=False)


def test_rename_filename_leading_token_rule() -> None:
    assert mig.rename_filename("model_specific_ccmr_subgroups.csv") == (
        "model_specific_croma_subgroups.csv"
    )
    assert mig.rename_filename("ccmr_report.csv") == "croma_report.csv"
    # No ccmr token -> unchanged.
    assert mig.rename_filename("model_ranks.csv") == "model_ranks.csv"
    # ccmr embedded inside a longer word is not a token -> unchanged.
    assert mig.rename_filename("accmrx.csv") == "accmrx.csv"


def test_migrate_filename_renames_file_on_disk(tmp_path: Path) -> None:
    src = tmp_path / "model_specific_ccmr_subgroups.csv"
    pd.DataFrame([{"model": "x", "ri": 1.0}]).to_csv(src, index=False)

    new_path = mig.migrate_filename(src)
    assert new_path == tmp_path / "model_specific_croma_subgroups.csv"
    assert new_path.exists()
    assert not src.exists()

    # A file without ccmr in the name is left alone.
    other = tmp_path / "model_ranks.csv"
    pd.DataFrame([{"model": "x", "ri": 1.0}]).to_csv(other, index=False)
    assert mig.migrate_filename(other) is None
    assert other.exists()


def test_value_and_filename_migration_is_idempotent(tmp_path: Path) -> None:
    csv_path = tmp_path / "model_specific_ccmr_subgroups.csv"
    _write_analysis_csv(csv_path)

    # First pass: values rewritten and file renamed.
    assert mig.migrate_csv_values(csv_path) is True
    new_path = mig.migrate_filename(csv_path)
    assert new_path is not None
    after_first = new_path.read_bytes()

    # Second pass on the migrated file: no-op, byte-for-byte identical.
    assert mig.migrate_csv_values(new_path) is False
    assert mig.migrate_filename(new_path) is None
    assert new_path.read_bytes() == after_first


def test_combined_header_value_filename_directory_walk(tmp_path: Path) -> None:
    # (1) header-only legacy file
    header_file = tmp_path / "metrics.csv"
    _write_legacy_csv(header_file)
    # (2) value + filename legacy file
    value_file = tmp_path / "model_specific_ccmr_subgroups.csv"
    _write_analysis_csv(value_file)

    changed = mig.migrate_directory(tmp_path)

    # Header file migrated in place (name unchanged).
    assert header_file in changed
    header_after = pd.read_csv(header_file)
    assert "croma" in header_after.columns
    assert "ccmr" not in header_after.columns

    # Value file: renamed on disk and cells rewritten.
    renamed = tmp_path / "model_specific_croma_subgroups.csv"
    assert renamed in changed
    assert not value_file.exists()
    value_after = pd.read_csv(renamed)
    assert list(value_after["metric"]) == ["croma", "croma_m5", "ri"]
    assert list(value_after["pair"]) == ["croma_vs_ri", "croma_vs_mari", "mari_vs_ri"]

    # Second walk is a complete no-op.
    assert mig.migrate_directory(tmp_path) == []


# --------------------------------------------------------------------------- #
# JSON artifact migration                                                      #
# --------------------------------------------------------------------------- #


def _legacy_json_obj() -> dict:
    """Synthetic JSON result artifact carrying legacy ``ccmr`` keys/values.

    Mirrors files like ``metrics.json`` / ``k_sweep_metrics.json``: legacy keys at
    the top level and nested (inside a list of dicts and inside nested dicts),
    whole-string metric-token values, a path value that merely *contains* ``ccmr``
    as a substring, and assorted non-string scalars that must be preserved exactly.
    """
    return {
        "ccmr": 0.123456789,
        "ccmr_std": 0.05,
        "ri": 0.42,
        "metric": "ccmr",
        "flag": "ccmr_m_sweep_gain_high",
        "note": "ri_vs_mari",
        "samples_path": "output/ccmr/samples.csv",
        "count": 400,
        "converged": True,
        "missing": None,
        "nested": {
            "ccmr_search": {"start": 200, "growth": 2},
            "ccmr_ltm_alpha": -0.35,
            "ri_delta": 0.03,
        },
        "rows": [
            {"ccmr": 0.5, "ccmr_m5": 0.111, "model": "resnet"},
            {"ccmr": -0.9, "ccmr_m5": 0.222, "model": "vit"},
        ],
    }


def test_migrate_json_value_renames_keys_at_top_and_nested() -> None:
    out = mig.migrate_json_value(_legacy_json_obj())

    # Top-level keys renamed; non-ccmr keys untouched.
    assert "croma" in out and "ccmr" not in out
    assert "croma_std" in out and "ccmr_std" not in out
    assert "ri" in out

    # Nested dict keys renamed.
    assert "croma_search" in out["nested"]
    assert "croma_ltm_alpha" in out["nested"]
    assert "ccmr_search" not in out["nested"]
    assert out["nested"]["ri_delta"] == 0.03

    # Keys inside a list of dicts renamed.
    for row in out["rows"]:
        assert "croma" in row and "ccmr" not in row
        assert "croma_m5" in row and "ccmr_m5" not in row
        assert "model" in row


def test_migrate_json_value_renames_whole_string_token_values() -> None:
    out = mig.migrate_json_value(_legacy_json_obj())
    # whole-string value that is exactly ``ccmr`` -> ``croma``
    assert out["metric"] == "croma"
    # whole-string value beginning with ``ccmr_`` -> leading token rewritten
    assert out["flag"] == "croma_m_sweep_gain_high"
    # non-ccmr string value untouched
    assert out["note"] == "ri_vs_mari"


def test_migrate_json_value_preserves_substring_and_scalars() -> None:
    src = _legacy_json_obj()
    out = mig.migrate_json_value(src)

    # A path value that merely *contains* ccmr as a substring is untouched.
    assert out["samples_path"] == "output/ccmr/samples.csv"

    # Non-string scalars preserved exactly (numeric equality after transform).
    assert out["croma"] == src["ccmr"]
    assert out["croma"] == 0.123456789
    assert out["count"] == 400 and isinstance(out["count"], int)
    assert out["converged"] is True
    assert out["missing"] is None
    assert out["nested"]["croma_ltm_alpha"] == -0.35
    assert out["rows"][1]["croma"] == -0.9

    # The source object is not mutated in place.
    assert "ccmr" in src


def test_migrate_json_value_preserves_key_order() -> None:
    obj = {"ccmr": 1, "ri": 2, "ccmr_std": 3, "mari": 4}
    out = mig.migrate_json_value(obj)
    assert list(out.keys()) == ["croma", "ri", "croma_std", "mari"]


def test_migrate_json_rewrites_file_and_preserves_numbers(tmp_path: Path) -> None:
    json_path = tmp_path / "metrics.json"
    src = _legacy_json_obj()
    json_path.write_text(json.dumps(src, indent=2), encoding="utf-8")

    changed = mig.migrate_json(json_path)
    assert changed is True

    after = json.loads(json_path.read_text(encoding="utf-8"))
    assert after["croma"] == src["ccmr"]
    assert after["metric"] == "croma"
    assert after["flag"] == "croma_m_sweep_gain_high"
    assert after["samples_path"] == "output/ccmr/samples.csv"
    assert after["nested"]["croma_ltm_alpha"] == -0.35
    assert after["rows"][0]["croma_m5"] == 0.111


def test_migrate_json_is_idempotent(tmp_path: Path) -> None:
    json_path = tmp_path / "metrics.json"
    json_path.write_text(json.dumps(_legacy_json_obj(), indent=2), encoding="utf-8")

    assert mig.migrate_json(json_path) is True
    first = json_path.read_bytes()

    # Second run is a no-op and leaves the file byte-for-byte identical.
    assert mig.migrate_json(json_path) is False
    assert json_path.read_bytes() == first


def test_migrate_json_leaves_non_ccmr_file_untouched(tmp_path: Path) -> None:
    json_path = tmp_path / "clean.json"
    obj = {"ri": 0.1, "mari": 0.2, "note": "ri_vs_mari"}
    json_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    before = json_path.read_bytes()

    assert mig.migrate_json(json_path) is False
    assert json_path.read_bytes() == before


def test_json_filename_rename() -> None:
    assert mig.rename_filename("ccmr_m_sweep_metrics.json") == (
        "croma_m_sweep_metrics.json"
    )
    # No ccmr token -> unchanged.
    assert mig.rename_filename("k_sweep_metrics.json") == "k_sweep_metrics.json"


def test_migrate_filename_renames_json_on_disk(tmp_path: Path) -> None:
    src = tmp_path / "ccmr_m_sweep_metrics.json"
    src.write_text(json.dumps({"ccmr": 1.0}), encoding="utf-8")

    new_path = mig.migrate_filename(src)
    assert new_path == tmp_path / "croma_m_sweep_metrics.json"
    assert new_path.exists() and not src.exists()

    other = tmp_path / "metrics.json"
    other.write_text(json.dumps({"ccmr": 1.0}), encoding="utf-8")
    assert mig.migrate_filename(other) is None
    assert other.exists()


def test_combined_csv_and_json_directory_walk(tmp_path: Path) -> None:
    # CSV: header + value + filename legacy files
    header_file = tmp_path / "metrics.csv"
    _write_legacy_csv(header_file)
    value_file = tmp_path / "model_specific_ccmr_subgroups.csv"
    _write_analysis_csv(value_file)
    # JSON: key/value legacy file (name unchanged) + filename legacy file
    json_file = tmp_path / "metrics.json"
    json_file.write_text(json.dumps(_legacy_json_obj(), indent=2), encoding="utf-8")
    json_named = tmp_path / "ccmr_m_sweep_metrics.json"
    json_named.write_text(json.dumps(_legacy_json_obj(), indent=2), encoding="utf-8")

    changed = mig.migrate_directory(tmp_path)

    # CSV header file migrated in place.
    assert header_file in changed
    assert "croma" in pd.read_csv(header_file).columns

    # CSV value file renamed + cells rewritten.
    csv_renamed = tmp_path / "model_specific_croma_subgroups.csv"
    assert csv_renamed in changed
    assert list(pd.read_csv(csv_renamed)["metric"]) == ["croma", "croma_m5", "ri"]

    # JSON key/value file migrated in place.
    assert json_file in changed
    after = json.loads(json_file.read_text(encoding="utf-8"))
    assert after["croma"] == 0.123456789
    assert after["metric"] == "croma"

    # JSON renamed on disk.
    json_renamed = tmp_path / "croma_m_sweep_metrics.json"
    assert json_renamed in changed
    assert not json_named.exists()

    # Second walk is a complete no-op.
    assert mig.migrate_directory(tmp_path) == []
