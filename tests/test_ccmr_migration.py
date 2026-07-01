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
