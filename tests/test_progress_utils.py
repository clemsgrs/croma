import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import progress_utils as pu


def test_resolve_progress_mode_on_is_true() -> None:
    assert pu.resolve_progress_mode("on") is True


def test_resolve_progress_mode_off_is_false() -> None:
    assert pu.resolve_progress_mode("off") is False


def test_resolve_progress_mode_auto_uses_isatty_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Err:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(pu.sys, "stderr", _Err())
    assert pu.resolve_progress_mode("auto") is True


def test_resolve_progress_mode_auto_uses_isatty_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Err:
        @staticmethod
        def isatty() -> bool:
            return False

    monkeypatch.setattr(pu.sys, "stderr", _Err())
    assert pu.resolve_progress_mode("auto") is False
