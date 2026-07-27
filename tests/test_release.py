import pytest

import release


def test_write_version_updates_pyproject_and_runtime_init(monkeypatch, tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    init_py = tmp_path / "__init__.py"
    pyproject.write_text('[project]\nname = "cross-margin"\nversion = "0.1.0"\n', encoding="utf-8")
    init_py.write_text('__version__ = "0.1.0"\n', encoding="utf-8")

    monkeypatch.setattr(release, "PYPROJECT", pyproject)
    monkeypatch.setattr(release, "INIT_PY", init_py)

    release.write_version("0.1.1")

    assert 'version = "0.1.1"' in pyproject.read_text(encoding="utf-8")
    assert '__version__ = "0.1.1"' in init_py.read_text(encoding="utf-8")


def test_ensure_clean_worktree_ignores_untracked_files(monkeypatch) -> None:
    commands: list[str] = []

    def fake_run(cmd: str, check: bool = True) -> str:
        commands.append(cmd)
        if cmd == "git status --porcelain --untracked-files=no":
            return ""
        return "?? scratch.txt"

    monkeypatch.setattr(release, "run", fake_run)

    release.ensure_clean_worktree()

    assert commands == ["git status --porcelain --untracked-files=no"]


def test_ensure_clean_worktree_rejects_tracked_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        release,
        "run",
        lambda cmd, check=True: " M release.py",
    )

    with pytest.raises(RuntimeError, match="Working tree is not clean"):
        release.ensure_clean_worktree()
