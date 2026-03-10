from pathlib import Path

import release


def test_release_targets_croma_init() -> None:
    assert release.INIT_PY == release.ROOT / "croma" / "__init__.py"


def test_main_stages_croma_init(monkeypatch) -> None:
    commands: list[str] = []

    def fake_run(cmd: str, check: bool = True) -> str:
        del check
        commands.append(cmd)
        if cmd == "git tag":
            return ""
        if cmd == "git remote get-url origin":
            return "https://github.com/clemsgrs/MaRI.git"
        return ""

    monkeypatch.setattr(release, "run", fake_run)
    monkeypatch.setattr(release, "ensure_clean_worktree", lambda: None)
    monkeypatch.setattr(release, "get_current_version", lambda: "0.1.0")
    monkeypatch.setattr(release, "write_version", lambda version: None)
    monkeypatch.setattr(release, "create_pull_request", lambda branch, version: None)
    monkeypatch.setattr(release, "open_release_draft", lambda tag: None)
    monkeypatch.setattr("sys.argv", ["release.py", "--no-pr", "--no-draft"])

    exit_code = release.main()

    assert exit_code == 0
    assert "git add pyproject.toml croma/__init__.py" in commands
