import release


def test_write_version_updates_pyproject_and_runtime_init(
    monkeypatch, tmp_path
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    init_py = tmp_path / "__init__.py"
    pyproject.write_text(
        '[project]\nname = "cross-margin"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    init_py.write_text('__version__ = "0.1.0"\n', encoding="utf-8")

    monkeypatch.setattr(release, "PYPROJECT", pyproject)
    monkeypatch.setattr(release, "INIT_PY", init_py)

    release.write_version("0.1.1")

    assert 'version = "0.1.1"' in pyproject.read_text(encoding="utf-8")
    assert '__version__ = "0.1.1"' in init_py.read_text(encoding="utf-8")
