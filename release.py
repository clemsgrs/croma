import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "croma" / "__init__.py"


def run(cmd: str, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        shell=True,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def get_current_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', text)
    if not m:
        raise RuntimeError("Could not find [project].version in pyproject.toml")
    return m.group(1).strip()


def bump_semver(version: str, level: str) -> str:
    parts = version.strip().split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"Version '{version}' is not simple semantic version 'X.Y.Z'; bump manually."
        )
    major, minor, patch = (int(parts[0]), int(parts[1]), int(parts[2]))
    if level == "major":
        major += 1
        minor = 0
        patch = 0
    elif level == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def _write_pyproject_version(version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'(?m)^(\s*version\s*=\s*")[^"]+("\s*)$',
        rf"\g<1>{version}\2",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Failed to update version in pyproject.toml")
    PYPROJECT.write_text(new_text, encoding="utf-8")


def _write_init_version(version: str) -> None:
    text = INIT_PY.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'(?m)^(__version__\s*=\s*")[^"]+("\s*)$',
        rf"\g<1>{version}\2",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Failed to update __version__ in croma/__init__.py")
    INIT_PY.write_text(new_text, encoding="utf-8")


def write_version(version: str) -> None:
    _write_pyproject_version(version)
    _write_init_version(version)


def ensure_clean_worktree() -> None:
    out = run("git status --porcelain --untracked-files=no")
    if out:
        raise RuntimeError(
            "Working tree is not clean. Commit or stash changes before running release.py."
        )


def create_pull_request(branch: str, version: str) -> None:
    run(
        f'gh pr create --title "Release {version}" '
        f'--body "This PR bumps the version to {version} and tags the release." '
        f"--base main --head {branch}"
    )


def open_release_draft(tag: str) -> None:
    repo = run("git remote get-url origin")
    match = re.search(r"github\\.com[:/](.*?)(\\.git)?$", repo)
    if not match:
        print("Could not detect GitHub repo URL; skipping release URL generation.")
        return
    repo_path = match.group(1)
    url = f"https://github.com/{repo_path}/releases/new?tag={tag}&title={tag}"
    print(f"Open the release page:\n{url}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        choices=["patch", "minor", "major"],
        default="patch",
        help="Version bump level.",
    )
    parser.add_argument(
        "--tag-prefix",
        default="v",
        help="Tag prefix, default 'v' (e.g. v0.1.1). Use empty string for bare tags.",
    )
    parser.add_argument(
        "--no-pr", action="store_true", help="Do not create a GitHub PR."
    )
    parser.add_argument(
        "--no-draft", action="store_true", help="Do not print GitHub release draft URL."
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow running even if worktree is dirty.",
    )
    args = parser.parse_args()

    try:
        if not args.allow_dirty:
            ensure_clean_worktree()

        run("git checkout main")
        run("git pull origin main")

        current = get_current_version()
        next_version = bump_semver(current, args.level)
        tag = f"{args.tag_prefix}{next_version}"
        branch = f"release-{next_version}"

        print(f"Bumping version: {current} -> {next_version}")
        write_version(next_version)

        run(f"git checkout -b {branch}")
        run("git add pyproject.toml croma/__init__.py")
        run(f'git commit -m "Bump version to {next_version}"')
        run(f"git push origin {branch}")

        existing_tags = set(run("git tag").split())
        if tag not in existing_tags:
            run(f"git tag {tag}")
        run(f"git push origin {tag}")

        if not args.no_pr:
            create_pull_request(branch, next_version)

        if not args.no_draft:
            open_release_draft(tag)

        print(f"Release flow completed for version {next_version} (tag: {tag}).")
        return 0
    except Exception as exc:
        print(f"Release flow failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
