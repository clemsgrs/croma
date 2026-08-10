# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`.
- **Read an issue**: `gh issue view <number> --comments`.
- **List issues**: use `gh issue list` with suitable state and label filters.
- **Comment**: `gh issue comment <number> --body "..."`.
- **Apply or remove labels**: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- **Close**: `gh issue close <number> --comment "..."`.

Infer the repository from `git remote -v`; `gh` does this automatically inside the clone.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill operations

When a skill says “publish to the issue tracker,” create a GitHub issue.

When a skill says “fetch the relevant ticket,” run:

`gh issue view <number> --comments`

GitHub shares one number space across issues and pull requests. Resolve an ambiguous `#N` with `gh pr view N`, then fall back to `gh issue view N`.

## Blocking relationships

Prefer GitHub’s native issue dependencies:

1. Resolve the blocking issue’s numeric database ID.
2. Add that issue through the child issue’s `blocked_by` dependency endpoint.
3. Treat a ticket as unblocked only when all blockers are closed.

If native dependencies are unavailable, use a `## Blocked by` section containing issue references.
