# Domain docs

This repository uses a single-context domain-documentation layout.

## Before exploring

Read:

- `CONTEXT.md` at the repository root.
- Relevant ADRs under `docs/adr/`.

If either is absent, proceed silently. Domain documentation is created lazily when terminology or architectural decisions are resolved.

## Vocabulary

Use terms exactly as defined in `CONTEXT.md`, including their canonical capitalization and explicit avoided synonyms.

When a needed concept is absent, reconsider whether existing vocabulary already covers it. If it is a genuine gap, record it through the domain-modeling workflow.

## Architectural decisions

Surface any conflict with an existing ADR explicitly rather than silently overriding it.
