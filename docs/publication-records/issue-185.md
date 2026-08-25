# Issue 185 — CRoMa–nIPD association verification

Checked on 2026-08-25 against a fresh warning-as-error Sphinx HTML build, the committed
publication payload, and the dependency-free DOM harness.

| Surface | Check | Result |
| --- | --- | --- |
| Synchronization | Select models from the overview and scatter with pointer activation, Enter, and Space. | Both directions update the trajectory heading and the scatter's labelled active point; focus returns to the control that initiated the change. |
| Scope and statistics | Change every cohort and ID/OOD regime and compare the rendered inputs with the committed payload/manuscript analysis. | Each view uses only its active context. All six tile-panel coefficients and both descriptive PCaBiop coefficients match; fitted trends use the same ranked pathology rows and observed CRoMa range. |
| Natural-image reference | Inspect DINOv2-B in all tile contexts. | A labelled diamond keeps it visually separate and selectable, while exact ranked-input tests exclude it from rho and the fitted trend. |
| Pointer/touch | Inspect point hit geometry and activation. | Every point has a transparent 44 px target and `touch-action: manipulation`; activation opens the same model in the trajectory explorer. |
| Keyboard | Traverse and operate scatter points. | Every point is a semantic button with `tabindex=0`, visible focus, and Enter/Space activation. |
| Screen reader | Inspect plot/point descriptions and live readout. | The plot names both axes and its active context; every point announces model/control identity and exact CRoMa/nIPD values; selection updates an `aria-live` readout. |
| Sign and identity without color | Inspect active, pathology, and control marks. | Active points gain a heavy outline and visible label; the control is a labelled diamond rather than a color-only variant. |
| Light/dark | Inspect the association style rules against both Furo palettes. | Guides, trend, marks, labels, focus, borders, and backgrounds use Furo theme variables; no fixed background assumes one theme. |
| Narrow layout | Inspect the 42 rem breakpoint and 375 px geometry. | The plot retains readable geometry inside the explorer's horizontal scroller; point targets and labels are not compressed. |
| JavaScript disabled | Inspect the built HTML without executing scripts. | Eight static cohort/regime tables publish median CRoMa, nIPD, ranked-panel rho, roster size, and the PCaBiop descriptive label; JSON and CSV downloads expose the exact source values. |

The association is presented as model-level, within-cohort evidence. Nearby prose explicitly
rules out a causal interpretation and sample-level pairing because representation metrics
and downstream probes may use different evaluation samples.

## Automated verification

- TDD red evidence: the association state/DOM tests initially failed because the view and
  scatter were absent; the publication-contract test then failed because derived
  association metadata was absent from the committed float basis.
- `node --test tests/js/test_nipd_explorer.cjs` — 20 passed after rebasing onto the
  two-model comparison explorer.
- Focused publication, association, page, explorer, and results-export tests — 67 passed,
  2 skipped.
- `PYTHONPATH=src pytest -q` — 858 passed, 15 skipped.
- `python scripts/tools/export_nipd.py --check` — passed.
- Canonical study re-export and byte comparison of `nipd.json` and `nipd.csv` — passed.
- `PYTHONPATH=src python -m sphinx -W -E -b html docs <fresh-directory>` — passed
  without warnings.
- Black, JavaScript syntax, and `git diff --check` — passed.
