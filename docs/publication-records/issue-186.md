# Issue 186 — two-model nIPD comparison verification

Checked on 2026-08-25 against the rendered comparison contract and a fresh
warning-as-error Sphinx HTML build.

| Surface | Manual check | Result |
| --- | --- | --- |
| Pointer | Select a second encoder, inspect points on both series, and swap focus. | One optional overlay is retained; point readouts name their encoder; swap transfers active emphasis without clearing the pair. |
| Keyboard | Tab through the native comparison select, overview rows, both series' points, and the swap button; operate custom SVG controls with Enter and Space. | Every target is focusable and operable; focus returns to the changed control, including the swap button after focus transfer. |
| Touch | Inspect hit geometry and the narrow-layout source at a 375 px content width. | Both series retain 44 px point targets; controls stack and the fixed-width plot scrolls instead of compressing labels. |
| Screen reader | Inspect control, model-card, series, interval, point, and live-readout names. | Active/comparison identity is explicit at every evidence surface, and each inspected point announces its model and interval. |
| Identity without color | Compare the two rendered series and model cards. | Active uses a solid line, circular points, and a solid card; comparison uses a dashed line, square points, and a dashed card. Only active signed area is hatched. |
| Light/dark | Inspect all new declarations against both Furo variable palettes. | Backgrounds and text use Furo variables; identity also uses line/point/card shape, so no claim depends on the comparison hue. |
| State transitions | Exercise self/invalid selections, model focus changes, and cohort/regime changes. | Self-selection clears comparison, invalid names leave the pair intact, selecting the comparison swaps focus, available pairs persist, and unavailable comparisons clear. |

The comparison remains a progressive enhancement of `results/nipd.json`. The static table
directives, committed JSON/CSV, and no-JavaScript fallback are unchanged.
