# Issue 184 — interactive nIPD explorer verification

Checked on 2026-08-25 against a fresh warning-as-error Sphinx HTML build.

| Surface | Manual check | Result |
| --- | --- | --- |
| Pointer | Select a pathology row, the separated DINOv2-B control, and sampled trajectory points. | Selection drives the heading, scalar/headroom cards, trajectory, and live point readout. |
| Keyboard | Tab through native cohort/regime selects, overview rows, and sampled points; operate custom controls with Enter and Space. | Every target is focusable, visibly highlighted, and operable without a pointer. |
| Touch | Inspect the rendered hit geometry and narrow-layout behavior. | Model rows and sampled points have 44 px minimum hit targets; native selects are 44 px high; plots scroll horizontally rather than compressing labels. |
| Screen reader | Inspect the rendered landmark/control names and live updates. | The mount, selects, overview, models, trajectory, and every sampled point have semantic labels; point values update an `aria-live` readout; static tables remain the complete equivalent. |
| Sign without color | Inspect positive and negative portions of trajectories. | Opposite hatch directions and explicit positive/negative plot description supplement hue. |
| Light/dark | Toggle both Furo theme palettes and inspect controls, guides, intervals, fills, focus, and text. | Marks use Furo foreground/background/brand variables; the two fixed sign hues retain patterned outlines, so meaning survives either theme. |
| Narrow layout | Inspect at a 375 px content viewport. | Controls stack, metric cards become two columns, and plot canvases retain readable geometry inside a bounded horizontal scroller. |

The explorer fetches only the site copy of `results/nipd.json`. Point inspection is limited
to its committed Cramér's-V samples, means, and paired-repeat interval bounds; no probe or
trajectory computation runs in the browser.
