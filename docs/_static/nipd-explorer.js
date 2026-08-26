/* Cohort-first browser for the committed nIPD publication artifact. */
(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && root.document) {
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", function () {
        api.boot(root.document);
      });
    } else {
      api.boot(root.document);
    }
  }
})(typeof window === "undefined" ? null : window, function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var OVERVIEW_WIDTH = 720;
  var OVERVIEW_LEFT = 154;
  var OVERVIEW_RIGHT = 200;
  var OVERVIEW_ROW = 44;
  var OVERVIEW_NIPD_COLUMN = 578;
  var OVERVIEW_END_COLUMN = OVERVIEW_WIDTH - 2;
  var DETAIL_WIDTH = 720;
  var DETAIL_HEIGHT = 360;
  var DETAIL_PAD = { top: 26, right: 24, bottom: 55, left: 62 };
  var ASSOCIATION_WIDTH = 720;
  var ASSOCIATION_HEIGHT = 410;
  var ASSOCIATION_PAD = { top: 34, right: 30, bottom: 58, left: 64 };
  // A normalized change of -1 is chance level: the probe has lost its whole
  // above-chance margin. Curves ending at or below this are reported as a collapse.
  var COLLAPSE = -0.9;

  function createView(payload) {
    if (!payload || payload.schema_version !== 2 || !Array.isArray(payload.cohorts)) {
      throw new Error("Unsupported nIPD publication payload");
    }
    var cohorts = payload.cohorts;
    var state = { cohort: cohorts[0].slug, regime: "id", model: null, comparison: null };
    state.model = pathologyModels(cohorts[0], "id")[0].model;

    function cohort() {
      return cohorts.find(function (candidate) {
        return candidate.slug === state.cohort;
      });
    }

    function model() {
      return cohort().models.find(function (candidate) {
        return candidate.model === state.model;
      });
    }

    function setContext(slug, regime) {
      var destination = cohorts.find(function (candidate) {
        return candidate.slug === slug;
      });
      if (!destination) throw new Error("Unknown nIPD cohort: " + slug);
      if (regime !== "id" && regime !== "ood") throw new Error("Unknown nIPD regime: " + regime);
      state.cohort = slug;
      state.regime = regime;
      if (!destination.models.some(function (candidate) { return candidate.model === state.model; })) {
        state.model = pathologyModels(destination, regime)[0].model;
      }
      if (!destination.models.some(function (candidate) {
        return candidate.model === state.comparison;
      }) || state.comparison === state.model) {
        state.comparison = null;
      }
    }

    function select(name) {
      if (!cohort().models.some(function (candidate) { return candidate.model === name; })) {
        throw new Error("Unknown model in active nIPD context: " + name);
      }
      if (name === state.comparison) {
        swapFocus();
      } else {
        state.model = name;
      }
    }

    function compare(name) {
      if (name === null || name === "" || name === state.model) {
        state.comparison = null;
        return;
      }
      if (!cohort().models.some(function (candidate) { return candidate.model === name; })) {
        throw new Error("Unknown comparison model in active nIPD context: " + name);
      }
      state.comparison = name;
    }

    function swapFocus() {
      if (state.comparison === null) return;
      var previous = state.model;
      state.model = state.comparison;
      state.comparison = previous;
    }

    function inspectPoint(index, name) {
      var activeCohort = cohort();
      var inspectedModel = name ? activeCohort.models.find(function (candidate) {
        return candidate.model === name;
      }) : model();
      if (!inspectedModel) {
        throw new Error("Unknown model in active nIPD context: " + name);
      }
      var result = inspectedModel.regimes[state.regime];
      if (!Number.isInteger(index) || index < 0 || index >= activeCohort.cramers_v.length) {
        throw new Error("Unknown sampled point: " + index);
      }
      return {
        model: inspectedModel.model,
        regime: state.regime.toUpperCase(),
        cramersV: activeCohort.cramers_v[index],
        mean: result.mean_normalized_trajectory[index],
        low: result.ci95_low[index],
        high: result.ci95_high[index],
        interval: payload.provenance.interval,
      };
    }

    function snapshot() {
      var activeCohort = cohort();
      var activeModel = model();
      var result = activeModel.regimes[state.regime];
      return {
        cohorts: cohorts.map(function (candidate) {
          return { slug: candidate.slug, label: candidate.label };
        }),
        cohort: activeCohort,
        models: activeCohort.models,
        regime: state.regime,
        selected: activeModel,
        comparison: activeCohort.models.find(function (candidate) {
          return candidate.model === state.comparison;
        }) || null,
        pathology: pathologyModels(activeCohort, state.regime),
        control: activeCohort.models.filter(function (candidate) { return candidate.is_control; }),
        association: association(activeCohort, state.regime),
        yDomain: contextYDomain(activeCohort, state.regime),
        readout: {
          nipd: result.nipd,
          normalizedChangeAtMaxV: endOfRange(result),
          collapsesToChance: collapses(result),
          baselineBalancedAccuracy: result.baseline_balanced_accuracy,
        },
      };
    }

    return {
      compare: compare,
      inspectPoint: inspectPoint,
      select: select,
      setContext: setContext,
      snapshot: snapshot,
      swapFocus: swapFocus,
    };
  }

  function association(cohort, regime) {
    var metadata = cohort.association;
    var result = metadata.regimes[regime];
    return {
      pathology: cohort.models.filter(function (model) { return model.ranked; })
        .map(function (model) { return associationPoint(model, regime); }),
      control: cohort.models.filter(function (model) { return model.is_control; })
        .map(function (model) { return associationPoint(model, regime); }),
      n: result.n,
      rho: result.spearman_rho,
      trend: result.trend,
      descriptive: metadata.descriptive,
    };
  }

  function associationPoint(model, regime) {
    return {
      model: model.model,
      x: model.croma_median_m5,
      y: model.regimes[regime].nipd,
      isControl: model.is_control,
    };
  }

  /* Ranked on the endpoint rather than the pooled area. The signed area lets an early
     gain pay for a late collapse, so a curve that ends at chance can outrank one that
     never moved; the endpoint cannot cancel with itself. */
  function pathologyModels(cohort, regime) {
    return cohort.models
      .filter(function (model) { return model.ranked && !model.is_control; })
      .slice()
      .sort(function (left, right) {
        return endOfRange(right.regimes[regime]) - endOfRange(left.regimes[regime]) ||
          left.model.localeCompare(right.model);
      });
  }

  function contextYDomain(cohort, regime) {
    var values = [0];
    cohort.models.forEach(function (model) {
      var result = model.regimes[regime];
      values = values.concat(
        result.mean_normalized_trajectory,
        result.ci95_low,
        result.ci95_high
      );
    });
    var low = Math.min.apply(null, values);
    var high = Math.max.apply(null, values);
    var padding = Math.max((high - low) * 0.06, 0.01);
    return [low - padding, high + padding];
  }

  function splitSignSegments(xs, ys) {
    if (xs.length !== ys.length || xs.length < 2) throw new Error("Trajectory shape mismatch");
    var segments = [];
    var current = null;

    function append(sign, point) {
      if (!current || current.sign !== sign) {
        current = { sign: sign, points: [] };
        segments.push(current);
      }
      var previous = current.points[current.points.length - 1];
      if (!previous || previous[0] !== point[0] || previous[1] !== point[1]) {
        current.points.push(point);
      }
    }

    for (var index = 0; index < xs.length - 1; index += 1) {
      var left = [xs[index], ys[index]];
      var right = [xs[index + 1], ys[index + 1]];
      var leftSign = left[1] < 0 ? "negative" : "positive";
      var rightSign = right[1] < 0 ? "negative" : "positive";
      if (left[1] !== 0 && right[1] !== 0 && leftSign !== rightSign) {
        var fraction = -left[1] / (right[1] - left[1]);
        var crossing = [left[0] + fraction * (right[0] - left[0]), 0];
        append(leftSign, left);
        append(leftSign, crossing);
        current = null;
        append(rightSign, crossing);
        append(rightSign, right);
      } else {
        var sign = left[1] === 0 ? rightSign : leftSign;
        append(sign, left);
        append(sign, right);
      }
    }
    return segments.filter(function (segment) { return segment.points.length > 1; });
  }

  function boot(document) {
    var mounts = document.querySelectorAll(".croma-nipd-explorer");
    if (!mounts.length) return;
    mounts.forEach(function (mount) {
      fetch(new URL(mount.dataset.payload, document.baseURI).href)
        .then(function (response) {
          if (!response.ok) throw new Error(response.status + " " + response.statusText);
          return response.json();
        })
        .then(function (payload) { renderExplorer(mount, createView(payload)); })
        .catch(function (error) {
          mount.textContent = "The committed nIPD evidence could not be loaded (" + error.message + ").";
        });
    });
  }

  function renderExplorer(mount, view, restoreFocus) {
    var state = view.snapshot();
    mount.textContent = "";

    var controls = htmlEl("div", "croma-nipd-controls");
    var cohortSelect = htmlEl("select");
    cohortSelect.setAttribute("aria-label", "Cohort");
    state.cohorts.forEach(function (cohort) {
      var option = htmlEl("option");
      option.value = cohort.slug;
      option.textContent = cohort.label;
      option.selected = cohort.slug === state.cohort.slug;
      cohortSelect.appendChild(option);
    });
    var regimeSelect = htmlEl("select");
    regimeSelect.setAttribute("aria-label", "Evaluation regime");
    [["id", "ID — shortcut susceptibility"], ["ood", "OOD — susceptibility + transfer effects"]]
      .forEach(function (entry) {
        var option = htmlEl("option");
        option.value = entry[0];
        option.textContent = entry[1];
        option.selected = entry[0] === state.regime;
        regimeSelect.appendChild(option);
      });
    var comparisonSelect = htmlEl("select");
    comparisonSelect.setAttribute("aria-label", "Comparison model");
    var noComparison = htmlEl("option");
    noComparison.value = "";
    noComparison.textContent = "No comparison";
    noComparison.selected = state.comparison === null;
    comparisonSelect.appendChild(noComparison);
    state.models.filter(function (candidate) {
      return candidate.model !== state.selected.model;
    }).forEach(function (candidate) {
      var option = htmlEl("option");
      option.value = candidate.model;
      option.textContent = candidate.model + (candidate.is_control ? " — natural-image control" : "");
      option.selected = state.comparison && candidate.model === state.comparison.model;
      comparisonSelect.appendChild(option);
    });
    controls.appendChild(labelled("Cohort", cohortSelect));
    controls.appendChild(labelled("Evaluation regime", regimeSelect));
    controls.appendChild(labelled("Compare with (optional)", comparisonSelect));
    mount.appendChild(controls);

    var context = htmlEl("p", "croma-nipd-context");
    context.textContent = state.regime === "id"
      ? "ID isolates the effect of increasing training-set confounding."
      : "OOD also reflects transfer effects from acquisition groups unseen during training.";
    mount.appendChild(context);

    var overviewHeading = htmlEl("h3", "croma-nipd-heading");
    overviewHeading.textContent = state.cohort.label + " · " + state.regime.toUpperCase() + " nIPD";
    mount.appendChild(overviewHeading);
    var direction = htmlEl("p", "croma-nipd-direction");
    direction.innerHTML = "<span aria-hidden=\"true\">←</span> more degradation · " +
      "zero net change · <strong>less degradation</strong> <span aria-hidden=\"true\">→</span>";
    mount.appendChild(direction);
    mount.appendChild(renderOverview(state, function (name) {
      view.select(name);
      renderExplorer(mount, view, "model");
    }));

    mount.appendChild(renderDetail(view));
    mount.appendChild(renderAssociation(state, function (name) {
      view.select(name);
      renderExplorer(mount, view, "association");
    }));

    cohortSelect.addEventListener("change", function () {
      view.setContext(cohortSelect.value, regimeSelect.value);
      renderExplorer(mount, view, "cohort");
    });
    regimeSelect.addEventListener("change", function () {
      view.setContext(cohortSelect.value, regimeSelect.value);
      renderExplorer(mount, view, "regime");
    });
    comparisonSelect.addEventListener("change", function () {
      view.compare(comparisonSelect.value);
      renderExplorer(mount, view, "comparison");
    });
    if (restoreFocus === "model") mount.querySelector(".croma-nipd-model.is-selected").focus();
    if (restoreFocus === "association") {
      mount.querySelector(".croma-nipd-association-point.is-selected").focus();
    }
    if (restoreFocus === "cohort") cohortSelect.focus();
    if (restoreFocus === "regime") regimeSelect.focus();
    if (restoreFocus === "comparison") comparisonSelect.focus();
    if (restoreFocus === "swap") mount.querySelector(".croma-nipd-swap").focus();
  }

  function renderAssociation(state, choose) {
    var associationState = state.association;
    var points = associationState.pathology.concat(associationState.control);
    var xValues = points.map(function (point) { return point.x; }).concat([0]);
    var yValues = points.map(function (point) { return point.y; }).concat([0]);
    if (associationState.trend) {
      yValues = yValues.concat([
        associationState.trend.intercept + associationState.trend.slope *
        associationState.trend.x_min,
        associationState.trend.intercept + associationState.trend.slope *
          associationState.trend.x_max,
      ]);
    }
    var xDomain = paddedDomain(xValues);
    var yDomain = paddedDomain(yValues);
    var x = linear(xDomain[0], xDomain[1], ASSOCIATION_PAD.left,
      ASSOCIATION_WIDTH - ASSOCIATION_PAD.right);
    var y = linear(yDomain[0], yDomain[1], ASSOCIATION_HEIGHT - ASSOCIATION_PAD.bottom,
      ASSOCIATION_PAD.top);
    var section = htmlEl("section", "croma-nipd-association");
    section.setAttribute("aria-labelledby", "croma-nipd-association-heading");
    var heading = htmlEl("h3", "croma-nipd-heading");
    heading.id = "croma-nipd-association-heading";
    heading.textContent = "CRoMa and downstream susceptibility";
    section.appendChild(heading);
    var annotation = htmlEl("p", "croma-nipd-association-stat");
    annotation.textContent = "Spearman ρ = " + decimal(associationState.rho) + "; n = " +
      associationState.n + " ranked pathology encoders" +
      (associationState.descriptive ? "; descriptive (fitted trend omitted)." : ".");
    section.appendChild(annotation);
    var descriptionId = "croma-nipd-association-description";
    var description = htmlEl("p", "croma-sr-only croma-nipd-association-description");
    description.id = descriptionId;
    description.textContent = "Model-level scatter of median CRoMa at m equals 5 against nIPD for " +
      state.cohort.label + " " + state.regime.toUpperCase() + ". Statistics use ranked pathology " +
      "encoders only. DINOv2-B is a separate natural-image reference when present. " +
      (associationState.trend
        ? "The least-squares trend over ranked pathology encoders has slope " +
          decimal(associationState.trend.slope) + "."
        : "The n=5 association is descriptive and has no fitted trend.");
    section.appendChild(description);
    var inspection = htmlEl("p", "croma-nipd-association-inspection");
    inspection.setAttribute("aria-live", "polite");
    var svg = svgEl("svg", {
      class: "croma-nipd-association-plot",
      viewBox: "0 0 " + ASSOCIATION_WIDTH + " " + ASSOCIATION_HEIGHT,
      role: "group",
      "aria-label": "CRoMa and nIPD association",
      "aria-describedby": descriptionId,
    });
    svg.appendChild(svgEl("line", {
      x1: x(0), x2: x(0), y1: ASSOCIATION_PAD.top,
      y2: ASSOCIATION_HEIGHT - ASSOCIATION_PAD.bottom, class: "croma-nipd-zero",
    }));
    svg.appendChild(svgEl("line", {
      x1: ASSOCIATION_PAD.left, x2: ASSOCIATION_WIDTH - ASSOCIATION_PAD.right,
      y1: y(0), y2: y(0), class: "croma-nipd-zero",
    }));
    if (associationState.trend) {
      svg.appendChild(svgEl("line", {
        x1: x(associationState.trend.x_min),
        y1: y(associationState.trend.intercept + associationState.trend.slope *
          associationState.trend.x_min),
        x2: x(associationState.trend.x_max),
        y2: y(associationState.trend.intercept + associationState.trend.slope *
          associationState.trend.x_max),
        class: "croma-nipd-association-trend",
        "aria-label": "Least-squares trend over ranked pathology encoders",
      }));
    }
    points.forEach(function (point) {
      var selected = point.model === state.selected.model;
      var group = svgEl("g", {
        class: "croma-nipd-association-point" + (selected ? " is-selected" : "") +
          (point.isControl ? " is-control" : ""),
        role: "button",
        tabindex: "0",
        "aria-pressed": selected ? "true" : "false",
        "aria-label": associationInspection(point),
      });
      group.appendChild(svgEl("circle", {
        cx: x(point.x), cy: y(point.y), r: 22, class: "croma-nipd-association-hit",
      }));
      if (point.isControl) {
        group.appendChild(svgEl("polygon", {
          points: [
            [x(point.x), y(point.y) - 7], [x(point.x) + 7, y(point.y)],
            [x(point.x), y(point.y) + 7], [x(point.x) - 7, y(point.y)],
          ].map(function (pair) { return pair.join(","); }).join(" "),
          class: "croma-nipd-association-mark",
        }));
      } else {
        group.appendChild(svgEl("circle", {
          cx: x(point.x), cy: y(point.y), r: 5, class: "croma-nipd-association-mark",
        }));
      }
      // Every point is named on hover or focus; the selected model and the control
      // keep their name on permanently.
      group.appendChild(svgText(x(point.x) + 9, y(point.y) - 9, point.model,
        "croma-nipd-association-label" + (selected || point.isControl ? " is-pinned" : "")));
      var title = svgEl("title");
      title.textContent = point.model;
      group.appendChild(title);
      activate(group, function () { choose(point.model); });
      function name() { inspection.textContent = associationInspection(point); }
      group.addEventListener("click", name);
      group.addEventListener("focus", name);
      group.addEventListener("mouseenter", name);
      group.addEventListener("mouseleave", function () {
        var active = points.find(function (candidate) {
          return candidate.model === state.selected.model;
        });
        if (active) inspection.textContent = associationInspection(active);
      });
      svg.appendChild(group);
      if (selected) inspection.textContent = associationInspection(point);
    });
    svg.appendChild(svgText(ASSOCIATION_WIDTH / 2, ASSOCIATION_HEIGHT - 12,
      "Median CRoMa (m=5)", "croma-nipd-axis-title"));
    var yTitle = svgText(16, ASSOCIATION_HEIGHT / 2, "nIPD", "croma-nipd-axis-title");
    yTitle.setAttribute("transform", "rotate(-90 16 " + ASSOCIATION_HEIGHT / 2 + ")");
    svg.appendChild(yTitle);
    [xDomain[0], 0, xDomain[1]].forEach(function (value) {
      svg.appendChild(svgText(x(value), ASSOCIATION_HEIGHT - 36, decimal(value),
        "croma-nipd-axis-label is-middle"));
    });
    [yDomain[0], 0, yDomain[1]].forEach(function (value) {
      svg.appendChild(svgText(ASSOCIATION_PAD.left - 8, y(value) + 4, percent(value),
        "croma-nipd-axis-label is-end"));
    });
    section.appendChild(svg);
    section.appendChild(inspection);
    return section;
  }

  function associationInspection(point) {
    return point.model + ", " + (point.isControl
      ? "natural-image reference excluded from trend and correlation"
      : "pathology encoder") + ", median CRoMa " + decimal(point.x) + ", nIPD " +
      percent(point.y);
  }

  function paddedDomain(values) {
    var low = Math.min.apply(null, values);
    var high = Math.max.apply(null, values);
    var padding = Math.max((high - low) * 0.08, 0.01);
    return [low - padding, high + padding];
  }

  function renderOverview(state, choose) {
    var rows = state.pathology.concat(state.control);
    var values = rows.map(function (model) { return model.regimes[state.regime].nipd; }).concat([0]);
    var low = Math.min.apply(null, values);
    var high = Math.max.apply(null, values);
    var padding = Math.max((high - low) * 0.08, 0.01);
    low -= padding;
    high += padding;
    var controlGap = state.control.length ? 24 : 0;
    var height = 64 + rows.length * OVERVIEW_ROW + controlGap;
    var svg = svgEl("svg", {
      class: "croma-nipd-overview",
      viewBox: "0 0 " + OVERVIEW_WIDTH + " " + height,
      role: "group",
      "aria-label": "Pathology encoders ordered from higher to lower nIPD, with the normalized " +
        "change at V = 1 beside it; natural-image control separated",
    });
    var x = linear(low, high, OVERVIEW_LEFT, OVERVIEW_WIDTH - OVERVIEW_RIGHT);
    var zero = svgEl("line", {
      x1: x(0), x2: x(0), y1: 28, y2: height - 20, class: "croma-nipd-zero",
    });
    svg.appendChild(zero);
    svg.appendChild(svgText(OVERVIEW_NIPD_COLUMN, 16, "nIPD", "croma-nipd-column-header"));
    svg.appendChild(svgText(OVERVIEW_END_COLUMN, 16, "at V = 1", "croma-nipd-column-header"));
    var pathologyCount = state.pathology.length;
    rows.forEach(function (model, index) {
      var offset = index >= pathologyCount ? controlGap : 0;
      var y = 42 + index * OVERVIEW_ROW + offset;
      if (index === pathologyCount && state.control.length) {
        svg.appendChild(svgText(8, y - 15, "Natural-image control (not ranked)", "croma-nipd-control-label"));
        svg.appendChild(svgEl("line", {
          x1: 8, x2: OVERVIEW_WIDTH - 8, y1: y - 10, y2: y - 10,
          class: "croma-nipd-control-separator",
        }));
      }
      var result = model.regimes[state.regime];
      var group = svgEl("g", {
        class: "croma-nipd-model" + (model.model === state.selected.model ? " is-selected" : "") +
          (state.comparison && model.model === state.comparison.model ? " is-comparison" : "") +
          (model.is_control ? " is-control" : ""),
        role: "button",
        tabindex: "0",
        "aria-label": model.model + ", nIPD " + percent(result.nipd) +
          ", normalized change at V = 1 " + percent(endOfRange(result)) +
          (collapses(result) ? ", collapses to chance" : "") +
          (state.comparison && model.model === state.comparison.model ? ", comparison model" : "") +
          (model.is_control ? ", natural-image control, not ranked" : ", pathology encoder"),
      });
      group.appendChild(svgEl("rect", {
        x: 0, y: y - 22, width: OVERVIEW_WIDTH, height: 44, class: "croma-nipd-hit",
      }));
      group.appendChild(svgText(OVERVIEW_LEFT - 10, y + 4, model.model, "croma-nipd-model-label"));
      group.appendChild(svgEl("circle", { cx: x(result.nipd), cy: y, r: 5, class: "croma-nipd-dot" }));
      group.appendChild(svgText(OVERVIEW_NIPD_COLUMN, y + 4, percent(result.nipd), "croma-nipd-value"));
      group.appendChild(svgText(OVERVIEW_END_COLUMN, y + 4, endOfRangeText(result),
        "croma-nipd-value" + (collapses(result) ? " is-collapsed" : "")));
      activate(group, function () { choose(model.model); });
      svg.appendChild(group);
    });
    svg.appendChild(svgText(OVERVIEW_LEFT, height - 4, percent(low), "croma-nipd-axis-label"));
    svg.appendChild(svgText(x(0), height - 4, "0", "croma-nipd-axis-label is-middle"));
    svg.appendChild(svgText(OVERVIEW_WIDTH - OVERVIEW_RIGHT, height - 4, percent(high), "croma-nipd-axis-label is-end"));
    var scroll = htmlEl("div", "croma-nipd-overview-scroll");
    scroll.appendChild(svg);
    return scroll;
  }

  function renderDetail(view) {
    var state = view.snapshot();
    var section = htmlEl("section", "croma-nipd-detail");
    section.setAttribute("aria-labelledby", "croma-nipd-active-heading");
    var heading = htmlEl("h3", "croma-nipd-heading");
    heading.id = "croma-nipd-active-heading";
    heading.textContent = state.selected.model + " normalized performance change" +
      (state.comparison ? " compared with " + state.comparison.model : "");
    section.appendChild(heading);

    if (state.comparison) {
      var swap = htmlEl("button", "croma-nipd-swap");
      swap.type = "button";
      swap.textContent = "Swap active model";
      swap.setAttribute("aria-label", "Make " + state.comparison.model + " the active model");
      swap.addEventListener("click", function () {
        view.swapFocus();
        renderExplorer(section.parentNode, view, "swap");
      });
      section.appendChild(swap);
    }

    var metrics = htmlEl("div", "croma-nipd-metrics");
    metrics.appendChild(modelMetrics(state, state.selected, "Active"));
    if (state.comparison) metrics.appendChild(modelMetrics(state, state.comparison, "Comparison"));
    section.appendChild(metrics);

    var descriptionId = "croma-nipd-plot-description";
    var description = htmlEl("p", "croma-sr-only");
    description.id = descriptionId;
    description.textContent = "Active and optional comparison normalized performance-change trajectories " +
      "with real sampled Cramér's V points, paired-repeat 95% t-intervals, and a zero reference. " +
      "Only the active model has patterned positive and negative signed-area lobes; the comparison " +
      "uses a dashed line and square points." +
      (state.yDomain[0] <= -1
        ? " A chance-level reference at minus 100% marks the loss of the whole above-chance margin."
        : "");
    section.appendChild(description);
    var inspection = htmlEl("p", "croma-nipd-inspection");
    inspection.setAttribute("aria-live", "polite");
    section.appendChild(renderTrajectory(state, view, inspection, descriptionId));
    updateInspection(inspection, view.inspectPoint(0));
    section.appendChild(inspection);
    var interpretation = htmlEl("p", "croma-nipd-cancellation");
    interpretation.textContent = "This signed area reports net change over the whole range. Positive and " +
      "negative lobes can cancel, and a model can hold a moderate area yet still fall to chance at V = 1, " +
      "so read the area together with the normalized change at V = 1 above.";
    section.appendChild(interpretation);
    return section;
  }

  function modelMetrics(state, selectedModel, role) {
    var result = selectedModel.regimes[state.regime];
    var list = htmlEl("dl", "croma-nipd-model-metrics is-" + role.toLowerCase());
    list.setAttribute("aria-label", role + " model " + selectedModel.model);
    var name = htmlEl("div", "croma-nipd-model-metric-name");
    var term = htmlEl("dt");
    var description = htmlEl("dd");
    term.textContent = role + " model";
    description.textContent = selectedModel.model;
    name.appendChild(term);
    name.appendChild(description);
    list.appendChild(name);
    metric(list, "nIPD from mean curve", percent(result.nipd));
    metric(list, "Normalized change at V = 1", endOfRangeText(result), collapses(result));
    metric(list, "Baseline balanced accuracy", decimal(result.baseline_balanced_accuracy));
    return list;
  }

  /* The endpoint the pooled area hides: at -100% the probe keeps none of its
     above-chance margin, so a near-100% fall is a collapse to chance. */
  function endOfRange(result) {
    var trajectory = result.mean_normalized_trajectory;
    return trajectory[trajectory.length - 1];
  }

  function collapses(result) {
    return endOfRange(result) <= COLLAPSE;
  }

  function endOfRangeText(result) {
    return percent(endOfRange(result)) + (collapses(result) ? " ≈ chance" : "");
  }

  function renderTrajectory(state, view, inspection, descriptionId) {
    var xs = state.cohort.cramers_v;
    var yDomain = state.yDomain;
    var x = linear(0, 1, DETAIL_PAD.left, DETAIL_WIDTH - DETAIL_PAD.right);
    var y = linear(yDomain[0], yDomain[1], DETAIL_HEIGHT - DETAIL_PAD.bottom, DETAIL_PAD.top);
    var svg = svgEl("svg", {
      class: "croma-nipd-trajectory", viewBox: "0 0 " + DETAIL_WIDTH + " " + DETAIL_HEIGHT,
      role: "group", "aria-describedby": descriptionId,
      "aria-label": state.selected.model + " " + state.regime.toUpperCase() + " sampled trajectory",
    });
    var defs = svgEl("defs");
    defs.innerHTML = '<pattern id="nipd-positive" width="8" height="8" patternUnits="userSpaceOnUse">' +
      '<path d="M0 8L8 0" class="croma-nipd-pattern-line"/></pattern>' +
      '<pattern id="nipd-negative" width="6" height="6" patternUnits="userSpaceOnUse">' +
      '<path d="M0 3H6" class="croma-nipd-pattern-line"/></pattern>';
    svg.appendChild(defs);
    svg.appendChild(svgEl("line", {
      x1: DETAIL_PAD.left, x2: DETAIL_WIDTH - DETAIL_PAD.right,
      y1: y(0), y2: y(0), class: "croma-nipd-zero",
    }));
    // Absolute floor: a curve reaching it has lost the whole above-chance margin.
    if (yDomain[0] <= -1) {
      svg.appendChild(svgEl("line", {
        x1: DETAIL_PAD.left, x2: DETAIL_WIDTH - DETAIL_PAD.right,
        y1: y(-1), y2: y(-1), class: "croma-nipd-floor",
      }));
      svg.appendChild(svgText(DETAIL_WIDTH - DETAIL_PAD.right, y(-1) - 6,
        "chance level", "croma-nipd-floor-label"));
    }
    if (state.comparison) {
      svg.appendChild(renderSeries(state.comparison, "comparison", false));
    }
    svg.appendChild(renderSeries(state.selected, "active", true));
    svg.appendChild(svgText(DETAIL_WIDTH / 2, DETAIL_HEIGHT - 10, "Cramér's V", "croma-nipd-axis-title"));
    var yTitle = svgText(15, DETAIL_HEIGHT / 2, "Normalized performance change g(V)", "croma-nipd-axis-title");
    yTitle.setAttribute("transform", "rotate(-90 15 " + DETAIL_HEIGHT / 2 + ")");
    svg.appendChild(yTitle);
    [0, 0.5, 1].forEach(function (value) {
      svg.appendChild(svgText(x(value), DETAIL_HEIGHT - 34, decimal(value), "croma-nipd-axis-label is-middle"));
    });
    [yDomain[0], 0, yDomain[1]].forEach(function (value) {
      var label = svgText(DETAIL_PAD.left - 8, y(value) + 4, percent(value), "croma-nipd-axis-label is-end");
      svg.appendChild(label);
    });
    return svg;

    function renderSeries(selectedModel, role, shadeArea) {
      var result = selectedModel.regimes[state.regime];
      var group = svgEl("g", {
        class: "croma-nipd-series is-" + role,
        role: "group",
        "aria-label": (role === "active" ? "Active model " : "Comparison model ") +
          selectedModel.model + " sampled trajectory",
      });
      group.appendChild(svgEl("polygon", {
        points: result.ci95_low.map(function (value, index) { return x(xs[index]) + "," + y(value); })
          .concat(result.ci95_high.slice().reverse().map(function (value, reverseIndex) {
            return x(xs[xs.length - 1 - reverseIndex]) + "," + y(value);
          })).join(" "),
        class: "croma-nipd-interval is-" + role,
        "aria-label": selectedModel.model + " paired-repeat 95% t-interval",
      }));
      if (shadeArea) {
        splitSignSegments(xs, result.mean_normalized_trajectory).forEach(function (segment) {
          var first = segment.points[0];
          var last = segment.points[segment.points.length - 1];
          var points = [[first[0], 0]].concat(segment.points, [[last[0], 0]]);
          group.appendChild(svgEl("polygon", {
            points: points.map(function (point) { return x(point[0]) + "," + y(point[1]); }).join(" "),
            class: "croma-nipd-lobe is-" + segment.sign,
            fill: "url(#nipd-" + segment.sign + ")",
            "aria-label": segment.sign + " signed-area lobe for active model " + selectedModel.model,
          }));
        });
      }
      group.appendChild(svgEl("polyline", {
        points: result.mean_normalized_trajectory.map(function (value, index) {
          return x(xs[index]) + "," + y(value);
        }).join(" "),
        class: "croma-nipd-mean is-" + role,
      }));
      xs.forEach(function (value, index) {
        var point = view.inspectPoint(index, selectedModel.model);
        var sample = svgEl("g", {
          class: "croma-nipd-sample is-" + role, role: "button", tabindex: "0",
          "aria-label": inspectionText(point),
        });
        sample.appendChild(svgEl("line", {
          x1: x(value), x2: x(value), y1: y(point.low), y2: y(point.high),
          class: "croma-nipd-whisker is-" + role,
        }));
        sample.appendChild(svgEl("circle", {
          cx: x(value), cy: y(point.mean), r: 22, class: "croma-nipd-sample-hit",
        }));
        sample.appendChild(role === "active" ? svgEl("circle", {
          cx: x(value), cy: y(point.mean), r: 4, class: "croma-nipd-sample-dot is-active",
        }) : svgEl("rect", {
          x: x(value) - 4, y: y(point.mean) - 4, width: 8, height: 8,
          class: "croma-nipd-sample-dot is-comparison",
        }));
        activate(sample, function () {
          svg.querySelectorAll(".croma-nipd-sample").forEach(function (candidate) {
            candidate.classList.remove("is-inspected");
          });
          sample.classList.add("is-inspected");
          updateInspection(inspection, point);
        });
        group.appendChild(sample);
      });
      return group;
    }
  }

  function inspectionText(point) {
    return point.model + ": Cramér's V " + decimal(point.cramersV) + ", normalized mean change " +
      percent(point.mean) + ", " + point.interval + " " + percent(point.low) + " to " +
      percent(point.high);
  }

  function updateInspection(node, point) {
    node.textContent = inspectionText(point);
  }

  function activate(node, callback) {
    node.addEventListener("click", callback);
    node.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        callback();
      }
    });
  }

  function metric(list, label, value, flagged) {
    var wrapper = htmlEl("div", flagged ? "is-collapsed" : null);
    var term = htmlEl("dt");
    var description = htmlEl("dd");
    term.textContent = label;
    description.textContent = value;
    wrapper.appendChild(term);
    wrapper.appendChild(description);
    list.appendChild(wrapper);
  }

  function labelled(text, control) {
    var label = htmlEl("label", "croma-nipd-field");
    var name = htmlEl("span");
    name.textContent = text;
    label.appendChild(name);
    label.appendChild(control);
    return label;
  }

  function linear(fromLow, fromHigh, toLow, toHigh) {
    return function (value) {
      return toLow + ((value - fromLow) / (fromHigh - fromLow)) * (toHigh - toLow);
    };
  }

  function decimal(value) { return Number(value).toFixed(3); }
  function percent(value) { return (100 * Number(value)).toFixed(1) + "%"; }

  function htmlEl(tag, className) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  function svgEl(tag, attributes) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attributes || {}).forEach(function (key) { node.setAttribute(key, attributes[key]); });
    return node;
  }

  function svgText(x, y, text, className) {
    var node = svgEl("text", { x: x, y: y, class: className });
    node.textContent = text;
    return node;
  }

  return {
    boot: boot,
    contextYDomain: contextYDomain,
    createView: createView,
    splitSignSegments: splitSignSegments,
  };
});
