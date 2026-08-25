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
  var OVERVIEW_RIGHT = 38;
  var OVERVIEW_ROW = 44;
  var DETAIL_WIDTH = 720;
  var DETAIL_HEIGHT = 360;
  var DETAIL_PAD = { top: 26, right: 24, bottom: 55, left: 62 };

  function createView(payload) {
    if (!payload || payload.schema_version !== 1 || !Array.isArray(payload.cohorts)) {
      throw new Error("Unsupported nIPD publication payload");
    }
    var cohorts = payload.cohorts;
    var state = { cohort: cohorts[0].slug, regime: "id", model: null };
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
    }

    function select(name) {
      if (!cohort().models.some(function (candidate) { return candidate.model === name; })) {
        throw new Error("Unknown model in active nIPD context: " + name);
      }
      state.model = name;
    }

    function inspectPoint(index) {
      var activeCohort = cohort();
      var activeModel = model();
      var result = activeModel.regimes[state.regime];
      if (!Number.isInteger(index) || index < 0 || index >= activeCohort.cramers_v.length) {
        throw new Error("Unknown sampled point: " + index);
      }
      return {
        model: activeModel.model,
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
        regime: state.regime,
        selected: activeModel,
        pathology: pathologyModels(activeCohort, state.regime),
        control: activeCohort.models.filter(function (candidate) { return candidate.is_control; }),
        yDomain: contextYDomain(activeCohort, state.regime),
        readout: {
          nipd: result.nipd,
          baselineBalancedAccuracy: result.baseline_balanced_accuracy,
          chance: activeCohort.chance,
          baselineSkill: result.baseline_skill,
        },
      };
    }

    return { inspectPoint: inspectPoint, select: select, setContext: setContext, snapshot: snapshot };
  }

  function pathologyModels(cohort, regime) {
    return cohort.models
      .filter(function (model) { return model.ranked && !model.is_control; })
      .slice()
      .sort(function (left, right) {
        return right.regimes[regime].nipd - left.regimes[regime].nipd ||
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
    controls.appendChild(labelled("Cohort", cohortSelect));
    controls.appendChild(labelled("Evaluation regime", regimeSelect));
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

    cohortSelect.addEventListener("change", function () {
      view.setContext(cohortSelect.value, regimeSelect.value);
      renderExplorer(mount, view, "cohort");
    });
    regimeSelect.addEventListener("change", function () {
      view.setContext(cohortSelect.value, regimeSelect.value);
      renderExplorer(mount, view, "regime");
    });
    if (restoreFocus === "model") mount.querySelector(".croma-nipd-model.is-selected").focus();
    if (restoreFocus === "cohort") cohortSelect.focus();
    if (restoreFocus === "regime") regimeSelect.focus();
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
    var height = 48 + rows.length * OVERVIEW_ROW + controlGap;
    var svg = svgEl("svg", {
      class: "croma-nipd-overview",
      viewBox: "0 0 " + OVERVIEW_WIDTH + " " + height,
      role: "group",
      "aria-label": "Pathology encoders ordered from higher to lower nIPD; natural-image control separated",
    });
    var x = linear(low, high, OVERVIEW_LEFT, OVERVIEW_WIDTH - OVERVIEW_RIGHT);
    var zero = svgEl("line", {
      x1: x(0), x2: x(0), y1: 12, y2: height - 20, class: "croma-nipd-zero",
    });
    svg.appendChild(zero);
    var pathologyCount = state.pathology.length;
    rows.forEach(function (model, index) {
      var offset = index >= pathologyCount ? controlGap : 0;
      var y = 26 + index * OVERVIEW_ROW + offset;
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
          (model.is_control ? " is-control" : ""),
        role: "button",
        tabindex: "0",
        "aria-label": model.model + ", nIPD " + percent(result.nipd) +
          (model.is_control ? ", natural-image control, not ranked" : ", pathology encoder"),
      });
      group.appendChild(svgEl("rect", {
        x: 0, y: y - 22, width: OVERVIEW_WIDTH, height: 44, class: "croma-nipd-hit",
      }));
      group.appendChild(svgText(OVERVIEW_LEFT - 10, y + 4, model.model, "croma-nipd-model-label"));
      group.appendChild(svgEl("circle", { cx: x(result.nipd), cy: y, r: 5, class: "croma-nipd-dot" }));
      group.appendChild(svgText(OVERVIEW_WIDTH - 2, y + 4, percent(result.nipd), "croma-nipd-value"));
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
    var result = state.selected.regimes[state.regime];
    var section = htmlEl("section", "croma-nipd-detail");
    section.setAttribute("aria-labelledby", "croma-nipd-active-heading");
    var heading = htmlEl("h3", "croma-nipd-heading");
    heading.id = "croma-nipd-active-heading";
    heading.textContent = state.selected.model + " normalized performance change";
    section.appendChild(heading);

    var metrics = htmlEl("dl", "croma-nipd-metrics");
    metric(metrics, "nIPD from mean curve", percent(result.nipd));
    metric(metrics, "Baseline balanced accuracy", decimal(result.baseline_balanced_accuracy));
    metric(metrics, "Chance", decimal(state.cohort.chance));
    metric(metrics, "Baseline skill", decimal(result.baseline_skill));
    section.appendChild(metrics);

    var descriptionId = "croma-nipd-plot-description";
    var description = htmlEl("p", "croma-sr-only");
    description.id = descriptionId;
    description.textContent = "Mean normalized performance-change trajectory with real sampled Cramér's V points, " +
      "paired-repeat 95% t-intervals, a zero reference, and signed area to zero. " +
      "Positive and negative lobes use different fill patterns.";
    section.appendChild(description);
    var inspection = htmlEl("p", "croma-nipd-inspection");
    inspection.setAttribute("aria-live", "polite");
    section.appendChild(renderTrajectory(state, view, inspection, descriptionId));
    updateInspection(inspection, view.inspectPoint(0));
    section.appendChild(inspection);
    var interpretation = htmlEl("p", "croma-nipd-cancellation");
    interpretation.textContent = "This signed area reports net change. Positive and negative lobes can cancel, " +
      "so a value near zero does not show that the trajectory changed little at every point.";
    section.appendChild(interpretation);
    return section;
  }

  function renderTrajectory(state, view, inspection, descriptionId) {
    var result = state.selected.regimes[state.regime];
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
    svg.appendChild(svgEl("polygon", {
      points: result.ci95_low.map(function (value, index) { return x(xs[index]) + "," + y(value); })
        .concat(result.ci95_high.slice().reverse().map(function (value, reverseIndex) {
          return x(xs[xs.length - 1 - reverseIndex]) + "," + y(value);
        })).join(" "),
      class: "croma-nipd-interval",
    }));
    splitSignSegments(xs, result.mean_normalized_trajectory).forEach(function (segment) {
      var first = segment.points[0];
      var last = segment.points[segment.points.length - 1];
      var points = [[first[0], 0]].concat(segment.points, [[last[0], 0]]);
      svg.appendChild(svgEl("polygon", {
        points: points.map(function (point) { return x(point[0]) + "," + y(point[1]); }).join(" "),
        class: "croma-nipd-lobe is-" + segment.sign,
        fill: "url(#nipd-" + segment.sign + ")",
        "aria-label": segment.sign + " signed-area lobe",
      }));
    });
    svg.appendChild(svgEl("polyline", {
      points: result.mean_normalized_trajectory.map(function (value, index) {
        return x(xs[index]) + "," + y(value);
      }).join(" "),
      class: "croma-nipd-mean",
    }));
    xs.forEach(function (value, index) {
      var point = view.inspectPoint(index);
      var group = svgEl("g", {
        class: "croma-nipd-sample", role: "button", tabindex: "0",
        "aria-label": inspectionText(point),
      });
      group.appendChild(svgEl("line", {
        x1: x(value), x2: x(value), y1: y(point.low), y2: y(point.high), class: "croma-nipd-whisker",
      }));
      group.appendChild(svgEl("circle", {
        cx: x(value), cy: y(point.mean), r: 22, class: "croma-nipd-sample-hit",
      }));
      group.appendChild(svgEl("circle", {
        cx: x(value), cy: y(point.mean), r: 4, class: "croma-nipd-sample-dot",
      }));
      activate(group, function () {
        svg.querySelectorAll(".croma-nipd-sample").forEach(function (candidate) {
          candidate.classList.remove("is-inspected");
        });
        group.classList.add("is-inspected");
        updateInspection(inspection, point);
      });
      svg.appendChild(group);
    });
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
  }

  function inspectionText(point) {
    return "Cramér's V " + decimal(point.cramersV) + ", normalized mean change " +
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

  function metric(list, label, value) {
    var wrapper = htmlEl("div");
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
