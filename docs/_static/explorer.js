/*
 * The CRoMa distribution explorer.
 *
 * The per-sample CRoMa distribution is the evidence object the tail statistics summarise,
 * so the widget shows all of it, all the time: an overview of every encoder's histogram
 * (master), the selected encoder's full histogram with a range brush (detail), and an
 * optional second encoder overlaid for comparison. Nothing is hidden behind an
 * interaction: the detail panel is always on screen, and the highlighted overview row is
 * the thing it details.
 *
 * The data is `results/distributions.json` -- 200-bin histograms per encoder and cohort,
 * committed alongside the tables, and copied to the site root by `html_extra_path`. It is
 * loaded on demand, so pages without the widget pay nothing but this file.
 *
 * Per-sample identifiers and tile thumbnails are deliberately absent. Identifiers would add
 * megabytes per cohort for a lookup nobody can act on without the cohort in hand, and
 * thumbnails would mean redistributing three datasets under three different licences.
 */
(function () {
  "use strict";

  /* Read synchronously, while the script is still executing: `document.currentScript` is
     null by the time DOMContentLoaded fires, and Sphinx injects this file into <head>, so
     the element it mounts on does not exist yet either. */
  var SCRIPT_SRC = document.currentScript && document.currentScript.src;

  var PAD = { top: 8, right: 8, bottom: 26, left: 44 };
  var WIDTH = 640;
  var HEIGHT = 220;
  var ROW_WIDTH = 480;
  var ROW_HEIGHT = 26;

  var container = null;
  var state = {
    data: null,
    cohort: null,
    model: null,
    compare: null,
    from: null,
    to: null,
    applySelection: null,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  function boot() {
    container = document.getElementById("croma-explorer");
    if (!container || !SCRIPT_SRC) return;

    fetch(new URL("../distributions.json", SCRIPT_SRC).href)
      .then(function (response) {
        if (!response.ok) throw new Error(response.status + " " + response.statusText);
        return response.json();
      })
      .then(function (data) {
        state.data = data;
        state.cohort = Object.keys(data.cohorts)[0];
        state.model = rankedModels(data.cohorts[state.cohort])[0];
        render();
      })
      .catch(function (error) {
        container.textContent =
          "The distribution data could not be loaded (" + error.message + ").";
      });
  }

  /* ------------------------------------------------------------------ ordering */

  /* Encoders in the result tables' order: by median CRoMa, descending -- the tables sort
     by the median, so the overview must agree with the page it sits on. The binned median
     is read off the histogram's CDF. The payload is keyed by name with no ordering. */
  function rankedModels(cohort) {
    var nBins = state.data.n_bins;
    return Object.keys(cohort.models).sort(function (a, b) {
      return (
        histogramMedian(cohort, cohort.models[b], nBins) -
          histogramMedian(cohort, cohort.models[a], nBins) ||
        a.localeCompare(b)
      );
    });
  }

  function histogramMedian(cohort, counts, nBins) {
    var width = (cohort.hi - cohort.lo) / nBins;
    var total = 0;
    for (var i = 0; i < counts.length; i++) total += counts[i];
    var seen = 0;
    for (var j = 0; j < counts.length; j++) {
      seen += counts[j];
      if (seen >= total / 2) return cohort.lo + (j + 0.5) * width;
    }
    return 0;
  }

  /* ---------------------------------------------------------------- rendering */

  function render() {
    var data = state.data;
    var cohort = data.cohorts[state.cohort];
    if (state.compare === state.model) state.compare = null;

    container.innerHTML = "";
    var bar = controls(data, cohort);
    container.appendChild(bar);
    container.appendChild(overview(cohort));

    var counts = cohort.models[state.model];
    var compareCounts = state.compare ? cohort.models[state.compare] : null;
    var view = {
      cohort: cohort,
      counts: counts,
      compareCounts: compareCounts,
      nBins: data.n_bins,
      bars: [],
    };
    container.appendChild(detailHeading());
    var svg = histogram(cohort, counts, compareCounts, data.n_bins, view);
    container.appendChild(svg);
    view.readout = el("p", "croma-explorer-readout");
    container.appendChild(view.readout);
    view.reset = bar.querySelector(".croma-explorer-reset");

    /* Repaint the selection on the DOM that is already there. A full render() inside the
       drag would destroy the SVG holding the pointer capture -- the bug that reduced the
       brush to single-bin clicks. */
    state.applySelection = function () {
      var selection = selectedBins(view.nBins);
      view.bars.forEach(function (rect, i) {
        if (!rect) return;
        var inRange = selection && i >= selection.lo && i <= selection.hi;
        rect.setAttribute(
          "class",
          "croma-explorer-bar" + (selection ? (inRange ? " is-selected" : " is-dimmed") : "")
        );
      });
      view.reset.disabled = selection === null;
      writeReadout(view);
    };
    state.applySelection();
  }

  function controls(data, cohort) {
    var bar = el("div", "croma-explorer-controls");

    var cohortSelect = el("select");
    cohortSelect.setAttribute("aria-label", "Cohort");
    Object.keys(data.cohorts).forEach(function (slug) {
      var option = el("option");
      option.value = slug;
      option.textContent = data.cohorts[slug].label;
      if (slug === state.cohort) option.selected = true;
      cohortSelect.appendChild(option);
    });
    cohortSelect.addEventListener("change", function () {
      state.cohort = cohortSelect.value;
      var next = data.cohorts[state.cohort];
      /* Keep the encoders across a cohort switch when they were scored there -- the point
         of switching is usually to follow a model. The brush is cleared because the axis
         domain changes with the cohort. */
      if (!next.models[state.model]) state.model = rankedModels(next)[0];
      if (state.compare && !next.models[state.compare]) state.compare = null;
      state.from = state.to = null;
      render();
    });

    var compareSelect = el("select");
    compareSelect.setAttribute("aria-label", "Compare with");
    var none = el("option");
    none.value = "";
    none.textContent = "None";
    compareSelect.appendChild(none);
    rankedModels(cohort).forEach(function (name) {
      if (name === state.model) return;
      var option = el("option");
      option.value = option.textContent = name;
      if (name === state.compare) option.selected = true;
      compareSelect.appendChild(option);
    });
    compareSelect.addEventListener("change", function () {
      state.compare = compareSelect.value || null;
      render();
    });

    bar.appendChild(labelled("Cohort", cohortSelect));
    bar.appendChild(labelled("Compare with", compareSelect));

    var reset = el("button", "croma-explorer-reset");
    reset.type = "button";
    reset.textContent = "Clear selection";
    reset.disabled = state.from === null;
    reset.addEventListener("click", function () {
      state.from = state.to = null;
      state.applySelection();
    });
    bar.appendChild(reset);
    return bar;
  }

  /* The master list: every encoder of the cohort as a compact histogram row on the shared
     axis, in table order. Clicking a row moves the detail; the highlight ties the two. */
  function overview(cohort) {
    var list = el("div", "croma-explorer-overview");
    list.setAttribute("role", "list");
    rankedModels(cohort).forEach(function (name) {
      var row = el("button", "croma-explorer-row");
      row.type = "button";
      if (name === state.model) row.className += " is-selected";
      else if (name === state.compare) row.className += " is-compare";
      row.setAttribute(
        "aria-label",
        name + (name === state.model ? " (shown in detail)" : "")
      );

      var label = el("span", "croma-explorer-row-name");
      label.textContent = name;
      row.appendChild(label);
      row.appendChild(miniHistogram(cohort, cohort.models[name]));

      row.addEventListener("click", function () {
        if (state.compare === name) state.compare = null;
        state.model = name;
        render();
      });
      list.appendChild(row);
    });
    return list;
  }

  function miniHistogram(cohort, counts) {
    var nBins = state.data.n_bins;
    var svg = svgEl("svg", {
      viewBox: "0 0 " + ROW_WIDTH + " " + ROW_HEIGHT,
      class: "croma-explorer-mini",
      "aria-hidden": "true",
    });
    var x = scale(cohort, ROW_WIDTH, 0);

    if (cohort.lo < 0) {
      svg.appendChild(
        svgEl("rect", {
          x: 0,
          y: 0,
          width: Math.max(0, x(Math.min(0, cohort.hi))),
          height: ROW_HEIGHT,
          class: "croma-explorer-fragile",
        })
      );
    }

    var peak = Math.max.apply(null, counts) || 1;
    var binWidth = ROW_WIDTH / nBins;
    for (var i = 0; i < nBins; i++) {
      if (!counts[i]) continue;
      var height = (counts[i] / peak) * ROW_HEIGHT;
      svg.appendChild(
        svgEl("rect", {
          x: i * binWidth,
          y: ROW_HEIGHT - height,
          width: Math.max(binWidth, 0.6),
          height: height,
          class: "croma-explorer-bar",
        })
      );
    }

    if (cohort.lo < 0 && cohort.hi > 0) {
      svg.appendChild(
        svgEl("line", {
          x1: x(0),
          x2: x(0),
          y1: 0,
          y2: ROW_HEIGHT,
          class: "croma-explorer-zero",
        })
      );
    }
    return svg;
  }

  function detailHeading() {
    var heading = el("p", "croma-explorer-detail-heading");
    var name = el("strong");
    name.textContent = state.model;
    heading.appendChild(name);
    if (state.compare) {
      heading.appendChild(document.createTextNode(" vs "));
      var other = el("strong", "croma-explorer-compare-name");
      other.textContent = state.compare;
      heading.appendChild(other);
    }
    return heading;
  }

  function histogram(cohort, counts, compareCounts, nBins, view) {
    var plotWidth = WIDTH - PAD.left - PAD.right;
    var plotHeight = HEIGHT - PAD.top - PAD.bottom;
    /* One count scale for both encoders, so the shapes are comparable. */
    var peak = Math.max.apply(null, compareCounts ? counts.concat(compareCounts) : counts) || 1;

    var svg = svgEl("svg", {
      viewBox: "0 0 " + WIDTH + " " + HEIGHT,
      class: "croma-explorer-plot",
      role: "img",
      "aria-label":
        "Per-sample CRoMa histogram for " +
        state.model +
        (state.compare ? " compared with " + state.compare : "") +
        " on " +
        cohort.label,
    });

    var x = scale(cohort, plotWidth, PAD.left);

    /* The confounder-dominant half. */
    if (cohort.lo < 0) {
      svg.appendChild(
        svgEl("rect", {
          x: x(cohort.lo),
          y: PAD.top,
          width: Math.max(0, x(Math.min(0, cohort.hi)) - x(cohort.lo)),
          height: plotHeight,
          class: "croma-explorer-fragile",
        })
      );
    }

    var binWidth = plotWidth / nBins;
    for (var i = 0; i < nBins; i++) {
      if (!counts[i]) continue;
      var height = (counts[i] / peak) * plotHeight;
      var rect = svgEl("rect", {
        x: PAD.left + i * binWidth,
        y: PAD.top + plotHeight - height,
        width: Math.max(binWidth, 0.6),
        height: height,
        class: "croma-explorer-bar",
      });
      view.bars[i] = rect;
      svg.appendChild(rect);
    }

    if (compareCounts) {
      svg.appendChild(
        svgEl("path", {
          d: outlinePath(compareCounts, nBins, peak, plotWidth, plotHeight),
          class: "croma-explorer-compare-outline",
        })
      );
    }

    if (cohort.lo < 0 && cohort.hi > 0) {
      svg.appendChild(
        svgEl("line", {
          x1: x(0),
          x2: x(0),
          y1: PAD.top,
          y2: PAD.top + plotHeight,
          class: "croma-explorer-zero",
        })
      );
    }

    svg.appendChild(
      svgEl("line", {
        x1: PAD.left,
        x2: PAD.left + plotWidth,
        y1: PAD.top + plotHeight,
        y2: PAD.top + plotHeight,
        class: "croma-explorer-axis",
      })
    );

    tickValues(cohort).forEach(function (value) {
      var tick = svgEl("text", {
        x: x(value),
        y: HEIGHT - 8,
        class: "croma-explorer-tick",
        "text-anchor": "middle",
      });
      tick.textContent = value.toFixed(1);
      svg.appendChild(tick);
    });

    var axisLabel = svgEl("text", {
      x: PAD.left,
      y: PAD.top + 10,
      class: "croma-explorer-tick",
    });
    axisLabel.textContent = "samples";
    svg.appendChild(axisLabel);

    attachBrush(svg, plotWidth, nBins);
    return svg;
  }

  /* The comparison encoder as a step outline over the same bins: the shapes stay
     distinguishable where translucent fills would blend. */
  function outlinePath(counts, nBins, peak, plotWidth, plotHeight) {
    var binWidth = plotWidth / nBins;
    var baseline = PAD.top + plotHeight;
    var parts = ["M" + PAD.left + " " + baseline];
    for (var i = 0; i < nBins; i++) {
      var y = baseline - (counts[i] / peak) * plotHeight;
      parts.push("H" + (PAD.left + i * binWidth));
      parts.push("V" + y);
      parts.push("H" + (PAD.left + (i + 1) * binWidth));
    }
    parts.push("V" + baseline);
    return parts.join(" ");
  }

  function scale(cohort, plotWidth, offset) {
    return function (value) {
      return offset + ((value - cohort.lo) / (cohort.hi - cohort.lo)) * plotWidth;
    };
  }

  function tickValues(cohort) {
    var ticks = [];
    for (var value = Math.ceil(cohort.lo * 4) / 4; value <= cohort.hi; value += 0.25) {
      ticks.push(Math.round(value * 100) / 100);
    }
    return ticks;
  }

  /* Drag across the plot to pick a bin range. Pointer events cover mouse, touch and pen
     with one code path, and capture keeps the drag alive past the edge of the SVG. */
  function attachBrush(svg, plotWidth, nBins) {
    var anchor = null;

    function binAt(event) {
      var box = svg.getBoundingClientRect();
      var scale = svg.viewBox.baseVal.width / box.width;
      var local = (event.clientX - box.left) * scale - PAD.left;
      return clamp(Math.floor((local / plotWidth) * nBins), 0, nBins - 1);
    }

    svg.addEventListener("pointerdown", function (event) {
      anchor = binAt(event);
      state.from = state.to = anchor;
      svg.setPointerCapture(event.pointerId);
      state.applySelection();
      event.preventDefault();
    });

    svg.addEventListener("pointermove", function (event) {
      if (anchor === null) return;
      var current = binAt(event);
      state.from = Math.min(anchor, current);
      state.to = Math.max(anchor, current);
      state.applySelection();
    });

    ["pointerup", "pointercancel"].forEach(function (type) {
      svg.addEventListener(type, function () {
        anchor = null;
      });
    });
  }

  function selectedBins(nBins) {
    if (state.from === null || state.to === null) return null;
    return { lo: clamp(state.from, 0, nBins - 1), hi: clamp(state.to, 0, nBins - 1) };
  }

  function writeReadout(view) {
    var selection = selectedBins(view.nBins);

    if (!selection) {
      view.readout.textContent =
        sum(view.counts).toLocaleString() +
        " samples. Drag across the histogram to count a range.";
      return;
    }

    var width = (view.cohort.hi - view.cohort.lo) / view.nBins;
    var lo = view.cohort.lo + selection.lo * width;
    var hi = view.cohort.lo + (selection.hi + 1) * width;

    var html =
      "CRoMa between <strong>" +
      lo.toFixed(2) +
      "</strong> and <strong>" +
      hi.toFixed(2) +
      "</strong>: " +
      rangeCount(state.model, view.counts, selection);
    if (view.compareCounts) {
      html +=
        " &middot; " + rangeCount(state.compare, view.compareCounts, selection);
    }
    view.readout.innerHTML = html;
  }

  function rangeCount(name, counts, selection) {
    var total = sum(counts);
    var inRange = 0;
    for (var i = selection.lo; i <= selection.hi; i++) inRange += counts[i];
    return (
      name +
      " <strong>" +
      inRange.toLocaleString() +
      "</strong> of " +
      total.toLocaleString() +
      " (" +
      ((100 * inRange) / (total || 1)).toFixed(1) +
      "%)"
    );
  }

  /* ------------------------------------------------------------------- helpers */

  function sum(values) {
    return values.reduce(function (acc, value) {
      return acc + value;
    }, 0);
  }

  function el(tag, className) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  function svgEl(tag, attributes) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attributes || {}).forEach(function (key) {
      node.setAttribute(key, attributes[key]);
    });
    return node;
  }

  function labelled(text, control) {
    var wrapper = el("label", "croma-explorer-field");
    var span = el("span");
    span.textContent = text;
    wrapper.appendChild(span);
    wrapper.appendChild(control);
    return wrapper;
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }
})();
