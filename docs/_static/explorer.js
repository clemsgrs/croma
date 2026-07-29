/*
 * The CRoMa distribution explorer.
 *
 * F(0) and LTM10 are two numbers summarising a shape. This lets a reader size the tail
 * themselves: pick a cohort and an encoder, drag across the histogram, and read how many
 * samples fall in the range.
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
  var HEIGHT = 220;

  var container = null;
  var state = { data: null, cohort: null, model: null, from: null, to: null };

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
        state.model = defaultModel(data.cohorts[state.cohort], data.n_bins);
        render();
      })
      .catch(function (error) {
        container.textContent =
          "The distribution data could not be loaded (" + error.message + ").";
      });
  }

  /* The encoder with the least confounder-dominant mass, so the widget opens on the
     cohort's strongest representation. The payload is keyed by name with no ordering. */
  function defaultModel(cohort, nBins) {
    var names = Object.keys(cohort.models);
    return names.reduce(function (best, name) {
      return negativeMass(cohort, cohort.models[name], nBins) <
        negativeMass(cohort, cohort.models[best], nBins)
        ? name
        : best;
    }, names[0]);
  }

  function negativeMass(cohort, counts, nBins) {
    var width = (cohort.hi - cohort.lo) / nBins;
    var total = 0;
    var below = 0;
    for (var i = 0; i < counts.length; i++) {
      total += counts[i];
      if (cohort.lo + (i + 1) * width <= 0) below += counts[i];
    }
    return total ? below / total : 0;
  }

  /* ---------------------------------------------------------------- rendering */

  function render() {
    var data = state.data;
    var cohort = data.cohorts[state.cohort];

    container.innerHTML = "";
    container.appendChild(controls(data, cohort));

    var counts = cohort.models[state.model];
    var svg = histogram(cohort, counts, data.n_bins);
    container.appendChild(svg);
    container.appendChild(readout(cohort, counts, data.n_bins));
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
      /* Keep the encoder across a cohort switch when it was scored there -- the point of
         switching is usually to follow one model. */
      if (!next.models[state.model]) state.model = defaultModel(next, data.n_bins);
      state.from = state.to = null;
      render();
    });

    var modelSelect = el("select");
    modelSelect.setAttribute("aria-label", "Encoder");
    Object.keys(cohort.models)
      .sort()
      .forEach(function (name) {
        var option = el("option");
        option.value = option.textContent = name;
        if (name === state.model) option.selected = true;
        modelSelect.appendChild(option);
      });
    modelSelect.addEventListener("change", function () {
      state.model = modelSelect.value;
      render();
    });

    bar.appendChild(labelled("Cohort", cohortSelect));
    bar.appendChild(labelled("Encoder", modelSelect));

    var reset = el("button", "croma-explorer-reset");
    reset.type = "button";
    reset.textContent = "Clear selection";
    reset.disabled = state.from === null;
    reset.addEventListener("click", function () {
      state.from = state.to = null;
      render();
    });
    bar.appendChild(reset);
    return bar;
  }

  function histogram(cohort, counts, nBins) {
    var width = 640;
    var plotWidth = width - PAD.left - PAD.right;
    var plotHeight = HEIGHT - PAD.top - PAD.bottom;
    var peak = Math.max.apply(null, counts) || 1;

    var svg = svgEl("svg", {
      viewBox: "0 0 " + width + " " + HEIGHT,
      class: "croma-explorer-plot",
      role: "img",
      "aria-label":
        "Per-sample CRoMa histogram for " + state.model + " on " + cohort.label,
    });

    var x = function (value) {
      return PAD.left + ((value - cohort.lo) / (cohort.hi - cohort.lo)) * plotWidth;
    };

    /* The confounder-dominant half, shaded the way the static ridgelines shade it. */
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

    var selection = selectedBins(nBins);
    var binWidth = plotWidth / nBins;
    for (var i = 0; i < nBins; i++) {
      if (!counts[i]) continue;
      var height = (counts[i] / peak) * plotHeight;
      var inRange = selection && i >= selection.lo && i <= selection.hi;
      svg.appendChild(
        svgEl("rect", {
          x: PAD.left + i * binWidth,
          y: PAD.top + plotHeight - height,
          width: Math.max(binWidth, 0.6),
          height: height,
          class: "croma-explorer-bar" + (selection ? (inRange ? " is-selected" : " is-dimmed") : ""),
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
      render();
      event.preventDefault();
    });

    svg.addEventListener("pointermove", function (event) {
      if (anchor === null) return;
      var current = binAt(event);
      state.from = Math.min(anchor, current);
      state.to = Math.max(anchor, current);
      render();
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

  function readout(cohort, counts, nBins) {
    var box = el("p", "croma-explorer-readout");
    var total = counts.reduce(function (sum, value) {
      return sum + value;
    }, 0);
    var selection = selectedBins(nBins);

    if (!selection) {
      box.textContent =
        total.toLocaleString() +
        " samples. Drag across the histogram to count a range.";
      return box;
    }

    var width = (cohort.hi - cohort.lo) / nBins;
    var lo = cohort.lo + selection.lo * width;
    var hi = cohort.lo + (selection.hi + 1) * width;
    var inRange = 0;
    for (var i = selection.lo; i <= selection.hi; i++) inRange += counts[i];

    box.innerHTML =
      "<strong>" +
      inRange.toLocaleString() +
      "</strong> of " +
      total.toLocaleString() +
      " samples (" +
      ((100 * inRange) / (total || 1)).toFixed(1) +
      "%) have CRoMa between <strong>" +
      lo.toFixed(2) +
      "</strong> and <strong>" +
      hi.toFixed(2) +
      "</strong>.";
    return box;
  }

  /* ------------------------------------------------------------------- helpers */

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
