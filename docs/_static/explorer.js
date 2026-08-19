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
 * loaded on demand, so pages without the widget pay nothing but this file. The curves are
 * those bins smoothed with the manuscript figures' kernel (Gaussian, Scott's bandwidth);
 * the brush readout counts the raw bins, never the smoothed curve.
 *
 * Every element with class `croma-explorer` becomes an independent instance. Each mount
 * may carry `data-panel="tile"` or `data-panel="slide"` to restrict its cohort list to
 * one roster -- the tile and slide panels are scored on different encoder rosters, and
 * the pages promise never to blend them, so the filter lives here rather than in prose.
 * A mount may instead carry `data-cohort="<slug>"` to pin itself to a single cohort --
 * the cohort pages' mounts -- which, being one-cohort, also gets no cohort dropdown.
 *
 * Per-sample identifiers and tile thumbnails are deliberately absent. Identifiers would add
 * megabytes per cohort for a lookup nobody can act on without the cohort in hand, and
 * thumbnails would mean redistributing the datasets under their different licences.
 */
(function () {
  "use strict";

  /* Read synchronously, while the script is still executing: `document.currentScript` is
     null by the time DOMContentLoaded fires, and Sphinx injects this file into <head>, so
     the elements it mounts on do not exist yet either. */
  var SCRIPT_SRC = document.currentScript && document.currentScript.src;

  var PAD = { top: 8, right: 8, bottom: 40, left: 44 };
  var WIDTH = 640;
  var HEIGHT = 234;
  var ROW_WIDTH = 480;
  var ROW_HEIGHT = 26;

  /* Distinguishes the clipPath ids of coexisting plots (two mounts, re-renders). */
  var uid = 0;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  function boot() {
    var containers = document.querySelectorAll(".croma-explorer");
    if (!containers.length || !SCRIPT_SRC) return;

    fetch(new URL("../distributions.json", SCRIPT_SRC).href)
      .then(function (response) {
        if (!response.ok) throw new Error(response.status + " " + response.statusText);
        return response.json();
      })
      .then(function (data) {
        containers.forEach(function (container) {
          mount(container, data);
        });
      })
      .catch(function (error) {
        containers.forEach(function (container) {
          container.textContent =
            "The distribution data could not be loaded (" + error.message + ").";
        });
      });
  }

  /* One explorer instance: its own cohort list, selection and DOM, so two mounts on one
     build (the tile panel's and the slide panel's) never share state. */
  function mount(container, data) {
    var panel = container.dataset.panel || null;
    var only = container.dataset.cohort || null;
    var slugs = Object.keys(data.cohorts).filter(function (slug) {
      if (only) return slug === only;
      /* Cohorts without a panel field predate the slide panel and are tile-roster. */
      return !panel || (data.cohorts[slug].panel || "tile") === panel;
    });
    if (!slugs.length) {
      container.textContent = "No cohorts in the distribution data match this page.";
      return;
    }

    var state = {
      cohort: slugs[0],
      model: null,
      compare: null,
      from: null,
      to: null,
      applySelection: null,
    };
    state.model = rankedModels(data.cohorts[state.cohort])[0];
    render();

    /* ---------------------------------------------------------------- ordering */

    /* Encoders in the result tables' order: by median CRoMa, descending -- the tables sort
       by the median, so the overview must agree with the page it sits on. The binned median
       is read off the histogram's CDF. The payload is keyed by name with no ordering. */
    function rankedModels(cohort) {
      var nBins = data.n_bins;
      return Object.keys(cohort.models).sort(function (a, b) {
        return (
          histogramMedian(cohort, cohort.models[b], nBins) -
            histogramMedian(cohort, cohort.models[a], nBins) ||
          a.localeCompare(b)
        );
      });
    }

    /* -------------------------------------------------------------- rendering */

    function render() {
      var cohort = data.cohorts[state.cohort];
      if (state.compare === state.model) state.compare = null;

      container.innerHTML = "";
      var bar = controls(cohort);
      container.appendChild(bar);
      container.appendChild(overview(cohort));

      var counts = cohort.models[state.model];
      var compareCounts = state.compare ? cohort.models[state.compare] : null;
      var view = {
        cohort: cohort,
        counts: counts,
        compareCounts: compareCounts,
        nBins: data.n_bins,
      };
      container.appendChild(detailHeading());
      var svg = histogram(cohort, counts, compareCounts, data.n_bins, view);
      container.appendChild(svg);
      view.readout = el("p", "croma-explorer-readout");
      container.appendChild(view.readout);
      view.reset = bar.querySelector(".croma-explorer-reset");

      /* Repaint the selection on the DOM that is already there. A full render() inside the
         drag would destroy the SVG holding the pointer capture -- the bug that reduced the
         brush to single-bin clicks. The selection is a clip window over a full-colour copy
         of the curve, revealed above the dimmed base copy. */
      state.applySelection = function () {
        var selection = selectedBins(view.nBins);
        var binWidth = view.plotWidth / view.nBins;
        view.base.setAttribute(
          "class",
          "croma-explorer-area" + (selection ? " is-dimmed" : "")
        );
        if (selection) {
          view.clipRect.setAttribute("x", PAD.left + selection.lo * binWidth);
          view.clipRect.setAttribute("width", (selection.hi - selection.lo + 1) * binWidth);
        } else {
          view.clipRect.setAttribute("width", 0);
        }
        view.reset.disabled = selection === null;
        writeReadout(view);
      };
      state.applySelection();
    }

    function controls(cohort) {
      var bar = el("div", "croma-explorer-controls");

      /* A one-cohort mount (the slide panel today) gets no cohort dropdown: a select with
         a single option is a control that cannot control anything. */
      if (slugs.length > 1) {
        var cohortSelect = el("select");
        cohortSelect.setAttribute("aria-label", "Cohort");
        slugs.forEach(function (slug) {
          var option = el("option");
          option.value = slug;
          option.textContent = data.cohorts[slug].label;
          if (slug === state.cohort) option.selected = true;
          cohortSelect.appendChild(option);
        });
        cohortSelect.addEventListener("change", function () {
          state.cohort = cohortSelect.value;
          var next = data.cohorts[state.cohort];
          /* Keep the encoders across a cohort switch when they were scored there -- the
             point of switching is usually to follow a model. The brush is cleared because
             the axis domain changes with the cohort. */
          if (!next.models[state.model]) state.model = rankedModels(next)[0];
          if (state.compare && !next.models[state.compare]) state.compare = null;
          state.from = state.to = null;
          render();
        });
        bar.appendChild(labelled("Cohort", cohortSelect));
      }

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
      var nBins = data.n_bins;
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

      var smooth = smoothedCounts(counts);
      var peak = Math.max.apply(null, smooth) || 1;
      svg.appendChild(
        svgEl("path", {
          d: areaPath(smooth, nBins, peak, ROW_WIDTH, ROW_HEIGHT, 0, 0),
          class: "croma-explorer-area",
        })
      );

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
      /* In a comparison each name takes its distribution's hue -- the detailed encoder the
         bars' brand colour, the overlay the compare colour. Solo, the name stays plain. */
      var name = el("strong", state.compare ? "croma-explorer-model-name" : null);
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
      var smooth = smoothedCounts(counts);
      var compareSmooth = compareCounts ? smoothedCounts(compareCounts) : null;
      /* One density scale for both encoders, so the shapes are comparable. */
      var peak =
        Math.max.apply(null, compareSmooth ? smooth.concat(compareSmooth) : smooth) || 1;

      var svg = svgEl("svg", {
        viewBox: "0 0 " + WIDTH + " " + HEIGHT,
        class: "croma-explorer-plot",
        role: "img",
        "aria-label":
          "Per-sample CRoMa distribution for " +
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

      var shape = areaPath(smooth, nBins, peak, plotWidth, plotHeight, PAD.left, PAD.top);
      var base = svgEl("path", { d: shape, class: "croma-explorer-area" });
      svg.appendChild(base);

      /* The selection: a clip window revealing a full-colour copy of the curve (over a
         background-coloured mask, so the dimmed base copy does not blend through the
         translucent fill) while the base copy is dimmed. Width 0 hides it. */
      var clipId = "croma-explorer-clip-" + ++uid;
      var clip = svgEl("clipPath", { id: clipId });
      var clipRect = svgEl("rect", { x: PAD.left, y: 0, width: 0, height: HEIGHT });
      clip.appendChild(clipRect);
      svg.appendChild(clip);
      var selected = svgEl("g", { "clip-path": "url(#" + clipId + ")" });
      selected.appendChild(svgEl("path", { d: shape, class: "croma-explorer-area-mask" }));
      selected.appendChild(svgEl("path", { d: shape, class: "croma-explorer-area" }));
      svg.appendChild(selected);

      view.plotWidth = plotWidth;
      view.base = base;
      view.clipRect = clipRect;

      /* Drawn above the selection layer, so its translucent fill tints the highlight
         rather than vanishing under the highlight's background mask. */
      if (compareSmooth) {
        svg.appendChild(
          svgEl("path", {
            d: areaPath(compareSmooth, nBins, peak, plotWidth, plotHeight, PAD.left, PAD.top),
            class: "croma-explorer-area is-compare",
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
          y: PAD.top + plotHeight + 16,
          class: "croma-explorer-tick",
          "text-anchor": "middle",
        });
        tick.textContent = value.toFixed(1);
        svg.appendChild(tick);
      });

      var axisTitle = svgEl("text", {
        x: PAD.left + plotWidth / 2,
        y: HEIGHT - 6,
        class: "croma-explorer-axis-title",
        "text-anchor": "middle",
      });
      axisTitle.textContent = "CRoMa";
      svg.appendChild(axisTitle);

      attachBrush(svg, plotWidth, nBins);
      return svg;
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
          " samples. Drag across the distribution to count a range.";
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
        rangeCount(
          state.model,
          view.counts,
          selection,
          view.compareCounts ? "croma-explorer-model-name" : null
        );
      if (view.compareCounts) {
        html +=
          " &middot; " +
          rangeCount(
            state.compare,
            view.compareCounts,
            selection,
            "croma-explorer-compare-name"
          );
      }
      view.readout.innerHTML = html;
    }
  }

  /* The manuscript's ridgelines are Gaussian KDEs at Scott's bandwidth on the raw
     samples. The payload carries 200-bin counts instead of samples, but a KDE is the
     histogram convolved with its kernel, and Scott's bandwidth needs only n and the
     standard deviation -- both recoverable from the bins. Display-only: the brush
     readout keeps counting the raw bins. */
  function smoothedCounts(counts) {
    var n = sum(counts);
    if (!n) return counts.slice();
    var mean = 0;
    for (var i = 0; i < counts.length; i++) mean += counts[i] * (i + 0.5);
    mean /= n;
    var variance = 0;
    for (i = 0; i < counts.length; i++) {
      variance += counts[i] * (i + 0.5 - mean) * (i + 0.5 - mean);
    }
    /* Scott's rule for one dimension, sigma * n^(-1/5), already in bin units. */
    var sigma = Math.sqrt(variance / n) * Math.pow(n, -0.2);
    if (sigma < 0.5) return counts.slice();
    var radius = Math.ceil(3 * sigma);
    var kernel = [];
    var mass = 0;
    for (var k = -radius; k <= radius; k++) {
      var weight = Math.exp(-(k * k) / (2 * sigma * sigma));
      kernel.push(weight);
      mass += weight;
    }
    var out = [];
    for (var j = 0; j < counts.length; j++) {
      var acc = 0;
      for (var m = -radius; m <= radius; m++) {
        var index = j + m;
        if (index >= 0 && index < counts.length) acc += counts[index] * kernel[m + radius];
      }
      out.push(acc / mass);
    }
    return out;
  }

  function curvePoints(counts, nBins, peak, width, height, left, top) {
    var binWidth = width / nBins;
    var baseline = top + height;
    var points = [];
    for (var i = 0; i < nBins; i++) {
      points.push(
        (left + (i + 0.5) * binWidth).toFixed(2) +
          " " +
          (baseline - (counts[i] / peak) * height).toFixed(2)
      );
    }
    return points;
  }

  /* The encoder's curve closed to the baseline, for the filled detail and mini shapes. */
  function areaPath(counts, nBins, peak, width, height, left, top) {
    var baseline = (top + height).toFixed(2);
    return (
      "M" + left + " " + baseline +
      " L" + curvePoints(counts, nBins, peak, width, height, left, top).join(" L") +
      " L" + (left + width) + " " + baseline +
      " Z"
    );
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

  /* `colorClass`, when given, ties the name to its distribution's hue in a comparison. */
  function rangeCount(name, counts, selection, colorClass) {
    var total = sum(counts);
    var inRange = 0;
    for (var i = selection.lo; i <= selection.hi; i++) inRange += counts[i];
    return (
      (colorClass ? '<span class="' + colorClass + '">' + name + "</span>" : name) +
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
