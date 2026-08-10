/* Interactive Pareto panels, drawn from the committed results/ CSVs.
 *
 * Each `div.croma-pareto` embed names its data with `data-cohort="<slug>"` (median CRoMa
 * against LTM from results/<slug>.csv) or `data-kind="rank"` (mean CRoMa rank against mean
 * tail rank from cross_benchmark.csv). The frontier is ringed; every point is anonymous
 * until hovered or focused, when a tooltip names it with its two values -- at 25 encoders a legend or a fully labelled static panel is harder to read
 * than "point at, get the name". The CSVs are fetched, not inlined, because html_extra_path
 * already publishes results/ at the site root; the panel and the tables beside it therefore
 * come from the same committed export (ADR-0016).
 *
 * Theming rides furo's CSS variables via classes in custom.css, so one render serves both
 * themes. No dependencies, like the explorer.
 */
(function () {
  "use strict";

  var SCRIPT_SRC = document.currentScript && document.currentScript.src;
  var NS = "http://www.w3.org/2000/svg";

  // One shared tooltip; panels move and fill it.
  var tip = null;

  function boot() {
    var embeds = document.querySelectorAll(".croma-pareto");
    if (!embeds.length || !SCRIPT_SRC) return;
    embeds.forEach(function (el) {
      var rank = el.dataset.kind === "rank";
      var file = rank ? "cross_benchmark.csv" : el.dataset.cohort + ".csv";
      fetch(new URL("../" + file, SCRIPT_SRC).href)
        .then(function (response) {
          if (!response.ok) throw new Error(file + ": HTTP " + response.status);
          return response.text();
        })
        .then(function (text) {
          render(el, extract(parseCsv(text), rank), rank);
        })
        .catch(function (error) {
          el.textContent = "Pareto panel unavailable (" + error.message + ").";
        });
    });
  }

  function parseCsv(text) {
    var lines = text.trim().split(/\r?\n/);
    var head = lines[0].split(",");
    return lines.slice(1).map(function (line) {
      var cells = line.split(",");
      var row = {};
      head.forEach(function (key, i) {
        row[key] = cells[i];
      });
      return row;
    });
  }

  /* Rows as {model, x, y} in *goodness* orientation: up-right is better on both panels.
   * Rank mode negates the ranks (1 = best) and the axis labels restore the positive
   * numbers, exactly as the manuscript's panel does. The natural-image control is a
   * calibration floor, not a competitor, and is excluded everywhere. */
  function extract(rows, rank) {
    return rows
      .filter(function (r) {
        return r.is_control !== "True" && (rank ? r.croma_rank !== "" : true);
      })
      .map(function (r) {
        return {
          model: r.model,
          x: rank ? -parseFloat(r.croma_rank) : parseFloat(r.croma),
          y: rank ? -parseFloat(r.ltm_rank) : parseFloat(r.croma_ltm10),
        };
      })
      .filter(function (p) {
        return isFinite(p.x) && isFinite(p.y);
      });
  }

  function frontierOf(points) {
    return points.filter(function (p) {
      return !points.some(function (q) {
        return (q.x > p.x && q.y >= p.y) || (q.x >= p.x && q.y > p.y);
      });
    });
  }

  function ticks(lo, hi, target) {
    var span = hi - lo;
    var step = Math.pow(10, Math.floor(Math.log10(span / target)));
    [5, 2, 1].forEach(function (m) {
      if (span / (step * m) >= target) step = step * m;
    });
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + step / 1e6; v += step) {
      out.push(Math.abs(v) < step / 1e6 ? 0 : v);
    }
    return out;
  }

  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name);
    Object.keys(attrs).forEach(function (k) {
      node.setAttribute(k, attrs[k]);
    });
    if (parent) parent.appendChild(node);
    return node;
  }

  function fmt(value, rank) {
    return rank ? Math.abs(value).toFixed(1) : value.toFixed(2);
  }

  function render(container, points, rank) {
    if (!points.length) return;
    var W = 430;
    var H = 430;
    var m = { l: 52, r: 16, t: 14, b: 44 };

    var xs = points.map(function (p) { return p.x; });
    var ys = points.map(function (p) { return p.y; });
    var xmin = Math.min.apply(null, xs);
    var xmax = Math.max.apply(null, xs);
    var ymin = Math.min.apply(null, ys);
    var ymax = Math.max.apply(null, ys);
    if (!rank) {
      // Keep both zero references in view, as the static panels did.
      xmin = Math.min(xmin, 0);
      xmax = Math.max(xmax, 0);
      ymax = Math.max(ymax, 0);
    }
    var xpad = (xmax - xmin) * 0.07 || 1;
    var ypad = (ymax - ymin) * 0.07 || 1;
    xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad;

    function sx(v) { return m.l + ((v - xmin) / (xmax - xmin)) * (W - m.l - m.r); }
    function sy(v) { return H - m.b - ((v - ymin) / (ymax - ymin)) * (H - m.t - m.b); }

    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H,
      class: "croma-pareto-svg",
      role: "img",
      "aria-label": rank
        ? "Mean CRoMa rank against mean tail rank, one point per encoder."
        : "Median CRoMa against lower-tail mean, one point per encoder.",
    });

    // Grid + ticks. Rank ticks restore the positive rank numbers.
    ticks(xmin, xmax, 6).forEach(function (v) {
      el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "croma-pareto-grid" }, svg);
      var t = el("text", { x: sx(v), y: H - m.b + 16, class: "croma-pareto-tick", "text-anchor": "middle" }, svg);
      t.textContent = rank ? String(Math.abs(v)) : String(Math.round(v * 100) / 100);
    });
    ticks(ymin, ymax, 6).forEach(function (v) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "croma-pareto-grid" }, svg);
      var t = el("text", { x: m.l - 7, y: sy(v) + 3.5, class: "croma-pareto-tick", "text-anchor": "end" }, svg);
      t.textContent = rank ? String(Math.abs(v)) : String(Math.round(v * 100) / 100);
    });
    el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "croma-pareto-frame" }, svg);

    // Dashed zero references (cohort panels only; ranks have no zero).
    if (!rank) {
      el("line", { x1: sx(0), x2: sx(0), y1: m.t, y2: H - m.b, class: "croma-pareto-ref" }, svg);
      el("line", { x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0), class: "croma-pareto-ref" }, svg);
    }

    // Frontier staircase and the shaded dominated region below-left of it.
    var frontier = frontierOf(points).sort(function (a, b) { return a.x - b.x; });
    var steps = [[m.l, sy(frontier[0].y)]];
    frontier.forEach(function (p, i) {
      steps.push([sx(p.x), sy(p.y)]);
      if (i + 1 < frontier.length) steps.push([sx(p.x), sy(frontier[i + 1].y)]);
    });
    steps.push([sx(frontier[frontier.length - 1].x), H - m.b]);
    var path = steps.map(function (p, i) { return (i ? "L" : "M") + p[0] + " " + p[1]; }).join("");
    el("path", { d: path + "L" + m.l + " " + (H - m.b) + "Z", class: "croma-pareto-shade" }, svg);
    el("path", { d: path, class: "croma-pareto-stair" }, svg);

    // Axis labels.
    var xl = el("text", { x: (m.l + W - m.r) / 2, y: H - 8, class: "croma-pareto-axis", "text-anchor": "middle" }, svg);
    xl.textContent = rank ? "Mean rank by median CRoMa  (1 = best)" : "Median CRoMa  (typical case)";
    var yl = el("text", {
      x: 14, y: (m.t + H - m.b) / 2, class: "croma-pareto-axis", "text-anchor": "middle",
      transform: "rotate(-90 14 " + (m.t + H - m.b) / 2 + ")",
    }, svg);
    yl.textContent = rank ? "Mean rank by tail severity LTM10  (1 = best)" : "Tail severity  LTM10";

    // Points: frontier ringed; every point named on hover/focus.
    var inFrontier = {};
    frontier.forEach(function (p) { inFrontier[p.model] = true; });
    points.forEach(function (p) {
      var g = el("g", { class: "croma-pareto-pt", tabindex: "0" }, svg);
      if (inFrontier[p.model]) {
        el("circle", { cx: sx(p.x), cy: sy(p.y), r: 9, class: "croma-pareto-ring" }, g);
      }
      el("circle", {
        cx: sx(p.x), cy: sy(p.y), r: 5,
        class: "croma-pareto-dot" + (inFrontier[p.model] ? " croma-pareto-dot-frontier" : ""),
      }, g);
      var title = document.createElementNS(NS, "title");
      title.textContent = p.model;
      g.appendChild(title);
      attachHover(g, container, p, rank);
    });
    container.textContent = "";
    container.appendChild(svg);
  }

  function attachHover(g, container, p, rank) {
    function show() {
      if (!tip) {
        tip = document.createElement("div");
        tip.className = "croma-pareto-tip";
        document.body.appendChild(tip);
      }
      var strong = document.createElement("strong");
      strong.textContent = p.model;
      tip.textContent = "";
      tip.appendChild(strong);
      tip.appendChild(document.createElement("br"));
      tip.appendChild(
        document.createTextNode(
          rank
            ? "CRoMa rank " + fmt(p.x, true) + " · tail rank " + fmt(p.y, true)
            : "median " + fmt(p.x) + " · LTM₁₀ " + fmt(p.y)
        )
      );
      var dot = g.querySelector(".croma-pareto-dot");
      var box = dot.getBoundingClientRect();
      tip.style.left = window.scrollX + box.left + box.width / 2 + "px";
      tip.style.top = window.scrollY + box.top - 8 + "px";
      tip.style.display = "block";
      g.classList.add("croma-pareto-pt-active");
    }
    function hide() {
      if (tip) tip.style.display = "none";
      g.classList.remove("croma-pareto-pt-active");
    }
    g.addEventListener("mouseenter", show);
    g.addEventListener("mouseleave", hide);
    g.addEventListener("focus", show);
    g.addEventListener("blur", hide);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
