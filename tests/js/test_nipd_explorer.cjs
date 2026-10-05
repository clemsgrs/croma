const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const { FakeDocument } = require("./fake-dom.cjs");
const root = path.resolve(__dirname, "../..");
const explorer = require(path.join(root, "docs/_static/nipd-explorer.js"));

function result(nipd, mean, baseline = 0.8) {
  return {
    baseline_balanced_accuracy: baseline,
    mean_normalized_trajectory: mean,
    ci95_low: mean.map((value) => value - 0.1),
    ci95_high: mean.map((value) => value + 0.1),
    nipd,
  };
}

function model(name, idNipd, oodNipd, options = {}) {
  return {
    model: name,
    ranked: !options.control,
    is_control: Boolean(options.control),
    croma_median_m5: options.croma ?? 0,
    regimes: {
      id: result(idNipd, options.idMean || [0, 0.2, -0.2], options.idBaseline),
      ood: result(oodNipd, options.oodMean || [0, -0.1, -0.3]),
    },
  };
}

function associationMetadata({
  n = 2, idRho = 0.5, oodRho = -0.5, descriptive = false,
  slope = 0.2368421053, intercept = -0.2157894737,
} = {}) {
  const trend = descriptive ? null : { slope, intercept, x_min: -0.2, x_max: 0.3 };
  return {
    descriptive,
    regimes: {
      id: { n, spearman_rho: idRho, trend },
      ood: { n, spearman_rho: oodRho, trend },
    },
  };
}

function publicationFixture() {
  return {
    schema_version: 2,
    provenance: { interval: "paired-repeat Student-t 95%" },
    cohorts: [
      {
        slug: "camelyon", label: "Camelyon", chance: 0.5, cramers_v: [0, 0.5, 1],
        association: associationMetadata({ n: 3 }),
        models: [
          model("Atlas", -0.1, -0.15, { croma: 0.1 }),
          model("Borealis", -0.3, -0.05, { croma: -0.2, idBaseline: 0.75 }),
          model("Cygnus", -0.2, -0.35, { croma: 0.3, idMean: [0, 0.2, -0.97] }),
          model("DINOv2-B", -0.2, -0.25, { control: true, croma: 0.4 }),
        ],
      },
      {
        slug: "tcga-4x4", label: "TCGA-4×4", chance: 0.25, cramers_v: [0, 0.5, 1],
        association: associationMetadata(),
        models: [model("Atlas", -0.4, -0.35), model("Cygnus", 0.1, 0.05)],
      },
      {
        slug: "tolkach-esca", label: "Tolkach-ESCA", chance: 1 / 6,
        cramers_v: [0, 0.5, 1],
        association: associationMetadata(),
        models: [model("Atlas", -0.2, -0.15), model("Draco", -0.1, -0.05)],
      },
      {
        slug: "pcabiop", label: "PCaBiop", chance: 0.5, cramers_v: [0, 0.5, 1],
        association: associationMetadata({ descriptive: true }),
        models: [model("Prism", -0.05, 0.05), model("Quartz", -0.2, -0.1)],
      },
    ],
  };
}

async function bootFixture(fetchImplementation) {
  const document = new FakeDocument();
  const mount = document.createElement("div");
  mount.className = "croma-nipd-explorer";
  mount.dataset.payload = "nipd.json";
  document.body.appendChild(mount);
  global.document = document;
  global.fetch = fetchImplementation || (async () => ({
    ok: true, json: async () => publicationFixture(),
  }));
  explorer.boot(document);
  await new Promise(setImmediate);
  await new Promise(setImmediate);
  return { document, mount };
}

function addBorealisComparison(mount) {
  const comparison = mount.querySelector('[aria-label="Comparison model"]');
  comparison.value = "Borealis";
  comparison.dispatch("change");
}

test("opens with exactly four publication cohorts and ID selected", () => {
  const state = explorer.createView(publicationFixture()).snapshot();
  assert.deepEqual(state.cohorts.map(({ slug }) => slug), [
    "camelyon", "tcga-4x4", "tolkach-esca", "pcabiop",
  ]);
  assert.equal(state.cohort.slug, "camelyon");
  assert.equal(state.regime, "id");
});

test("orders pathology models by descending endpoint without the control", () => {
  const state = explorer.createView(publicationFixture()).snapshot();
  // By nIPD this would read Atlas, Cygnus, Borealis. Cygnus ends at -97% and drops
  // behind Borealis, which ends at -20% -- the ranking is the endpoint, not the area.
  assert.deepEqual(state.pathology.map(({ model: name }) => name), ["Atlas", "Borealis", "Cygnus"]);
  assert.equal(state.selected.model, "Atlas");
});

test("separates the natural-image control from pathology ordering", () => {
  const state = explorer.createView(publicationFixture()).snapshot();
  assert.deepEqual(state.control.map(({ model: name }) => name), ["DINOv2-B"]);
});

test("association is scoped to the active cohort and regime with ranked-only statistics", () => {
  const view = explorer.createView(publicationFixture());
  let association = view.snapshot().association;
  assert.deepEqual(association.pathology.map(({ model: name, x, y }) => [name, x, y]), [
    ["Atlas", 0.1, -0.1], ["Borealis", -0.2, -0.3], ["Cygnus", 0.3, -0.2],
  ]);
  assert.deepEqual(association.control.map(({ model: name }) => name), ["DINOv2-B"]);
  assert.equal(association.n, 3);
  assert.ok(Math.abs(association.rho - 0.5) < 1e-12);
  assert.equal(association.trend.slope, 0.2368421053);
  assert.equal(association.trend.intercept, -0.2157894737);

  view.setContext("camelyon", "ood");
  association = view.snapshot().association;
  assert.deepEqual(association.pathology.map(({ model: name, y }) => [name, y]), [
    ["Atlas", -0.15], ["Borealis", -0.05], ["Cygnus", -0.35],
  ]);
});

test("real publication association statistics match the manuscript analysis", () => {
  const payload = require(path.join(root, "results/nipd.json"));
  const view = explorer.createView(payload);
  const expected = {
    "camelyon/id": [26, 0.945982906, 0.3039544444],
    "camelyon/ood": [26, 0.7442735043, 0.1144040624],
    "tcga-4x4/id": [26, 0.9042735043, 0.2402571277],
    "tcga-4x4/ood": [26, 0.8851282051, 0.3945676619],
    "tolkach-esca/id": [26, 0.9309401709, 0.0957444143],
    "tolkach-esca/ood": [26, 0.7285470085, 0.0285616383],
    "pcabiop/id": [5, 0.9, null],
    "pcabiop/ood": [5, 0.6, null],
  };
  for (const [scope, [n, rho, slope]] of Object.entries(expected)) {
    const [cohort, regime] = scope.split("/");
    view.setContext(cohort, regime);
    const association = view.snapshot().association;
    assert.equal(association.n, n, scope);
    assert.ok(Math.abs(association.rho - rho) < 1e-12, scope);
    assert.equal(association.descriptive, cohort === "pcabiop", scope);
    if (slope === null) assert.equal(association.trend, null, scope);
    else assert.ok(Math.abs(association.trend.slope - slope) < 1e-12, scope);
  }
});

test("context changes preserve a model present at the destination", () => {
  const view = explorer.createView(publicationFixture());
  view.select("Atlas");
  view.setContext("tcga-4x4", "ood");
  assert.equal(view.snapshot().selected.model, "Atlas");
});

test("context changes fall back to the destination's highest-nIPD pathology model", () => {
  const view = explorer.createView(publicationFixture());
  view.select("DINOv2-B");
  view.setContext("pcabiop", "id");
  assert.equal(view.snapshot().selected.model, "Prism");
});

test("selects one optional comparison from the active context", () => {
  const view = explorer.createView(publicationFixture());
  view.compare("Borealis");
  assert.equal(view.snapshot().comparison.model, "Borealis");
});

test("self comparison clears the optional comparison", () => {
  const view = explorer.createView(publicationFixture());
  view.compare("Borealis");
  view.compare("Atlas");
  assert.equal(view.snapshot().comparison, null);
});

test("invalid comparison is rejected without changing the valid pair", () => {
  const view = explorer.createView(publicationFixture());
  view.compare("Borealis");
  assert.throws(() => view.compare("Not published"), /Unknown comparison model/);
  assert.equal(view.snapshot().comparison.model, "Borealis");
});

test("focus swap keeps the pair and transfers the active model", () => {
  const view = explorer.createView(publicationFixture());
  view.compare("Borealis");
  view.swapFocus();
  assert.equal(view.snapshot().selected.model, "Borealis");
  assert.equal(view.snapshot().comparison.model, "Atlas");
});

test("selecting the comparison as primary swaps focus without duplicating it", () => {
  const view = explorer.createView(publicationFixture());
  view.compare("Borealis");
  view.select("Borealis");
  assert.equal(view.snapshot().selected.model, "Borealis");
  assert.equal(view.snapshot().comparison.model, "Atlas");
});

test("context changes preserve an available comparison and clear an unavailable one", () => {
  const view = explorer.createView(publicationFixture());
  view.compare("Borealis");
  view.setContext("camelyon", "ood");
  assert.equal(view.snapshot().comparison.model, "Borealis");
  view.setContext("tcga-4x4", "ood");
  assert.equal(view.snapshot().comparison, null);
  view.setContext("camelyon", "id");
  assert.equal(view.snapshot().comparison, null);
});

test("trajectory scale is fixed across model selection in one context", () => {
  const view = explorer.createView(publicationFixture());
  const domain = view.snapshot().yDomain;
  view.select("Borealis");
  assert.deepEqual(view.snapshot().yDomain, domain);
});

test("a zero crossing is split into labelled positive and negative lobes", () => {
  assert.deepEqual(explorer.splitSignSegments([0, 0.5, 1], [0.2, -0.2, 0.4]), [
    { sign: "positive", points: [[0, 0.2], [0.25, 0]] },
    { sign: "negative", points: [[0.25, 0], [0.5, -0.2], [2 / 3, 0]] },
    { sign: "positive", points: [[2 / 3, 0], [1, 0.4]] },
  ]);
});

test("sample inspection returns the selected committed point and interval", () => {
  assert.deepEqual(explorer.createView(publicationFixture()).inspectPoint(1), {
    model: "Atlas", regime: "ID", cramersV: 0.5, mean: 0.2,
    low: 0.1, high: 0.30000000000000004, interval: "paired-repeat Student-t 95%",
  });
});

test("boot renders controls and keyboard interaction updates focus and inspection", async () => {
  let requestedUrl;
  const { document, mount } = await bootFixture(async (url) => {
    requestedUrl = url;
    return { ok: true, json: async () => publicationFixture() };
  });

  assert.equal(mount.querySelectorAll("select").length, 3);
  assert.equal(requestedUrl, "https://example.test/nipd.json");
  assert.equal(mount.querySelector(".croma-nipd-association-plot")
    .getAttribute("aria-label"), "CRoMa and nIPD association");
  assert.match(mount.querySelector(".croma-nipd-association-description").textContent,
    /least-squares trend over ranked pathology encoders has slope 0\.237/);
  assert.equal(mount.querySelectorAll(".croma-nipd-model").length, 4);
  mount.querySelectorAll(".croma-nipd-model")[1].dispatch("keydown", { key: "Enter" });
  assert.match(document.activeElement.getAttribute("aria-label"), /^Borealis, nIPD/);
  assert.match(mount.querySelector(".croma-nipd-association-point.is-selected")
    .getAttribute("aria-label"), /^Borealis, pathology encoder/);
  assert.equal(mount.querySelector(".croma-nipd-association-point.is-selected")
    .getAttribute("aria-pressed"), "true");

  mount.querySelectorAll(".croma-nipd-association-point")[2]
    .dispatch("keydown", { key: "Enter" });
  assert.match(mount.querySelector("#croma-nipd-active-heading")?.textContent ||
    mount.querySelector(".croma-nipd-detail").textContent, /Cygnus normalized performance change/);
  assert.match(document.activeElement.getAttribute("aria-label"), /^Cygnus, pathology encoder/);

  mount.querySelectorAll(".croma-nipd-sample")[1].dispatch("keydown", { key: " " });
  assert.match(mount.querySelector(".croma-nipd-inspection").textContent, /Cramér's V 0.500/);

  const positiveLobe = mount.querySelector(".croma-nipd-lobe.is-positive");
  assert.equal(positiveLobe.getAttribute("fill"), "url(#nipd-positive)");

  const regime = mount.querySelectorAll("select")[1];
  regime.value = "ood";
  regime.dispatch("change");
  assert.equal(document.activeElement.getAttribute("aria-label"), "Evaluation regime");
  assert.match(mount.querySelector(".croma-nipd-context").textContent, /transfer effects/);
});

test("every association point carries a name, revealed on hover", async () => {
  const { mount } = await bootFixture();
  const points = mount.querySelectorAll(".croma-nipd-association-point");
  assert.deepEqual(
    points.map((point) => point.querySelector(".croma-nipd-association-label").textContent),
    ["Atlas", "Borealis", "Cygnus", "DINOv2-B"],
  );
  // Only the selected model and the control keep their name on permanently.
  assert.deepEqual(
    points.map((point) =>
      point.querySelector(".croma-nipd-association-label").classList.contains("is-pinned")),
    [true, false, false, true],
  );

  const inspection = mount.querySelector(".croma-nipd-association-inspection");
  assert.match(inspection.textContent, /^Atlas,/);
  points[1].dispatch("mouseenter");
  assert.match(inspection.textContent, /^Borealis, pathology encoder, median CRoMa -0.200, nIPD -30.0%/);
  points[1].dispatch("mouseleave");
  assert.match(inspection.textContent, /^Atlas,/);
});

test("a curve ending at chance is flagged wherever its nIPD is shown", async () => {
  const { mount } = await bootFixture();
  const rows = mount.querySelectorAll(".croma-nipd-model");
  const [atlas, cygnus] = [rows[0], rows[2]];
  assert.match(atlas.getAttribute("aria-label"), /nIPD -10.0%, normalized change at V = 1 -20.0%,/);
  assert.match(cygnus.getAttribute("aria-label"), /normalized change at V = 1 -97.0%, collapses to chance/);
  assert.equal(atlas.querySelectorAll(".croma-nipd-value.is-collapsed").length, 0);
  assert.match(
    cygnus.querySelector(".croma-nipd-value.is-collapsed").textContent,
    /^-97.0% ≈ chance$/,
  );

  cygnus.dispatch("keydown", { key: "Enter" });
  assert.match(
    mount.querySelector(".croma-nipd-model-metrics").querySelector(".is-collapsed").textContent,
    /Normalized change at V = 1-97.0% ≈ chance/,
  );
  assert.equal(mount.querySelectorAll(".croma-nipd-floor").length, 1);
  assert.equal(mount.querySelector(".croma-nipd-floor-label").textContent, "chance level");
});

test("panels that stay far from chance draw no chance-level reference", async () => {
  const { mount } = await bootFixture();
  const regime = mount.querySelectorAll("select")[1];
  regime.value = "ood";
  regime.dispatch("change");
  assert.equal(mount.querySelectorAll(".croma-nipd-floor").length, 0);
  assert.equal(mount.querySelectorAll(".croma-nipd-value.is-collapsed").length, 0);
});

test("rendered comparison shows two series and shades only the active area", async () => {
  const { mount } = await bootFixture();
  addBorealisComparison(mount);
  assert.equal(mount.querySelectorAll("select").length, 3);
  assert.equal(mount.querySelectorAll(".croma-nipd-series").length, 2);
  assert.equal(mount.querySelectorAll(".croma-nipd-interval").length, 2);
  assert.equal(mount.querySelectorAll(".croma-nipd-mean").length, 2);
  assert.equal(mount.querySelectorAll(".croma-nipd-sample").length, 6);
  assert.equal(mount.querySelectorAll(".croma-nipd-lobe").length, 2);
  assert.equal(
    mount.querySelector(".croma-nipd-series.is-active").querySelectorAll(".croma-nipd-lobe").length,
    2,
  );
  assert.equal(
    mount.querySelector(".croma-nipd-series.is-comparison")
      .querySelectorAll(".croma-nipd-lobe").length,
    0,
  );
});

test("rendered comparison associates readouts and inspected points with each model", async () => {
  const { mount } = await bootFixture();
  addBorealisComparison(mount);
  assert.match(
    mount.querySelector(".croma-nipd-series.is-comparison").getAttribute("aria-label"),
    /Borealis/,
  );
  assert.match(
    mount.querySelector(".croma-nipd-sample.is-comparison").getAttribute("aria-label"),
    /Borealis/,
  );
  assert.deepEqual(
    mount.querySelectorAll(".croma-nipd-model-metrics").map((node) => node.getAttribute("aria-label")),
    ["Active model Atlas", "Comparison model Borealis"],
  );
  assert.match(mount.querySelectorAll(".croma-nipd-model-metrics")[0].textContent,
    /Normalized change at V = 1-20.0%Baseline balanced accuracy0.800/);
  assert.match(mount.querySelectorAll(".croma-nipd-model-metrics")[1].textContent,
    /Normalized change at V = 1-20.0%Baseline balanced accuracy0.750/);
  mount.querySelector(".croma-nipd-sample.is-comparison").dispatch("keydown", { key: "Enter" });
  assert.match(mount.querySelector(".croma-nipd-inspection").textContent, /^Borealis:/);
});

test("swap transfers active focus and readouts without clearing the pair", async () => {
  const { document, mount } = await bootFixture();
  addBorealisComparison(mount);
  mount.querySelector(".croma-nipd-swap").dispatch("click");
  assert.equal(document.activeElement.getAttribute("aria-label"), "Make Atlas the active model");
  assert.match(mount.querySelector(".croma-nipd-series.is-active").getAttribute("aria-label"), /Borealis/);
  assert.deepEqual(
    mount.querySelectorAll(".croma-nipd-model-metrics").map((node) => node.getAttribute("aria-label")),
    ["Active model Borealis", "Comparison model Atlas"],
  );
  assert.equal(mount.querySelectorAll(".croma-nipd-series").length, 2);
});
