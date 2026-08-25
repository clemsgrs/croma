const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const { FakeDocument } = require("./fake-dom.cjs");
const root = path.resolve(__dirname, "../..");
const explorer = require(path.join(root, "docs/_static/nipd-explorer.js"));

function result(nipd, mean) {
  return {
    baseline_balanced_accuracy: 0.8,
    baseline_skill: 0.3,
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
    regimes: {
      id: result(idNipd, options.idMean || [0, 0.2, -0.2]),
      ood: result(oodNipd, options.oodMean || [0, -0.1, -0.3]),
    },
  };
}

function publicationFixture() {
  return {
    schema_version: 1,
    provenance: { interval: "paired-repeat Student-t 95%" },
    cohorts: [
      {
        slug: "camelyon", label: "Camelyon", chance: 0.5, cramers_v: [0, 0.5, 1],
        models: [
          model("Atlas", -0.1, -0.15),
          model("Borealis", -0.3, -0.05),
          model("DINOv2-B", -0.2, -0.25, { control: true }),
        ],
      },
      {
        slug: "tcga-4x4", label: "TCGA-4×4", chance: 0.25, cramers_v: [0, 0.5, 1],
        models: [model("Atlas", -0.4, -0.35), model("Cygnus", 0.1, 0.05)],
      },
      {
        slug: "tolkach-esca", label: "Tolkach-ESCA", chance: 1 / 6,
        cramers_v: [0, 0.5, 1],
        models: [model("Atlas", -0.2, -0.15), model("Draco", -0.1, -0.05)],
      },
      {
        slug: "pcabiop", label: "PCaBiop", chance: 0.5, cramers_v: [0, 0.5, 1],
        models: [model("Prism", -0.05, 0.05), model("Quartz", -0.2, -0.1)],
      },
    ],
  };
}

test("opens with exactly four publication cohorts and ID selected", () => {
  const state = explorer.createView(publicationFixture()).snapshot();
  assert.deepEqual(state.cohorts.map(({ slug }) => slug), [
    "camelyon", "tcga-4x4", "tolkach-esca", "pcabiop",
  ]);
  assert.equal(state.cohort.slug, "camelyon");
  assert.equal(state.regime, "id");
});

test("orders pathology models by descending nIPD without the control", () => {
  const state = explorer.createView(publicationFixture()).snapshot();
  assert.deepEqual(state.pathology.map(({ model: name }) => name), ["Atlas", "Borealis"]);
  assert.equal(state.selected.model, "Atlas");
});

test("separates the natural-image control from pathology ordering", () => {
  const state = explorer.createView(publicationFixture()).snapshot();
  assert.deepEqual(state.control.map(({ model: name }) => name), ["DINOv2-B"]);
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
  const document = new FakeDocument();
  const mount = document.createElement("div");
  mount.className = "croma-nipd-explorer";
  mount.dataset.payload = "nipd.json";
  document.body.appendChild(mount);
  global.document = document;
  let requestedUrl;
  global.fetch = async (url) => {
    requestedUrl = url;
    return { ok: true, json: async () => publicationFixture() };
  };

  explorer.boot(document);
  await new Promise(setImmediate);
  await new Promise(setImmediate);

  assert.equal(mount.querySelectorAll("select").length, 2);
  assert.equal(requestedUrl, "https://example.test/nipd.json");
  assert.equal(mount.querySelectorAll(".croma-nipd-model").length, 3);
  mount.querySelectorAll(".croma-nipd-model")[1].dispatch("keydown", { key: "Enter" });
  assert.match(document.activeElement.getAttribute("aria-label"), /^Borealis, nIPD/);

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
