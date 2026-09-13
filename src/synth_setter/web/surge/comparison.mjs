const element = (id) => document.getElementById(id);
const frames = new Map();
let catalog, pending = new Set(), downloadUrl;

function dispose(state) {
  clearTimeout(state.timer);
  state.iframe.remove();
  state.card.remove();
  frames.delete(state.entry.id);
}

function finish() {
  if (pending.size) return;
  const comparison = window.surgeComparison;
  comparison.state = Object.keys(comparison.errors).length ? "error" : "complete";
  element("status").textContent = comparison.state === "complete"
    ? "All selected models complete — concurrent timings are not isolated benchmarks"
    : `Comparison finished with errors: ${JSON.stringify(comparison.errors)}`;
  element("run").disabled = false;
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([JSON.stringify(comparison)], { type: "application/json" }));
  let link = element("comparison-download");
  if (!link) {
    link = document.createElement("a");
    link.id = "comparison-download";
    element("results").before(link);
  }
  link.href = downloadUrl;
  link.download = `surge-comparison-${comparison.comparisonId}.json`;
  link.textContent = "Download comparison records";
}

function renewLoadingDeadline(state) {
  clearTimeout(state.timer);
  state.timer = setTimeout(() => {
    state.failed = true;
    state.reject(new Error("Model loading made no progress for four minutes; retry to reload it"));
  }, 240000);
}

function createFrame(entry) {
  const card = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = entry.label;
  const iframe = document.createElement("iframe");
  iframe.dataset.model = entry.id;
  iframe.title = entry.label;
  iframe.style.cssText = "width:100%;height:440px;border:0";
  const source = new URL("./index.html", import.meta.url);
  source.searchParams.set("model", entry.id);
  source.searchParams.set("embedded", "1");
  iframe.src = source.href;
  const state = { entry, card, iframe, loaded: false, failed: false };
  state.ready = new Promise((resolve, reject) => {
    state.resolve = resolve;
    state.reject = reject;
    renewLoadingDeadline(state);
  });
  frames.set(entry.id, state);
  card.append(heading, iframe);
  element("results").append(card);
  return state;
}

window.addEventListener("message", ({ source, origin, data }) => {
  if (origin !== location.origin) return;
  const state = [...frames.values()].find((item) => item.iframe.contentWindow === source);
  if (!state || data?.modelId !== state.entry.id) return;
  if (data.type === "comparison-progress" && !state.loaded && !state.failed) {
    renewLoadingDeadline(state);
  } else if (data.type === "comparison-ready") {
    clearTimeout(state.timer);
    if (data.manifest.checkpointSha256 !== state.entry.checkpointSha256) {
      state.failed = true;
      state.reject(new Error("Selected checkpoint identity does not match its catalog"));
      return;
    }
    state.loaded = true;
    state.resolve();
  } else if (data.type === "comparison-error") {
    if (!state.loaded) {
      clearTimeout(state.timer);
      state.failed = true;
      state.reject(new Error(data.message));
      return;
    }
    if (!pending.has(state.entry.id)) return;
    window.surgeComparison.errors[state.entry.id] = data.message;
    pending.delete(state.entry.id);
    finish();
  } else if (data.type === "comparison-complete" && pending.has(state.entry.id)) {
    window.surgeComparison.records[state.entry.id] = data.record;
    pending.delete(state.entry.id);
    finish();
  }
  if (Number.isFinite(data.height)) state.iframe.style.height = `${Math.min(3000, Math.max(440, data.height + 16))}px`;
});

element("sketchSource").addEventListener("change", () => {
  const authored = element("sketchSource").value === "authored";
  element("author").hidden = !authored;
  element("sketchUpload").hidden = authored;
});

element("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  const selected = catalog.models.filter((entry) => element(`model-${entry.id}`).checked);
  if (!selected.length) {
    element("status").textContent = "Select at least one model";
    return;
  }
  if (element("mode").value === "sketch_only" && selected.some((entry) => !entry.sketch)) {
    element("status").textContent = "Sketch-only sampling requires selecting sketch models only";
    return;
  }
  const settings = { mode: element("mode").value, sketchSource: element("sketchSource").value,
    neuralBackend: element("webgpu").checked ? "webgpu" : "wasm" };
  for (const id of ["contentCfg", "sketchCfg", "steps", "seed", "pitch", "start", "end", "loudness", "centroid"])
    settings[id] = Number(element(id).value);
  const contentFile = element("contentAudio").files[0];
  const sketchFile = element("sketchAudio").files[0];
  if (!contentFile || (selected.some((entry) => entry.sketch) && settings.sketchSource === "extracted" && !sketchFile)) {
    element("status").textContent = "Choose the required content and sketch audio";
    return;
  }
  element("run").disabled = true;
  element("comparison-download")?.remove();
  window.surgeComparison = { state: "running", comparisonId: crypto.randomUUID(), settings, records: {}, errors: {} };
  pending = new Set(selected.map((entry) => entry.id));
  for (const state of frames.values())
    if (!pending.has(state.entry.id) || state.failed) dispose(state);
  const active = selected.map((entry) => frames.get(entry.id) ?? createFrame(entry));
  element("status").textContent = "Loading selected models before concurrent inference";
  const readiness = await Promise.allSettled(active.map((state) => state.ready));
  readiness.forEach((result, index) => {
    const state = active[index];
    if (result.status === "rejected") {
      window.surgeComparison.errors[state.entry.id] = result.reason.message;
      pending.delete(state.entry.id);
    } else {
      state.iframe.contentWindow.postMessage({ type: "comparison-run", settings, contentFile, sketchFile }, location.origin);
    }
  });
  if (pending.size) element("status").textContent = "Selected models are running concurrently";
  else finish();
});

try {
  const response = await fetch(new URL("../models/catalog.json", import.meta.url));
  if (!response.ok) throw new Error(`Model catalog HTTP ${response.status}`);
  catalog = await response.json();
  if (!Array.isArray(catalog.models) || catalog.models.length !== 4 || new Set(catalog.models.map((entry) => entry.id)).size !== 4 ||
      catalog.models.some((entry) => !/^[a-z0-9-]+$/.test(entry.id) || typeof entry.label !== "string" || typeof entry.sketch !== "boolean" || !/^[a-f0-9]{64}$/.test(entry.checkpointSha256)))
    throw new Error("Invalid four-model catalog");
  const choices = document.createElement("fieldset");
  choices.id = "models";
  const legend = document.createElement("legend");
  legend.textContent = "Models to run concurrently";
  choices.append(legend);
  for (const entry of catalog.models) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `model-${entry.id}`;
    input.checked = true;
    label.append(input, document.createTextNode(entry.label));
    choices.append(label);
  }
  element("controls").prepend(choices);
  for (const id of ["steps", "contentCfg", "sketchCfg"]) element(id).value = String(catalog.sampling[id]);
  document.body.style.maxWidth = "1800px";
  element("results").style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:20px";
  element("run").textContent = "Run selected models";
  element("run").disabled = false;
  element("status").textContent = "Ready — same seed gives matching noise within each parameter-space pair";
  window.surgeComparison = { state: "ready" };
} catch (error) {
  element("status").textContent = `Error: ${error.message}`;
  window.surgeComparison = { state: "error", message: error.message };
}
