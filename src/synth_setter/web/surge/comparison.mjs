import { gaussianNoise } from "../fdn/noise.mjs";
import { decodeStereo, encodeStereoWav } from "./audio.mjs";
import { spectrogramCanvas } from "./spectrogram.mjs";

const element = (id) => document.getElementById(id);
const runners = new Map();
let catalog;
let downloadUrl;
let resultUrls = [];
let pending = new Set();

function prepareRunners(selected) {
  const selectedIds = new Set(selected.map((entry) => entry.id));
  for (const [id, runner] of runners) {
    if (!selectedIds.has(id) || runner.failed) {
      runner.worker.terminate();
      runners.delete(id);
    }
  }
  for (const url of resultUrls) URL.revokeObjectURL(url);
  resultUrls = [];
  element("results").replaceChildren();
  return selected.map((entry) => {
    const runner = runners.get(entry.id) ?? createRunner(entry);
    runner.status.textContent = runner.loaded ? "Ready" : "Loading";
    runner.output.replaceChildren();
    element("results").append(runner.card);
    return runner;
  });
}

function resultLink(bytes, type, filename, label) {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  resultUrls.push(url);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.textContent = label;
  return link;
}

function showResult(runner, record) {
  const channels = record.audio.map((channel) => Float32Array.from(channel));
  const wav = resultLink(
    encodeStereoWav(channels, record.sampleRate),
    "audio/wav",
    `${runner.entry.id}-pred.wav`,
    `Download ${runner.entry.id}-pred.wav`,
  );
  const player = document.createElement("audio");
  player.controls = true;
  player.src = wav.href;
  const fxp = resultLink(
    Uint8Array.from(record.fxp),
    "application/octet-stream",
    `${runner.entry.id}.fxp`,
    "Download FXP patch",
  );
  const json = resultLink(
    JSON.stringify(record),
    "application/json",
    `evaluation-${runner.entry.id}-${record.neuralBackend}.json`,
    "Download evaluation record",
  );
  const patchWarning = document.createElement("p");
  patchWarning.textContent =
    "FXP saved by Surge 1.4. Surge 1.3 reports a patch-version warning and may omit newer features.";
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `${record.params.length} encoded coordinates`;
  const provenance = document.createElement("p");
  provenance.textContent = `${record.paramSpecName}; ${record.neuralBackend}; ${record.neuralInferenceMs.toFixed(1)} ms neural inference`;
  details.append(summary, provenance);
  runner.output.replaceChildren(
    player,
    wav,
    document.createTextNode(" "),
    fxp,
    document.createTextNode(" "),
    json,
    patchWarning,
    spectrogramCanvas(channels, "Prediction", record.sampleRate),
    details,
  );
}

function finish() {
  if (pending.size) return;
  const comparison = window.surgeComparison;
  comparison.state = Object.keys(comparison.errors).length
    ? "error"
    : "complete";
  element("status").textContent =
    comparison.state === "complete"
      ? "All selected models complete — concurrent timings are not isolated benchmarks"
      : `Comparison finished with errors: ${JSON.stringify(comparison.errors)}`;
  element("run").disabled = false;
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(
    new Blob([JSON.stringify(comparison)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.id = "comparison-download";
  link.href = downloadUrl;
  link.download = `surge-comparison-${comparison.comparisonId}.json`;
  link.textContent = "Download comparison records";
  element("results").before(link);
}

function renewLoadingDeadline(runner) {
  clearTimeout(runner.timer);
  runner.timer = setTimeout(() => {
    runner.failed = true;
    runner.reject(
      new Error(
        "Model loading made no progress for four minutes; retry to reload it",
      ),
    );
  }, 240000);
}

function createRunner(entry) {
  const card = document.createElement("section");
  card.className = "comparison-card";
  card.dataset.model = entry.id;
  const heading = document.createElement("h3");
  heading.textContent = entry.label;
  const status = document.createElement("p");
  status.textContent = "Starting model download…";
  const output = document.createElement("div");
  card.append(heading, status, output);
  element("results").append(card);

  const worker = new Worker(new URL("./worker.mjs", import.meta.url), {
    type: "module",
  });
  const runner = { entry, worker, card, status, output, loaded: false };
  runner.ready = new Promise((resolve, reject) => {
    runner.resolve = resolve;
    runner.reject = reject;
    renewLoadingDeadline(runner);
  });
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") {
      status.textContent = data.message;
      if (!runner.loaded) renewLoadingDeadline(runner);
    } else if (data.type === "ready") {
      clearTimeout(runner.timer);
      if (data.manifest.checkpointSha256 !== entry.checkpointSha256) {
        runner.failed = true;
        runner.reject(
          new Error("Selected checkpoint identity does not match its catalog"),
        );
        return;
      }
      runner.loaded = true;
      runner.manifest = data.manifest;
      status.textContent = "Ready";
      runner.resolve();
    } else if (data.type === "error") {
      if (!runner.loaded) {
        clearTimeout(runner.timer);
        runner.failed = true;
        runner.reject(new Error(data.message));
      } else {
        window.surgeComparison.errors[entry.id] = data.message;
        status.textContent = `Error: ${data.message}`;
        pending.delete(entry.id);
        finish();
      }
    } else if (data.type === "complete" && pending.has(entry.id)) {
      const record = { ...data.record, ...runner.runSettings };
      window.surgeComparison.records[entry.id] = record;
      status.textContent = "Complete";
      showResult(runner, record);
      pending.delete(entry.id);
      finish();
    }
  };
  worker.onerror = ({ message }) => {
    if (!runner.loaded) {
      clearTimeout(runner.timer);
      runner.failed = true;
      runner.reject(new Error(message));
    } else {
      window.surgeComparison.errors[entry.id] = message;
      pending.delete(entry.id);
      finish();
    }
  };
  worker.postMessage({ type: "load", modelId: entry.id });
  runners.set(entry.id, runner);
  return runner;
}

function runInput(runner, settings, content, sketchAudio) {
  const supportsSketch = Boolean(runner.manifest.sketch);
  return {
    type: "run",
    ...settings,
    mode:
      !supportsSketch && settings.mode === "both" ? "mel_only" : settings.mode,
    sketchCfg: supportsSketch ? settings.sketchCfg : 0,
    sketchSource: supportsSketch ? settings.sketchSource : "disabled",
    content,
    sketchAudio: supportsSketch
      ? sketchAudio
      : new Float32Array(sketchAudio.length),
    noise: gaussianNoise(settings.seed, runner.manifest.encodedWidth),
    authored:
      supportsSketch && settings.sketchSource === "authored"
        ? {
            pitch: settings.pitch,
            start: settings.start,
            end: settings.end,
            loudness: settings.loudness,
            centroid: settings.centroid,
          }
        : null,
  };
}

element("sketchSource").addEventListener("change", () => {
  const authored = element("sketchSource").value === "authored";
  element("author").hidden = !authored;
  element("sketchUpload").hidden = authored;
});

element("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  const selected = catalog.models.filter(
    (entry) => element(`model-${entry.id}`).checked,
  );
  if (!selected.length) {
    element("status").textContent = "Select at least one model";
    return;
  }
  if (
    element("mode").value === "sketch_only" &&
    selected.some((entry) => !entry.sketch)
  ) {
    element("status").textContent =
      "Sketch-only sampling requires selecting sketch models only";
    return;
  }
  const settings = {
    mode: element("mode").value,
    sketchSource: element("sketchSource").value,
    neuralBackend: element("webgpu").checked ? "webgpu" : "wasm",
  };
  for (const id of [
    "contentCfg",
    "sketchCfg",
    "steps",
    "seed",
    "pitch",
    "start",
    "end",
    "loudness",
    "centroid",
  ])
    settings[id] = Number(element(id).value);
  const contentFile = element("contentAudio").files[0];
  const sketchFile = element("sketchAudio").files[0];
  if (
    !contentFile ||
    (selected.some((entry) => entry.sketch) &&
      settings.sketchSource === "extracted" &&
      !sketchFile)
  ) {
    element("status").textContent =
      "Choose the required content and sketch audio";
    return;
  }

  element("run").disabled = true;
  element("comparison-download")?.remove();
  const active = prepareRunners(selected);
  window.surgeComparison = {
    state: "running",
    comparisonId: crypto.randomUUID(),
    settings,
    records: {},
    errors: {},
  };
  pending = new Set(selected.map((entry) => entry.id));
  element("status").textContent = "Decoding shared audio inputs";
  try {
    const content = await decodeStereo(
      await contentFile.arrayBuffer(),
      44100,
      176400,
    );
    const sketchAudio =
      settings.sketchSource === "authored" || !sketchFile
        ? new Float32Array(content.length)
        : await decodeStereo(await sketchFile.arrayBuffer(), 44100, 176400);
    element("status").textContent =
      "Loading selected models before concurrent inference";
    const readiness = await Promise.allSettled(
      active.map((runner) => runner.ready),
    );
    readiness.forEach((result, index) => {
      const runner = active[index];
      if (result.status === "rejected") {
        window.surgeComparison.errors[runner.entry.id] = result.reason.message;
        runner.status.textContent = `Error: ${result.reason.message}`;
        pending.delete(runner.entry.id);
      } else {
        runner.status.textContent = "Running";
        const input = runInput(runner, settings, content, sketchAudio);
        runner.runSettings = {
          ...settings,
          mode: input.mode,
          sketchCfg: input.sketchCfg,
          sketchSource: input.sketchSource,
        };
        runner.worker.postMessage(input);
      }
    });
    if (pending.size)
      element("status").textContent =
        "Selected models are running concurrently";
    else finish();
  } catch (error) {
    window.surgeComparison.errors.input = error.message;
    pending.clear();
    finish();
  }
});

try {
  const response = await fetch(
    new URL("../models/catalog.json", import.meta.url),
  );
  if (!response.ok) throw new Error(`Model catalog HTTP ${response.status}`);
  catalog = await response.json();
  if (
    !Array.isArray(catalog.models) ||
    catalog.models.length !== 4 ||
    new Set(catalog.models.map((entry) => entry.id)).size !== 4 ||
    catalog.models.some(
      (entry) =>
        !/^[a-z0-9-]+$/.test(entry.id) ||
        typeof entry.label !== "string" ||
        typeof entry.sketch !== "boolean" ||
        !/^[a-f0-9]{64}$/.test(entry.checkpointSha256),
    )
  )
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
  for (const id of ["steps", "contentCfg", "sketchCfg"])
    element(id).value = String(catalog.sampling[id]);
  element("results").classList.add("comparison-results");
  element("run").textContent = "Run selected models";
  element("run").classList.add("btn", "btn-outline-dark");
  element("run").disabled = false;
  element("status").textContent =
    "Ready — same seed gives matching noise within each parameter-space pair";
  window.surgeComparison = { state: "ready" };
} catch (error) {
  element("status").textContent = `Error: ${error.message}`;
  window.surgeComparison = { state: "error", message: error.message };
}
