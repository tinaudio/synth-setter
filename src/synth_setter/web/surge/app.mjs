import { gaussianNoise } from "../fdn/noise.mjs";
import { audioMetrics, decodeStereo, encodeStereoWav } from "./audio.mjs";
import { spectrogramCanvas } from "./spectrogram.mjs";

const element = (id) => document.getElementById(id);
const number = (id) => Number(element(id).value);
const query = new URLSearchParams(location.search);
const modelId = query.get("model");
const embedded = query.has("embedded") && parent !== window;
const notifyParent = (data) => {
  if (embedded) parent.postMessage({ ...data, modelId, height: document.documentElement.scrollHeight }, location.origin);
};
const worker = new Worker(new URL("./worker.mjs", import.meta.url), {
  type: "module",
});
let contract, content, sketchAudio, runSettings, previousRun;
let urls = [];

function table(title, entries) {
  const section = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = title;
  const table = document.createElement("table");
  for (const [name, value] of entries) {
    const row = table.insertRow();
    row.insertCell().textContent = name;
    row.insertCell().textContent =
      typeof value === "number" ? value.toPrecision(7) : JSON.stringify(value);
  }
  section.append(heading, table);
  element("results").append(section);
  return section;
}

function playback(title, channels, filename) {
  const section = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = title;
  const url = URL.createObjectURL(
    new Blob([encodeStereoWav(channels, contract.sampleRate)], {
      type: "audio/wav",
    }),
  );
  urls.push(url);
  const player = document.createElement("audio");
  player.controls = true;
  player.src = url;
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.textContent = `Download ${filename}`;
  section.append(heading, player, link, spectrogramCanvas(channels, title, contract.sampleRate));
  element("results").append(section);
}

function showSketch(controls) {
  const canvas = document.createElement("canvas");
  const { numFrames, numControls } = contract.sketch;
  canvas.width = numFrames;
  canvas.height = numControls;
  canvas.setAttribute(
    "aria-label",
    "Loudness, centroid and MIDI pitch activation heatmap",
  );
  const context = canvas.getContext("2d");
  for (let row = 0; row < numControls; row++) {
    for (let frame = 0; frame < numFrames; frame++) {
      const value = controls[row * numFrames + frame];
      const level = Math.round(
        255 * Math.max(0, Math.min(1, row < 2 ? (value + 1) / 2 : value)),
      );
      context.fillStyle = `rgb(${level},${level},${255 - level})`;
      context.fillRect(frame, row, 1, 1);
    }
  }
  const section = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = "Music-sketch controls (not a spectrogram)";
  section.append(heading, canvas);
  element("results").append(section);
}

function comparePrevious(record) {
  if (!previousRun) return null;
  for (const key of ["mode", "contentCfg", "sketchCfg", "steps", "seed", "checkpointSha256", "statsSha256"]) {
    if (record[key] !== previousRun[key]) return null;
  }
  for (const key of ["noise", "mel", "sketch"]) {
    if (record[key].length !== previousRun[key].length || !record[key].every((value, index) => value === previousRun[key][index])) return null;
  }
  return {
    previousRunId: previousRun.runId,
    previousNeuralBackend: previousRun.neuralBackend,
    previousNeuralInferenceMs: previousRun.neuralInferenceMs,
    parameterMaxAbsDifference: Math.max(...record.params.map((value, index) => Math.abs(value - previousRun.params[index]))),
  };
}

function fail(message) {
  window.surgeEval = { state: "error", message };
  element("status").textContent = `Error: ${message}`;
  element("run").disabled = !contract;
  notifyParent({ type: "comparison-error", message });
}

worker.onmessage = ({ data }) => {
  if (data.type === "ready") {
    contract = data.manifest;
    element("steps").value = String(contract.sampling.steps);
    element("contentCfg").value = String(contract.sampling.contentCfg);
    element("sketchCfg").value = String(contract.sampling.sketchCfg);
    if (!contract.sketch) {
      element("mode").value = "mel_only";
      for (const mode of ["both", "sketch_only"])
        element("mode").querySelector(`option[value="${mode}"]`).disabled = true;
      element("sketchCfg").disabled = true;
      element("sketchSource").disabled = true;
      element("sketchUpload").hidden = true;
      element("author").hidden = true;
    }
    element("status").textContent = `Ready — ${contract.paramSpecName}, ${contract.encodedWidth} encoded outputs, ${contract.sketch ? "sketch" : "baseline"}`;
    element("run").disabled = false;
    notifyParent({ type: "comparison-ready", manifest: contract });
  } else if (data.type === "progress") {
    element("status").textContent = data.message;
    notifyParent({ type: "comparison-progress", message: data.message });
  } else if (data.type === "error") {
    fail(data.message);
  } else if (data.type === "complete") {
    try {
      const target = [
        content.slice(0, contract.frames),
        content.slice(contract.frames),
      ];
      const record = { ...data.record, ...runSettings, state: "complete" };
      record.comparisonToPrevious = comparePrevious(record);
      record.metrics = {
        normalizedMelMae: record.normalizedMelMae,
        ...audioMetrics(target, record.audio),
      };
      table(
        "Mel and signal comparison (not a perceptual quality score)",
        Object.entries(record.metrics),
      );
      const parameters = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Predicted synth parameters";
      parameters.append(summary, table("Predicted parameters", Object.entries(record.patch.synth)));
      element("results").append(parameters);
      table("Predicted note", Object.entries(record.patch.note));
      table("Run", Object.entries({
        ...runSettings,
        neuralInferenceMs: record.neuralInferenceMs,
        gpuAdapter: record.gpuAdapter,
        frontendBackend: record.frontendBackend,
        rendererBackend: record.rendererBackend,
      }));
      if (record.comparisonToPrevious)
        table("Neural comparison — previous matching inputs/settings (not audio parity)", Object.entries(record.comparisonToPrevious));
      if (contract.sketch) showSketch(record.sketch);
      playback("Target", target, "target.wav");
      if (runSettings.sketchSource === "extracted") {
        playback("Sketch", [sketchAudio.slice(0, contract.frames), sketchAudio.slice(contract.frames)], "sketch.wav");
      } else {
        const note = document.createElement("p");
        note.textContent = contract.sketch
          ? "Authored contour: no source sketch recording or audio spectrogram."
          : "Baseline: sketch conditioning is disabled.";
        element("results").append(note);
      }
      playback("Prediction", record.audio, modelId ? `${modelId}-pred.wav` : "pred.wav");
      const patchLink = document.createElement("a");
      const patchUrl = URL.createObjectURL(new Blob([Uint8Array.from(record.fxp)], { type: "application/octet-stream" }));
      urls.push(patchUrl);
      patchLink.href = patchUrl;
      patchLink.download = modelId ? `${modelId}.fxp` : "pred.fxp";
      patchLink.textContent = "Download FXP patch";
      const patchWarning = document.createElement("p");
      patchWarning.textContent = "FXP saved by Surge 1.4. Surge 1.3 reports a patch-version warning and may omit newer features.";
      element("results").append(patchLink, patchWarning);
      window.surgeEval = record;
      previousRun = record;
      const link = document.createElement("a");
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(record)], { type: "application/json" }),
      );
      urls.push(url);
      link.href = url;
      link.download = `evaluation-${modelId ? `${modelId}-` : ""}${record.neuralBackend}.json`;
      link.textContent = "Download evaluation record";
      element("results").append(link);
      element("status").textContent = "Evaluation complete";
      element("run").disabled = false;
      notifyParent({ type: "comparison-complete", record });
    } catch (error) {
      fail(error.message);
    }
  }
};
worker.onerror = (event) => fail(event.message);
worker.postMessage({ type: "load", modelId });
if (embedded) {
  element("controls").hidden = true;
  document.querySelector("h1").hidden = true;
  for (const paragraph of document.querySelectorAll("body > p:not([id])")) paragraph.hidden = true;
  document.body.style.margin = "0";
  document.body.style.padding = "0";
}
window.addEventListener("message", ({ source, origin, data }) => {
  if (!embedded || source !== parent || origin !== location.origin || data?.type !== "comparison-run") return;
  if (!contract || element("run").disabled) return fail("model is not ready for comparison");
  for (const [id, file] of [["contentAudio", data.contentFile], ["sketchAudio", data.sketchFile]]) {
    const transfer = new DataTransfer();
    if (file instanceof File) transfer.items.add(file);
    element(id).files = transfer.files;
  }
  for (const id of ["contentCfg", "sketchCfg", "steps", "seed", "pitch", "start", "end", "loudness", "centroid"])
    element(id).value = String(data.settings[id]);
  element("sketchSource").value = data.settings.sketchSource;
  element("mode").value = !contract.sketch && data.settings.mode === "both" ? "mel_only" : data.settings.mode;
  element("webgpu").checked = data.settings.neuralBackend === "webgpu";
  if (!element("controls").checkValidity()) return fail("invalid comparison controls");
  element("controls").requestSubmit();
});

element("sketchSource").addEventListener("change", () => {
  const authored = element("sketchSource").value === "authored";
  element("author").hidden = !authored;
  element("sketchUpload").hidden = authored;
});

element("controls").addEventListener("submit", async (event) => {
  event.preventDefault();
  element("run").disabled = true;
  urls.forEach((url) => URL.revokeObjectURL(url));
  urls = [];
  element("results").replaceChildren();
  window.surgeEval = { state: "running" };
  try {
    const contentFile = element("contentAudio").files[0];
    const sketchFile = element("sketchAudio").files[0];
    const authored = Boolean(contract.sketch) && element("sketchSource").value === "authored";
    if (!contentFile || (contract.sketch && !authored && !sketchFile))
      throw new Error("choose content and sketch audio");
    element("status").textContent = "Decoding audio";
    content = await decodeStereo(
      await contentFile.arrayBuffer(),
      contract.sampleRate,
      contract.frames,
    );
    sketchAudio = authored || !contract.sketch
      ? new Float32Array(content.length)
      : await decodeStereo(
          await sketchFile.arrayBuffer(),
          contract.sampleRate,
          contract.frames,
        );
    runSettings = {
      mode: element("mode").value,
      contentCfg: number("contentCfg"),
      sketchCfg: contract.sketch ? number("sketchCfg") : 0,
      steps: number("steps"),
      seed: number("seed"),
      sketchSource: contract.sketch ? element("sketchSource").value : "disabled",
      neuralBackend: element("webgpu").checked ? "webgpu" : "wasm",
    };
    worker.postMessage({
      type: "run",
      ...runSettings,
      content,
      sketchAudio,
      noise: gaussianNoise(runSettings.seed, contract.encodedWidth),
      authored: authored
        ? {
            pitch: number("pitch"),
            start: number("start"),
            end: number("end"),
            loudness: number("loudness"),
            centroid: number("centroid"),
          }
        : null,
    });
  } catch (error) {
    fail(error.message);
  }
});
