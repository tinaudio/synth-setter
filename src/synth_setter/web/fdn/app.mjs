import * as ort from "../ort/ort.wasm.min.mjs";
import { decodeToContract } from "./audio.mjs";
import { loadFaustBundle, loadModelBundle } from "./bundle.mjs";
import { decodeHouseholderRow } from "./decode.mjs";
import { loadGraphs, sampleParameters } from "./inference.mjs";
import { evaluateImpulseResponses } from "./metrics.mjs";
import { gaussianNoise } from "./noise.mjs";
import { canonicalPatch } from "./patch.mjs";
import { renderImpulseResponse } from "./render.mjs";
import { extractReverbSketch } from "./sketch.mjs";
import { encodeWav } from "./wav.mjs";
import { branchWeights } from "../guidance.mjs";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("../ort/", import.meta.url).href;

const form = document.querySelector("#controls");
const status = document.querySelector("#status");
const runButton = document.querySelector("#run");
const results = document.querySelector("#results");
const field = (name) => form.elements[name];

let bundles;

function setStatus(text) {
  status.textContent = text;
}

async function loadBundles() {
  setStatus("Loading model bundle and Faust artifact");
  const model = await loadModelBundle("model");
  const faust = await loadFaustBundle("faust");
  const graphs = await loadGraphs(ort, model.graphs);
  const { sampling, paramSpecName, frames, sampleRate } = model.manifest;
  field("content").value = sampling.contentCfg;
  field("sketch").value = sampling.sketchCfg;
  bundles = { model, faust, graphs };
  setStatus(`Ready: ${paramSpecName}, ${frames} frames at ${sampleRate} Hz; checkpoint default is ${sampling.steps} steps`);
  runButton.disabled = false;
}

function renderTable(container, title, rows) {
  const section = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = title;
  const table = document.createElement("table");
  for (const [key, value] of rows) {
    const row = table.insertRow();
    row.insertCell().textContent = key;
    row.insertCell().textContent = typeof value === "number" ? value.toPrecision(6) : String(value);
  }
  section.append(heading, table);
  container.append(section);
}

function renderAudio(container, title, samples, sampleRate, filename) {
  const section = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = title;
  const url = URL.createObjectURL(new Blob([encodeWav(samples, sampleRate)], { type: "audio/wav" }));
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = url;
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.textContent = `Download ${filename}`;
  section.append(heading, audio, link);
  container.append(section);
}

function drawDecay(container, target, pred, sampleRate) {
  const section = document.createElement("section");
  const heading = document.createElement("h2");
  heading.textContent = "Broadband energy decay (dB)";
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 240;
  const context = canvas.getContext("2d");
  context.fillStyle = "#fff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const curve = (samples) => {
    const decay = new Float64Array(samples.length);
    let total = 0;
    for (let n = samples.length - 1; n >= 0; n--) {
      total += samples[n] * samples[n];
      decay[n] = total;
    }
    return decay.map((energy) => 10 * Math.log10(energy / decay[0] + 1e-30));
  };
  for (const [samples, color] of [[target, "#1f77b4"], [pred, "#d62728"]]) {
    const db = curve(samples);
    context.strokeStyle = color;
    context.beginPath();
    for (let x = 0; x < canvas.width; x++) {
      const n = Math.min(samples.length - 1, Math.floor((x / canvas.width) * samples.length));
      const y = Math.min(canvas.height, (-db[n] / 80) * canvas.height);
      if (x === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
  }
  const legend = document.createElement("p");
  legend.textContent = `blue: target, red: prediction; ${(target.length / sampleRate).toFixed(2)} s, 0 to -80 dB`;
  section.append(heading, canvas, legend);
  container.append(section);
}

async function run(event) {
  event.preventDefault();
  runButton.disabled = true;
  results.replaceChildren();
  window.fdnEval = { state: "running" };
  try {
    const { manifest } = bundles.model;
    const contract = { sampleRate: manifest.sampleRate, frames: manifest.frames };
    const file = field("target").files[0];
    if (!file) throw new Error("choose a WAV file first");
    setStatus("Decoding target audio");
    const decoded = await decodeToContract(await file.arrayBuffer(), contract);
    const target = decoded.samples;
    setStatus("Extracting reverb sketch");
    const sketch = extractReverbSketch(target, contract.sampleRate);
    const mode = field("mode").value;
    const contentCfg = Number(field("content").value);
    const sketchCfg = Number(field("sketch").value);
    const steps = Number(field("steps").value);
    const seed = Number(field("seed").value);
    const weights = branchWeights(mode, contentCfg, sketchCfg);
    const noise = gaussianNoise(seed, manifest.encodedWidth);
    const started = performance.now();
    const { params } = await sampleParameters({
      ort,
      graphs: bundles.graphs,
      waveform: Float32Array.from(target),
      sketch,
      weights,
      noise,
      steps,
      onStep: (step, total) => setStatus(`Sampling ${step}/${total}`),
    });
    const samplingMs = performance.now() - started;
    setStatus("Rendering the predicted FDN");
    const native = decodeHouseholderRow(params);
    const pred = await renderImpulseResponse(bundles.faust, native, contract);
    setStatus("Scoring");
    const metrics = evaluateImpulseResponses(target, pred, contract.sampleRate);
    const patch = canonicalPatch(native);
    renderTable(results, "Metrics", Object.entries(metrics));
    renderTable(results, "Predicted parameters", Object.entries(patch));
    renderTable(results, "Run", [
      ["mode", mode],
      ["content CFG", contentCfg],
      ["sketch CFG", sketchCfg],
      ["steps", steps],
      ["seed", seed],
      ["sampling ms", Math.round(samplingMs)],
      ["source", `${decoded.sourceChannels} ch, ${decoded.sourceSampleRate} Hz, ${decoded.sourceFrames} frames`],
    ]);
    renderAudio(results, "Target", target, contract.sampleRate, "target.wav");
    renderAudio(results, "Prediction", pred, contract.sampleRate, "pred.wav");
    drawDecay(results, target, pred, contract.sampleRate);
    window.fdnEval = {
      state: "complete",
      mode,
      contentCfg,
      sketchCfg,
      steps,
      seed,
      weights: Array.from(weights),
      noise: Array.from(noise),
      params: Array.from(params),
      patch,
      metrics,
      sketch: Array.from(sketch),
      target: Array.from(target),
      pred: Array.from(pred),
    };
    setStatus("Evaluation complete");
  } catch (error) {
    window.fdnEval = { state: "error", message: error.message };
    setStatus(`Error: ${error.message}`);
  } finally {
    runButton.disabled = false;
  }
}

form.addEventListener("submit", run);
loadBundles().catch((error) => {
  window.fdnEval = { state: "error", message: error.message };
  setStatus(`Error: ${error.message}`);
});
