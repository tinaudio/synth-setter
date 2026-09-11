import * as ort from "../ort/ort.wasm.min.mjs";
import { decodeToContract } from "./audio.mjs";
import { AUTHOR_DEFAULTS, authorReverbSketch, BAND_CENTRES_HZ, RT60_RANGE_SECONDS } from "./author.mjs";
import { loadFaustBundle, loadModelBundle } from "./bundle.mjs";
import { decodeHouseholderRow } from "./decode.mjs";
import { loadGraphs, sampleParameters } from "./inference.mjs";
import { evaluateImpulseResponses } from "./metrics.mjs";
import { gaussianNoise } from "./noise.mjs";
import { canonicalPatch } from "./patch.mjs";
import { renderImpulseResponse } from "./render.mjs";
import { extractReverbSketch, SKETCH_CONTROLS, SKETCH_INTERVALS } from "./sketch.mjs";
import { encodeWav } from "./wav.mjs";
import { branchWeights } from "../guidance.mjs";

// Fewer steps than the checkpoint default so a first run on the WASM backend finishes in seconds.
const PAGE_DEFAULT_STEPS = 50;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("../ort/", import.meta.url).href;

const form = document.querySelector("#controls");
const status = document.querySelector("#status");
const runButton = document.querySelector("#run");
const results = document.querySelector("#results");
const field = (name) => form.elements[name];
const authorFieldset = document.querySelector("#author");
const heatmap = document.querySelector("#heatmap");
const bandSliders = document.querySelector("#bands");

let bundles;
// Object URLs from the previous run are released when results are replaced.
let objectUrls = [];

function setStatus(text) {
  status.textContent = text;
}

function bindSlider(input) {
  const output = input.parentElement.querySelector("output");
  const show = () => (output.value = Number(input.value).toFixed(input.step.includes(".") ? 2 : 0));
  input.addEventListener("input", () => {
    show();
    if (authorFieldset.hidden === false) drawHeatmap(authorReverbSketch(authorParams(), currentContract()));
  });
  show();
}

function buildBandSliders() {
  BAND_CENTRES_HZ.forEach((centre, index) => {
    const label = document.createElement("label");
    label.append(`${centre} Hz `);
    const input = document.createElement("input");
    input.type = "range";
    input.name = `rt60_${index}`;
    input.min = String(RT60_RANGE_SECONDS[0]);
    input.max = String(RT60_RANGE_SECONDS[1]);
    input.step = "0.01";
    input.value = String(AUTHOR_DEFAULTS.rt60Seconds[index]);
    const output = document.createElement("output");
    label.append(input, output);
    bandSliders.append(label);
  });
  authorFieldset.querySelectorAll("input[type=range]").forEach(bindSlider);
}

const authorField = (name) => Number(authorFieldset.querySelector(`[name=${name}]`).value);

// Tilt scales the four low and four high bands so most edits are one gesture; clamped to the RT60 range.
function authorParams() {
  const [low, high] = [authorField("tiltLow"), authorField("tiltHigh")];
  const rt60Seconds = BAND_CENTRES_HZ.map((_, index) => {
    const scaled = authorField(`rt60_${index}`) * (index < 4 ? low : high);
    return Math.min(RT60_RANGE_SECONDS[1], Math.max(RT60_RANGE_SECONDS[0], scaled));
  });
  return {
    rt60Seconds,
    preDelayMs: authorField("preDelayMs"),
    mixingTimeMs: authorField("mixingTimeMs"),
    initialDensity: authorField("initialDensity"),
    flatnessStart: authorField("flatnessStart"),
    flatnessEnd: authorField("flatnessEnd"),
  };
}

function currentContract() {
  const { sampleRate, frames } = bundles.model.manifest;
  return { sampleRate, frames };
}

function drawHeatmap(sketch) {
  const context = heatmap.getContext("2d");
  const cellWidth = heatmap.width / SKETCH_INTERVALS;
  const cellHeight = heatmap.height / SKETCH_CONTROLS;
  for (let row = 0; row < SKETCH_CONTROLS; row++) {
    for (let k = 0; k < SKETCH_INTERVALS; k++) {
      const level = Math.round(((sketch[row * SKETCH_INTERVALS + k] + 1) / 2) * 255);
      context.fillStyle = `rgb(${level}, ${Math.round(level * 0.8)}, ${255 - level})`;
      context.fillRect(k * cellWidth, row * cellHeight, cellWidth, cellHeight);
    }
  }
}

function syncSketchSource() {
  const authored = field("sketchSource").value === "authored";
  authorFieldset.hidden = !authored;
  if (authored && bundles) drawHeatmap(authorReverbSketch(authorParams(), currentContract()));
}

async function loadBundles() {
  setStatus("Loading model bundle and Faust artifact");
  const model = await loadModelBundle(new URL("../model", import.meta.url).href);
  const faust = await loadFaustBundle(new URL("../faust", import.meta.url).href);
  const graphs = await loadGraphs(ort, model.graphs);
  const { sampling, paramSpecName, frames, sampleRate } = model.manifest;
  field("content").value = sampling.contentCfg;
  field("sketch").value = sampling.sketchCfg;
  field("steps").value = PAGE_DEFAULT_STEPS;
  bundles = { model, faust, graphs };
  syncSketchSource();
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
  objectUrls.push(url);
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
  objectUrls.forEach((url) => URL.revokeObjectURL(url));
  objectUrls = [];
  window.fdnEval = { state: "running" };
  try {
    const { manifest } = bundles.model;
    const contract = { sampleRate: manifest.sampleRate, frames: manifest.frames };
    const file = field("target").files[0];
    if (!file) throw new Error("choose a WAV file first");
    setStatus("Decoding target audio");
    const decoded = await decodeToContract(await file.arrayBuffer(), contract);
    const target = decoded.samples;
    const sketchSource = field("sketchSource").value;
    setStatus(sketchSource === "authored" ? "Authoring reverb sketch" : "Extracting reverb sketch");
    const authored = sketchSource === "authored" ? authorParams() : null;
    const sketch = authored ? authorReverbSketch(authored, contract) : extractReverbSketch(target, contract.sampleRate);
    drawHeatmap(sketch);
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
      ["sketch source", sketchSource],
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
      sketchSource,
      authored,
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
field("sketchSource").addEventListener("change", syncSketchSource);
buildBandSliders();
loadBundles().catch((error) => {
  window.fdnEval = { state: "error", message: error.message };
  setStatus(`Error: ${error.message}`);
});
