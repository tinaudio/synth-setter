import * as ort from "../ort/ort.wasm.min.mjs";
import { renderSurge } from "../engine/runtime.mjs";
import { branchWeights } from "../guidance.mjs";
import { integrateRK4 } from "../rk4.mjs";
import { decodeParameters } from "./decode.mjs";
import { authorMusicSketch } from "./author.mjs";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("../ort/", import.meta.url).href;
const tensor = (values, shape) => new ort.Tensor("float32", values, shape);
const engineCommit = "dd68c74346c828ef25bd6504867936648161b7a7";
let model;

async function bytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function loadModel() {
  const root = new URL("../model/", import.meta.url);
  const manifest = JSON.parse(
    new TextDecoder().decode(await bytes(new URL("manifest.json", root))),
  );
  if (
    manifest.schemaVersion !== 1 ||
    manifest.paramSpecName !== "surge_simple" ||
    manifest.encodedWidth !== 92 ||
    manifest.channels !== 2 ||
    manifest.frames !== 176400 ||
    manifest.sampleRate !== 44100 ||
    manifest.sketch.numControls !== 386 ||
    manifest.sketch.numFrames !== 32
  ) {
    throw new Error("unsupported Surge music model contract");
  }
  const loaded = {};
  for (const name of [
    "frontend.onnx",
    "sketch.onnx",
    "conditioning.onnx",
    "velocity.onnx",
    "preset.fxp",
  ]) {
    const value = await bytes(new URL(name, root));
    const hash = [
      ...new Uint8Array(await crypto.subtle.digest("SHA-256", value)),
    ]
      .map((v) => v.toString(16).padStart(2, "0"))
      .join("");
    if (hash !== manifest.files[name]?.sha256)
      throw new Error(`model digest mismatch: ${name}`);
    loaded[name] = value;
  }
  const graphs = {};
  try {
    for (const name of ["frontend", "sketch", "conditioning", "velocity"]) {
      graphs[name] = await ort.InferenceSession.create(loaded[`${name}.onnx`], {
        executionProviders: ["wasm"],
      });
    }
  } catch (error) {
    await Promise.all(Object.values(graphs).map((graph) => graph.release()));
    throw error;
  }
  return { manifest, graphs, preset: loaded["preset.fxp"] };
}

async function evaluate(input) {
  const { manifest, graphs, preset } = model;
  for (const waveform of [input.content, input.sketchAudio]) {
    if (
      !(waveform instanceof Float32Array) ||
      waveform.length !== 2 * manifest.frames ||
      !waveform.every(Number.isFinite)
    )
      throw new Error("invalid audio contract");
  }
  if (
    !Number.isInteger(input.steps) ||
    input.steps < 1 ||
    input.steps > 2000 ||
    input.noise.length !== 92 ||
    !input.noise.every(Number.isFinite)
  )
    throw new Error("invalid sampling contract");
  const melOut = await graphs.frontend.run({
    [graphs.frontend.inputNames[0]]: tensor(input.content, [
      1,
      2,
      manifest.frames,
    ]),
  });
  const mel = melOut[graphs.frontend.outputNames[0]];
  let controls;
  if (input.authored) {
    controls = authorMusicSketch(input.authored);
  } else {
    const sketchOut = await graphs.sketch.run({
      [graphs.sketch.inputNames[0]]: tensor(input.sketchAudio, [
        1,
        2,
        manifest.frames,
      ]),
    });
    controls = Float32Array.from(sketchOut[graphs.sketch.outputNames[0]].data);
  }
  const encoded = await graphs.conditioning.run({
    mel,
    sketch_ctrl: tensor(controls, [1, 386, 32]),
  });
  const weights = tensor(
    branchWeights(input.mode, input.contentCfg, input.sketchCfg),
    [4],
  );
  const params = await integrateRK4({
    noise: input.noise,
    steps: input.steps,
    onStep: (step, total) =>
      postMessage({ type: "progress", message: `Sampling ${step}/${total}` }),
    field: async (state, time) => {
      const output = await graphs.velocity.run({
        ...encoded,
        x: tensor(state, [1, 92]),
        t: tensor(Float32Array.of(time), [1, 1]),
        branch_weights: weights,
      });
      return output.velocity.data;
    },
  });
  const patch = decodeParameters(params, manifest);
  const [noteStart, noteEnd] = [...patch.note.note_start_and_end].sort(
    (a, b) => a - b,
  );
  postMessage({ type: "progress", message: "Rendering predicted Surge patch" });
  const audio = await renderSurge({
    root: new URL("../engine/", import.meta.url).href,
    preset,
    parameters: patch.parameters,
    note: patch.note.pitch,
    velocity: 100,
    noteStart,
    noteEnd,
    sampleRate: manifest.sampleRate,
    frames: manifest.frames,
  });
  const predictedWaveform = new Float32Array(2 * manifest.frames);
  predictedWaveform.set(audio.left);
  predictedWaveform.set(audio.right, manifest.frames);
  const predictedMelOut = await graphs.frontend.run({
    [graphs.frontend.inputNames[0]]: tensor(predictedWaveform, [
      1,
      2,
      manifest.frames,
    ]),
  });
  const predictedMel = predictedMelOut[graphs.frontend.outputNames[0]].data;
  let melError = 0;
  for (let index = 0; index < predictedMel.length; index++)
    melError += Math.abs(predictedMel[index] - mel.data[index]);
  return {
    params: Array.from(params),
    mel: Array.from(mel.data),
    sketch: Array.from(controls),
    noise: Array.from(input.noise),
    patch,
    audio: [Array.from(audio.left), Array.from(audio.right)],
    normalizedMelMae: melError / predictedMel.length,
    rendererVersion: audio.version,
    engineCommit,
    sampleRate: manifest.sampleRate,
    checkpointSha256: manifest.checkpointSha256,
    statsSha256: manifest.statsSha256,
  };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      model = await loadModel();
      postMessage({ type: "ready", manifest: model.manifest });
    } else if (data.type === "run" && model) {
      postMessage({ type: "complete", record: await evaluate(data) });
    } else {
      throw new Error("model is not ready");
    }
  } catch (error) {
    postMessage({ type: "error", message: error.message });
  }
};
