import { FaustMonoDspGenerator, FaustPolyDspGenerator } from "./vendor/faustwasm.mjs";
import { applyCanonicalPatch, loadFaustArtifact } from "./runtime.mjs";

const AUDIO_BLOCK_SIZE = 128;
const CAPTURE_LEAD_BLOCKS = 32;
const DEFAULT_VELOCITY = 100;
const startButton = document.querySelector("#start");
const status = document.querySelector("#status");
const controls = document.querySelector("#controls");
const notes = document.querySelector("#notes");

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}: HTTP ${response.status}`);
  return response.json();
}

async function loadBytes(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function createFaustNode(context, manifest, artifact) {
  if (manifest.mode === "poly") {
    const generator = new FaustPolyDspGenerator();
    return generator.createNode(
      context,
      manifest.voices,
      manifest.identity,
      artifact.dspFactory,
      artifact.mixerModule,
      artifact.effectFactory,
      false,
      AUDIO_BLOCK_SIZE,
    );
  }
  const generator = new FaustMonoDspGenerator();
  return generator.createNode(
    context,
    manifest.identity,
    artifact.dspFactory,
    false,
    AUDIO_BLOCK_SIZE,
  );
}

function renderControls(node, parameters) {
  for (const parameter of parameters) {
    const row = document.createElement("label");
    row.className = "control";
    row.textContent = parameter.canonicalAddress;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = parameter.kind === "discrete" ? "1" : "0.001";
    const nativeValue = node.getParamValue(parameter.wasmAddress);
    const nativeRange = parameter.max - parameter.min;
    const normalizedValue = nativeRange === 0 ? 0 : (nativeValue - parameter.min) / nativeRange;
    slider.value = String(Math.min(1, Math.max(0, normalizedValue)));
    slider.dataset.canonicalAddress = parameter.canonicalAddress;

    const value = document.createElement("output");
    value.value = Number(slider.value).toFixed(3);
    slider.addEventListener("input", () => {
      const normalized = Number(slider.value);
      const wasmValue = parameter.min + normalized * (parameter.max - parameter.min);
      node.setParamValue(parameter.wasmAddress, wasmValue);
      value.value = normalized.toFixed(3);
    });
    row.append(slider, value);
    controls.append(row);
  }
}

function enableNotes(node, isPolyphonic) {
  if (!isPolyphonic) return;
  notes.hidden = false;
  for (const button of notes.querySelectorAll("button[data-note]")) {
    const note = Number(button.dataset.note);
    const noteOff = () => node.keyOff(0, note, DEFAULT_VELOCITY);
    button.addEventListener("pointerdown", () => node.keyOn(0, note, DEFAULT_VELOCITY));
    button.addEventListener("pointerup", noteOff);
    button.addEventListener("pointercancel", noteOff);
    button.addEventListener("pointerleave", noteOff);
  }
}

function nextCaptureFrame(context) {
  const currentFrame = Math.ceil((context.currentTime * context.sampleRate) / AUDIO_BLOCK_SIZE);
  return (currentFrame + CAPTURE_LEAD_BLOCKS) * AUDIO_BLOCK_SIZE;
}

function captureFrames(captureNode, frameCount, startFrame) {
  if (!Number.isInteger(frameCount) || frameCount <= 0) {
    return Promise.reject(new Error("Capture frame count must be a positive integer"));
  }
  return new Promise((resolve) => {
    captureNode.port.addEventListener("message", (event) => resolve(event.data), { once: true });
    captureNode.port.start();
    captureNode.port.postMessage({ frames: frameCount, startFrame });
  });
}

function captureOutput(context, captureNode, frameCount) {
  return captureFrames(captureNode, frameCount, nextCaptureFrame(context));
}

function renderNote(context, node, captureNode, request) {
  const { endFrame, frames, note, startFrame, velocity } = request;
  if (!(0 <= startFrame && startFrame < endFrame && endFrame <= frames)) {
    return Promise.reject(new Error("Note frames must satisfy 0 <= start < end <= frames"));
  }
  const captureStartFrame = nextCaptureFrame(context);
  const captured = captureFrames(captureNode, frames, captureStartFrame);
  node.keyOn(0, note, velocity, (captureStartFrame + startFrame) / context.sampleRate);
  node.keyOff(0, note, 0, (captureStartFrame + endFrame) / context.sampleRate);
  return captured;
}

async function startPlayer() {
  startButton.disabled = true;
  status.id = "status";
  status.textContent = "Loading artifact…";
  let context;
  try {
    context = new AudioContext({ latencyHint: "interactive" });
    await context.resume();
    const manifest = await fetchJson("./manifest.json");
    const artifact = await loadFaustArtifact(manifest, loadBytes);
    const node = await createFaustNode(context, manifest, artifact);
    if (!node) throw new Error("FaustWasm did not create an audio node");

    await context.audioWorklet.addModule("./capture-worklet.js");
    const captureNode = new AudioWorkletNode(context, "faust-output-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });
    node.connect(captureNode).connect(context.destination);
    renderControls(node, manifest.parameters);
    enableNotes(node, manifest.mode === "poly");
    document.querySelector("#title").textContent = manifest.identity;
    status.textContent = "Audio running";
    window.faustPlayer = {
      applyPatch: (params) => applyCanonicalPatch({ processor: node }, manifest, params),
      recordFrames: (frameCount) => captureOutput(context, captureNode, frameCount),
      context,
      manifest,
      node,
      renderNote: (request) => renderNote(context, node, captureNode, request),
    };
  } catch (error) {
    if (context) await context.close();
    status.id = "error";
    status.textContent = `Could not start audio: ${error.message}`;
    startButton.disabled = false;
  }
}

startButton.addEventListener("click", startPlayer);
