import { gaussianNoise } from "../fdn/noise.mjs";
import { audioMetrics, decodeStereo, encodeStereoWav } from "./audio.mjs";

const element = (id) => document.getElementById(id);
const number = (id) => Number(element(id).value);
const worker = new Worker(new URL("./worker.mjs", import.meta.url), {
  type: "module",
});
let contract, content, runSettings;
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
  section.append(heading, player, link);
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
  element("results").append(canvas);
}

function fail(message) {
  window.surgeEval = { state: "error", message };
  element("status").textContent = `Error: ${message}`;
  element("run").disabled = !contract;
}

worker.onmessage = ({ data }) => {
  if (data.type === "ready") {
    contract = data.manifest;
    element("steps").value = String(contract.sampling.steps);
    element("contentCfg").value = String(contract.sampling.contentCfg);
    element("sketchCfg").value = String(contract.sampling.sketchCfg);
    element("status").textContent = "Ready — choose content and sketch inputs";
    element("run").disabled = false;
  } else if (data.type === "progress") {
    element("status").textContent = data.message;
  } else if (data.type === "error") {
    fail(data.message);
  } else if (data.type === "complete") {
    try {
      const target = [
        content.slice(0, contract.frames),
        content.slice(contract.frames),
      ];
      const record = { ...data.record, ...runSettings, state: "complete" };
      record.metrics = {
        normalizedMelMae: record.normalizedMelMae,
        ...audioMetrics(target, record.audio),
      };
      table(
        "Mel and signal comparison (not a perceptual quality score)",
        Object.entries(record.metrics),
      );
      table("Predicted parameters", Object.entries(record.patch.synth));
      table("Predicted note", Object.entries(record.patch.note));
      table("Run", Object.entries(runSettings));
      showSketch(record.sketch);
      playback("Target", target, "target.wav");
      playback("Prediction", record.audio, "pred.wav");
      window.surgeEval = record;
      const link = document.createElement("a");
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(record)], { type: "application/json" }),
      );
      urls.push(url);
      link.href = url;
      link.download = "evaluation.json";
      link.textContent = "Download evaluation record";
      element("results").append(link);
      element("status").textContent = "Evaluation complete";
      element("run").disabled = false;
    } catch (error) {
      fail(error.message);
    }
  }
};
worker.onerror = (event) => fail(event.message);
worker.postMessage({ type: "load" });

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
    const authored = element("sketchSource").value === "authored";
    if (!contentFile || (!authored && !sketchFile))
      throw new Error("choose content and sketch audio");
    element("status").textContent = "Decoding audio";
    content = await decodeStereo(
      await contentFile.arrayBuffer(),
      contract.sampleRate,
      contract.frames,
    );
    const sketchAudio = authored
      ? new Float32Array(content.length)
      : await decodeStereo(
          await sketchFile.arrayBuffer(),
          contract.sampleRate,
          contract.frames,
        );
    runSettings = {
      mode: element("mode").value,
      contentCfg: number("contentCfg"),
      sketchCfg: number("sketchCfg"),
      steps: number("steps"),
      seed: number("seed"),
      sketchSource: element("sketchSource").value,
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
