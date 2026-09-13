import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const url = process.env.SURGE_BROWSER_URL;
const content = process.env.SURGE_BROWSER_CONTENT;
const sketch = process.env.SURGE_BROWSER_SKETCH;
const output = process.env.SURGE_BROWSER_OUTPUT;
if (!url || !content || !sketch || !output)
  throw new Error("Set SURGE_BROWSER_URL, SURGE_BROWSER_CONTENT, SURGE_BROWSER_SKETCH, SURGE_BROWSER_OUTPUT");

async function prepare(page) {
  await page.goto(url);
  assert.equal(await page.getByLabel("WebGPU neural inference").count(), 1);
  assert.equal(await page.getByLabel("WebGPU neural inference").isChecked(), false);
  await page.waitForFunction(() => !document.getElementById("run").disabled, null, { timeout: 240000 });
  await page.locator("#contentAudio").setInputFiles(path.resolve(content));
  await page.locator("#sketchAudio").setInputFiles(path.resolve(sketch));
  await page.locator("#steps").fill("8");
  await page.locator("#seed").fill("17");
}

async function run(page, name) {
  await page.getByRole("button", { name: "Run evaluation" }).click();
  await page.waitForFunction(() => ["complete", "error"].includes(window.surgeEval?.state), null, { timeout: 480000 });
  assert.equal(await page.evaluate(() => window.surgeEval.state), "complete", await page.locator("#status").textContent());
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download evaluation record" }).click();
  await mkdir(output, { recursive: true });
  const destination = path.join(output, `${name}.json`);
  await (await download).saveAs(destination);
  const wav = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download pred.wav" }).click();
  await (await wav).saveAs(`${destination}.wav`);
  execFileSync(path.resolve(".venv/bin/python"), ["-c", `
import json,sys,numpy as np,soundfile as sf
record=json.load(open(sys.argv[1]))
audio,rate=sf.read(sys.argv[1]+'.wav',dtype='float32')
assert rate==44100 and audio.shape==(176400,2) and np.isfinite(audio).all()
np.testing.assert_array_equal(audio.T,np.asarray(record['audio'],dtype=np.float32))
`, destination]);
  return JSON.parse(await readFile(destination, "utf8"));
}

test("neural checkbox switches real execution while preserving features and three audio views", { timeout: 1200000 }, async () => {
  const browser = await chromium.launch({ headless: true, args: JSON.parse(process.env.SURGE_BROWSER_GPU_ARGS ?? "[]") });
  try {
    const page = await browser.newPage();
    await prepare(page);
    const cpu = await run(page, "wasm");
    for (const name of ["Target", "Sketch", "Prediction"]) {
      assert.equal(await page.getByRole("img", { name: `${name} spectrogram` }).count(), 1);
      const pixels = await page.getByRole("img", { name: `${name} spectrogram` }).evaluate((canvas) => Array.from(canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data));
      assert.ok(new Set(pixels).size > 4, `${name} spectrogram must contain signal, not a blank canvas`);
    }
    assert.equal(await page.locator("audio").count(), 3);
    const sketchPlayer = page.getByRole("heading", { name: "Sketch", exact: true }).locator("..").locator("audio");
    await sketchPlayer.evaluate(async (audio) => { await audio.play(); });
    await page.waitForFunction(() => [...document.querySelectorAll("audio")].some((audio) => audio.currentTime > 0));
    await page.getByLabel("WebGPU neural inference").check();
    const gpu = await run(page, "webgpu");
    assert.equal(gpu.comparisonToPrevious.previousRunId, cpu.runId);
    assert.equal(gpu.comparisonToPrevious.previousNeuralBackend, "wasm");
    assert.equal(gpu.comparisonToPrevious.parameterMaxAbsDifference, Math.max(...gpu.params.map((v, i) => Math.abs(v - cpu.params[i]))));
    assert.equal(cpu.neuralBackend, "wasm");
    assert.equal(gpu.neuralBackend, "webgpu");
    assert.equal(gpu.frontendBackend, "wasm");
    assert.equal(gpu.rendererBackend, "wasm");
    assert.ok(gpu.gpuAdapter);
    assert.equal(gpu.neuralCpuFallback, false);
    assert.deepEqual(gpu.noise, cpu.noise);
    assert.deepEqual(gpu.mel, cpu.mel);
    assert.deepEqual(gpu.sketch, cpu.sketch);
    assert.ok(gpu.params.every(Number.isFinite));
    assert.ok(gpu.neuralInferenceMs > 0);
    assert.ok(gpu.audio.flat().some((v) => Math.abs(v) > 1e-5));
    console.log(JSON.stringify({ parameterMaxAbsDifference: Math.max(...gpu.params.map((v, i) => Math.abs(v - cpu.params[i]))), wasmMs: cpu.neuralInferenceMs, webgpuMs: gpu.neuralInferenceMs, adapter: gpu.gpuAdapter }));
    await page.getByLabel("WebGPU neural inference").uncheck();
    const again = await run(page, "wasm-again");
    assert.equal(again.neuralBackend, "wasm");
    assert.deepEqual(again.params, cpu.params);
    await page.screenshot({ path: path.join(output, "comparison.png"), fullPage: true });
    await page.locator("#seed").fill("18");
    const different = await run(page, "different-seed");
    assert.equal(different.comparisonToPrevious, null);
  } finally {
    await browser.close();
  }
});

test("unavailable WebGPU reports an error instead of silently running neural WASM", { timeout: 360000 }, async () => {
  const browser = await chromium.launch({ headless: true, args: ["--disable-webgpu"] });
  try {
    const page = await browser.newPage();
    await prepare(page);
    await page.getByLabel("WebGPU neural inference").check();
    await page.getByRole("button", { name: "Run evaluation" }).click();
    await page.waitForFunction(() => ["complete", "error"].includes(window.surgeEval?.state), null, { timeout: 240000 });
    assert.equal(await page.evaluate(() => window.surgeEval.state), "error");
    assert.match(await page.locator("#status").textContent(), /WebGPU/);
    assert.equal(await page.getByRole("link", { name: "Download evaluation record" }).count(), 0);
    await page.getByLabel("WebGPU neural inference").uncheck();
    const recovered = await run(page, "unavailable-recovered");
    assert.equal(recovered.neuralBackend, "wasm");
  } finally {
    await browser.close();
  }
});
