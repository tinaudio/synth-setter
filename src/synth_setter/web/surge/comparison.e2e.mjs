import assert from "node:assert/strict";
import test from "node:test";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const url = new URL(process.env.SURGE_BROWSER_URL);
url.searchParams.set("compare", "1");
const output = process.env.SURGE_BROWSER_OUTPUT;

for (const backend of ["wasm", "webgpu"]) {
test(`four selected checkpoints run concurrently on ${backend} and download distinct predictions`, { timeout: 1200000 }, async () => {
  const browser = await chromium.launch({ headless: true, args: JSON.parse(process.env.SURGE_BROWSER_GPU_ARGS ?? "[]") });
  try {
    const page = await browser.newPage();
    await page.goto(url.href);
    await page.waitForFunction(() => !document.getElementById("run").disabled, null, { timeout: 240000 });
    assert.equal(await page.locator("#models input:checked").count(), 4);
    await page.locator("#contentAudio").setInputFiles(path.resolve(process.env.SURGE_BROWSER_CONTENT));
    await page.locator("#sketchAudio").setInputFiles(path.resolve(process.env.SURGE_BROWSER_SKETCH));
    await page.locator("#steps").fill("8");
    if (backend === "webgpu") await page.getByLabel("WebGPU neural inference").check();
    await page.getByRole("button", { name: "Run selected models" }).click();
    await page.waitForFunction(() => ["complete", "error"].includes(window.surgeComparison?.state), null, { timeout: 900000 });
    assert.equal(await page.evaluate(() => window.surgeComparison.state), "complete", await page.locator("#status").textContent());
    const download = page.waitForEvent("download");
    await page.getByRole("link", { name: "Download comparison records" }).click();
    await mkdir(output, { recursive: true });
    const destination = path.join(output, `comparison-${backend}.json`);
    await (await download).saveAs(destination);
    const comparison = JSON.parse(await readFile(destination, "utf8"));
    const records = comparison.records;
    assert.deepEqual(Object.keys(records).sort(), ["simple-baseline", "simple-sketch", "xt-baseline", "xt-sketch"]);
    assert.deepEqual(records["simple-baseline"].noise, records["simple-sketch"].noise);
    assert.deepEqual(records["xt-baseline"].noise, records["xt-sketch"].noise);
    assert.ok(Math.max(...Object.values(records).map((r) => Date.parse(r.startedAt))) < Math.min(...Object.values(records).map((r) => Date.parse(r.completedAt))));
    assert.equal(new Set(Object.values(records).map((r) => r.checkpointSha256)).size, 4);
    assert.equal(new Set(Object.values(records).map((r) => r.runId)).size, 4);
    for (const [id, record] of Object.entries(records)) {
      assert.equal(record.params.length, id.startsWith("simple") ? 92 : 300);
      assert.equal(record.neuralBackend, backend);
      assert.ok(record.params.every(Number.isFinite));
      assert.ok(record.audio.flat().some((v) => Math.abs(v) > 1e-5));
      const frame = page.frameLocator(`iframe[data-model="${id}"]`);
      for (const [label, extension] of [[`Download ${id}-pred.wav`, "wav"], ["Download FXP patch", "fxp"]]) {
        const download = page.waitForEvent("download");
        await frame.getByRole("link", { name: label }).click();
        await (await download).saveAs(path.join(output, `${id}-${backend}.${extension}`));
      }
      assert.equal(await frame.getByRole("img", { name: "Prediction spectrogram" }).count(), 1);
    }
    await page.screenshot({ path: path.join(output, `comparison-${backend}.png`), fullPage: true });
  } finally {
    await browser.close();
  }
});
}
