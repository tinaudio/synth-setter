import assert from "node:assert/strict";
import test from "node:test";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "@playwright/test";

import { renderSurge } from "../../surgewasm/runtime.mjs";

const url = new URL(process.env.SURGE_BROWSER_URL);
url.searchParams.set("compare", "1");
const output = process.env.SURGE_BROWSER_OUTPUT;
const engineBundle = process.env.SURGE_WASM_BUNDLE;
if (!engineBundle)
  throw new Error("SURGE_WASM_BUNDLE must name the tested engine bundle");
const loader = (
  await import(pathToFileURL(path.join(engineBundle, "surge-host.mjs")).href)
).default;

for (const backend of ["wasm", "webgpu"]) {
  test(
    `four selected checkpoints run concurrently on ${backend} and download distinct predictions`,
    { timeout: 1200000 },
    async (context) => {
      const browser = await chromium.launch({
        headless: true,
        args: JSON.parse(process.env.SURGE_BROWSER_GPU_ARGS ?? "[]"),
      });
      try {
        const page = await browser.newPage();
        await page.goto(url.href);
        if (
          backend === "webgpu" &&
          !(await page.evaluate(async () =>
            Boolean(await navigator.gpu?.requestAdapter()),
          ))
        ) {
          context.skip(
            "WebGPU adapter is unavailable in this browser configuration",
          );
          return;
        }
        await page.waitForFunction(
          () => !document.getElementById("run").disabled,
          null,
          { timeout: 240000 },
        );
        assert.equal(await page.locator("#models input:checked").count(), 4);
        await page
          .locator("#contentAudio")
          .setInputFiles(path.resolve(process.env.SURGE_BROWSER_CONTENT));
        await page
          .locator("#sketchAudio")
          .setInputFiles(path.resolve(process.env.SURGE_BROWSER_SKETCH));
        await page.locator("#steps").fill("8");
        if (backend === "webgpu")
          await page.getByLabel("WebGPU neural inference").check();
        await page.getByRole("button", { name: "Run selected models" }).click();
        await page.waitForFunction(
          () => ["complete", "error"].includes(window.surgeComparison?.state),
          null,
          { timeout: 900000 },
        );
        assert.equal(
          await page.evaluate(() => window.surgeComparison.state),
          "complete",
          await page.locator("#status").textContent(),
        );
        const download = page.waitForEvent("download");
        await page
          .getByRole("link", { name: "Download comparison records" })
          .click();
        await mkdir(output, { recursive: true });
        const destination = path.join(output, `comparison-${backend}.json`);
        await (await download).saveAs(destination);
        const comparison = JSON.parse(await readFile(destination, "utf8"));
        const records = comparison.records;
        assert.deepEqual(Object.keys(records).sort(), [
          "simple-baseline",
          "simple-sketch",
          "xt-baseline",
          "xt-sketch",
        ]);
        assert.deepEqual(
          records["simple-baseline"].noise,
          records["simple-sketch"].noise,
        );
        assert.deepEqual(
          records["xt-baseline"].noise,
          records["xt-sketch"].noise,
        );
        assert.ok(
          Math.max(
            ...Object.values(records).map((r) => Date.parse(r.startedAt)),
          ) <
            Math.min(
              ...Object.values(records).map((r) => Date.parse(r.completedAt)),
            ),
        );
        assert.equal(
          new Set(Object.values(records).map((r) => r.checkpointSha256)).size,
          4,
        );
        assert.equal(
          new Set(Object.values(records).map((r) => r.runId)).size,
          4,
        );
        for (const [id, record] of Object.entries(records)) {
          assert.equal(
            record.params.length,
            id.startsWith("simple") ? 92 : 300,
          );
          assert.equal(record.neuralBackend, backend);
          assert.ok(record.params.every(Number.isFinite));
          assert.ok(record.audio.flat().some((v) => Math.abs(v) > 1e-5));
          const card = page.locator(`[data-model="${id}"]`);
          assert.equal(await card.locator("iframe").count(), 0);
          const wavLink = card.getByRole("link", {
            name: `Download ${id}-pred.wav`,
          });
          const decodedWav = await wavLink.evaluate(async (link) => {
            const context = new AudioContext();
            try {
              const audio = await context.decodeAudioData(
                await (await fetch(link.href)).arrayBuffer(),
              );
              return {
                channels: audio.numberOfChannels,
                frames: audio.length,
                sampleRate: audio.sampleRate,
              };
            } finally {
              await context.close();
            }
          });
          assert.equal(decodedWav.channels, 2);
          assert.equal(decodedWav.frames, decodedWav.sampleRate * 4);
          const wavDownload = page.waitForEvent("download");
          await wavLink.click();
          await (
            await wavDownload
          ).saveAs(path.join(output, `${id}-${backend}.wav`));

          const recordDownload = page.waitForEvent("download");
          await card
            .getByRole("link", { name: "Download evaluation record" })
            .click();
          const recordPath = path.join(output, `${id}-${backend}.json`);
          await (await recordDownload).saveAs(recordPath);
          assert.equal(
            JSON.parse(await readFile(recordPath, "utf8")).runId,
            record.runId,
          );

          const fxpDownload = page.waitForEvent("download");
          await card.getByRole("link", { name: "Download FXP patch" }).click();
          const fxpPath = path.join(output, `${id}-${backend}.fxp`);
          await (await fxpDownload).saveAs(fxpPath);
          const reloaded = await renderSurge({
            root: new URL("../engine/", url).href,
            loader,
            preset: new Uint8Array(await readFile(fxpPath)),
            parameters: [],
            note: 60,
            velocity: 100,
            noteStart: 0,
            noteEnd: 0.5,
            sampleRate: 44100,
            frames: 44100,
          });
          assert.ok(reloaded.left.every(Number.isFinite));
          assert.ok(reloaded.right.every(Number.isFinite));
          assert.ok(
            reloaded.left.some((sample) => sample !== 0) ||
              reloaded.right.some((sample) => sample !== 0),
          );
          assert.equal(
            await card
              .getByRole("img", { name: "Prediction spectrogram" })
              .count(),
            1,
          );
        }
        await page.screenshot({
          path: path.join(output, `comparison-${backend}.png`),
          fullPage: true,
        });
      } finally {
        await browser.close();
      }
    },
  );
}
