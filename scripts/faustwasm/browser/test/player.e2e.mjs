import { expect, test } from "@playwright/test";
import { createServer } from "node:http";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { exportBrowserBundle } from "../export-browser.mjs";

const artifactDirectory = process.env.FAUSTWASM_E2E_ARTIFACT;
const monoArtifactDirectory = process.env.FAUSTWASM_MONO_E2E_ARTIFACT;
const browserDirectory = path.resolve(import.meta.dirname, "..");
const runtimePath =
  process.env.FAUSTWASM_RUNTIME_PATH ??
  path.resolve(browserDirectory, "../../../src/synth_setter/faustwasm/runtime.mjs");
let activeArtifactDirectory;
let baseUrl;
let server;
let siteDirectory;
let temporaryDirectory;

function contentType(filePath) {
  if (filePath.endsWith(".html")) return "text/html";
  if (filePath.endsWith(".js") || filePath.endsWith(".mjs")) return "text/javascript";
  if (filePath.endsWith(".json")) return "application/json";
  if (filePath.endsWith(".wasm")) return "application/wasm";
  if (filePath.endsWith(".css")) return "text/css";
  return "application/octet-stream";
}

function rms(samples) {
  return Math.sqrt(samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length);
}

test.beforeEach(async ({}, testInfo) => {
  activeArtifactDirectory = testInfo.title.startsWith("mono_")
    ? monoArtifactDirectory
    : artifactDirectory;
  test.skip(
    !activeArtifactDirectory,
    testInfo.title.startsWith("mono_")
      ? "Set FAUSTWASM_MONO_E2E_ARTIFACT to a PR2-exported filter-osc artifact directory"
      : "Set FAUSTWASM_E2E_ARTIFACT to a PR2-exported bright-organ artifact directory",
  );
  try {
    await access(path.join(activeArtifactDirectory, "manifest.json"));
  } catch {
    test.skip(true, `PR2-exported artifact is unavailable: ${activeArtifactDirectory}`);
  }
  temporaryDirectory = await mkdtemp(path.join(tmpdir(), "faust-browser-site-"));
  siteDirectory = path.join(temporaryDirectory, "player");
  await exportBrowserBundle({
    artifactDirectory: activeArtifactDirectory,
    outputDirectory: siteDirectory,
    runtimePath,
  });

  server = createServer(async (request, response) => {
    try {
      const requestPath = new URL(request.url, "http://localhost").pathname;
      const relativePath = requestPath === "/" ? "index.html" : requestPath.slice(1);
      const filePath = path.resolve(siteDirectory, relativePath);
      if (!filePath.startsWith(`${siteDirectory}${path.sep}`)) throw new Error("Invalid path");
      response.writeHead(200, { "Content-Type": contentType(filePath) });
      response.end(await readFile(filePath));
    } catch {
      response.writeHead(404);
      response.end("Not found");
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

test.afterEach(async () => {
  if (server) await new Promise((resolve) => server.close(resolve));
  if (temporaryDirectory) await rm(temporaryDirectory, { force: true, recursive: true });
});

test("mono_filter_osc_capture_matches_offline_native_channel_shape", async ({ page }) => {
  await page.goto(baseUrl);
  await page.getByRole("button", { name: "Start audio" }).click();
  await expect(page.locator("#status")).toHaveText("Audio running");
  const browserResult = await page.evaluate(async () => {
    const { context, manifest } = window.faustPlayer;
    const patch = Object.fromEntries(
      manifest.parameters.map((parameter) => [
        parameter.canonicalAddress,
        parameter.min + (parameter.max - parameter.min) * 0.5,
      ]),
    );
    window.faustPlayer.applyPatch(patch);
    const channels = await window.faustPlayer.recordFrames(4096);
    return {
      channels: channels.map((channel) => Array.from(channel)),
      patch,
      sampleRate: context.sampleRate,
    };
  });

  const manifest = JSON.parse(
    await readFile(path.join(activeArtifactDirectory, "manifest.json"), "utf8"),
  );
  const { applyCanonicalPatch, createOfflineSynth, loadFaustArtifact, renderNote } =
    await import(runtimePath);
  const artifact = await loadFaustArtifact(
    manifest,
    async (relativePath) =>
      new Uint8Array(await readFile(path.join(activeArtifactDirectory, relativePath))),
    manifest.faustwasmVersion,
  );
  const synth = await createOfflineSynth(artifact, {
    blockSize: 128,
    sampleRate: browserResult.sampleRate,
  });
  applyCanonicalPatch(synth, manifest, browserResult.patch);
  const offlineAudio = renderNote(synth, {
    endFrame: 2048,
    frames: 4096,
    note: 60,
    startFrame: 256,
    velocity: 100,
  });

  expect(manifest.identity).toBe("faust_filter_osc");
  expect(browserResult.channels).toHaveLength(offlineAudio.length);
  expect(browserResult.channels[0]).toHaveLength(offlineAudio[0].length);
  expect(browserResult.channels.flat().every(Number.isFinite)).toBe(true);
  expect(rms(browserResult.channels[0])).toBeGreaterThan(0.0001);
});

test("player_requires_gesture_then_emits_finite_stereo_audio_with_note_causality", async ({ page }) => {
  await page.goto(baseUrl);
  await expect(page.locator("#status")).toContainText("Ready");
  expect(await page.evaluate(() => window.faustPlayer)).toBeUndefined();

  await page.getByRole("button", { name: "Start audio" }).click();
  await expect(page.locator("#status")).toHaveText("Audio running");
  const beforeNote = await page.evaluate(async () =>
    (await window.faustPlayer.recordFrames(4096)).map((channel) => Array.from(channel)),
  );
  await page.getByRole("button", { name: "C4" }).dispatchEvent("pointerdown");
  const duringNote = await page.evaluate(async () =>
    (await window.faustPlayer.recordFrames(16384)).map((channel) => Array.from(channel)),
  );
  await page.getByRole("button", { name: "C4" }).dispatchEvent("pointerup");

  expect(duringNote).toHaveLength(2);
  expect(duringNote.flat().every(Number.isFinite)).toBe(true);
  expect(rms(duringNote[0])).toBeGreaterThan(0.0001);
  expect(rms(duringNote[1])).toBeGreaterThan(0.0001);
  expect(rms(duringNote[0])).toBeGreaterThan(rms(beforeNote[0]) * 10 + 0.0001);
});

test("overlapping_capture_rejects_second_request_and_preserves_first_length", async ({ page }) => {
  await page.goto(baseUrl);
  await page.getByRole("button", { name: "Start audio" }).click();
  await expect(page.locator("#status")).toHaveText("Audio running");

  const result = await page.evaluate(async () => {
    const firstCapture = window.faustPlayer.recordFrames(128);
    let secondError = null;
    try {
      await window.faustPlayer.recordFrames(256);
    } catch (error) {
      secondError = error.message;
    }
    const first = await firstCapture;
    return { firstLengths: first.map((channel) => channel.length), secondError };
  });

  expect(result.secondError).toMatch(/capture.*already in progress/i);
  expect(result.firstLengths).toEqual([128, 128]);
});

test("renderNote_rejected_during_capture_does_not_schedule_midi", async ({ page }) => {
  await page.goto(baseUrl);
  await page.getByRole("button", { name: "Start audio" }).click();
  await expect(page.locator("#status")).toHaveText("Audio running");

  const result = await page.evaluate(async () => {
    const capture = window.faustPlayer.recordFrames(16384);
    let renderError = null;
    try {
      await window.faustPlayer.renderNote({
        endFrame: 2048,
        frames: 4096,
        note: 60,
        startFrame: 256,
        velocity: 100,
      });
    } catch (error) {
      renderError = error.message;
    }
    return {
      channels: (await capture).map((channel) => Array.from(channel)),
      renderError,
    };
  });

  expect(result.renderError).toMatch(/capture.*already in progress/i);
  expect(Math.max(...result.channels.flat().map(Math.abs))).toBeLessThan(0.000001);
});

test("canonical_volume_control_changes_actual_captured_output", async ({ page }) => {
  await page.goto(baseUrl);
  await page.getByRole("button", { name: "Start audio" }).click();
  await expect(page.locator("#status")).toHaveText("Audio running");
  const volume = page.locator('input[data-canonical-address$="/Main/volume"]');
  await volume.fill("0");
  await page.getByRole("button", { name: "C4" }).dispatchEvent("pointerdown");
  const quiet = await page.evaluate(async () => Array.from((await window.faustPlayer.recordFrames(16384))[0]));
  await page.getByRole("button", { name: "C4" }).dispatchEvent("pointerup");
  await volume.fill("1");
  await page.getByRole("button", { name: "C4" }).dispatchEvent("pointerdown");
  const loud = await page.evaluate(async () => Array.from((await window.faustPlayer.recordFrames(16384))[0]));
  await page.getByRole("button", { name: "C4" }).dispatchEvent("pointerup");

  expect(rms(loud)).toBeGreaterThan(rms(quiet) * 2 + 0.0001);
});

test("browser_and_offline_paths_render_identical_patch_and_note_timeline", async ({ page }) => {
  await page.goto(baseUrl);
  await page.getByRole("button", { name: "Start audio" }).click();
  await expect(page.locator("#status")).toHaveText("Audio running");
  const renderRequest = {
    endFrame: 8192,
    frames: 65536,
    note: 60,
    startFrame: 256,
    velocity: 100,
  };
  const browserResult = await page.evaluate(async (request) => {
    const { context, manifest } = window.faustPlayer;
    const patch = Object.fromEntries(
      manifest.parameters.map((parameter) => [
        parameter.canonicalAddress,
        parameter.min + (parameter.max - parameter.min) * 0.5,
      ]),
    );
    window.faustPlayer.applyPatch(patch);
    const channels = await window.faustPlayer.renderNote(request);
    return { channels: channels.map((channel) => Array.from(channel)), patch, sampleRate: context.sampleRate };
  }, renderRequest);

  const manifest = JSON.parse(await readFile(path.join(artifactDirectory, "manifest.json"), "utf8"));
  const { applyCanonicalPatch, createOfflineSynth, loadFaustArtifact, renderNote } = await import(runtimePath);
  const artifact = await loadFaustArtifact(
    manifest,
    async (relativePath) =>
      new Uint8Array(await readFile(path.join(artifactDirectory, relativePath))),
    manifest.faustwasmVersion,
  );
  const synth = await createOfflineSynth(artifact, {
    blockSize: 128,
    sampleRate: browserResult.sampleRate,
  });
  applyCanonicalPatch(synth, manifest, browserResult.patch);
  const offlineAudio = await renderNote(synth, renderRequest);

  const metrics = browserResult.channels.map((browserChannel, channelIndex) => {
    const offlineChannel = Array.from(offlineAudio[channelIndex]);
    const squaredError = browserChannel.reduce((sum, sample, index) => {
      const error = sample - offlineChannel[index];
      return sum + error * error;
    }, 0);
    const maxAbsoluteError = browserChannel.reduce(
      (maximum, sample, index) => Math.max(maximum, Math.abs(sample - offlineChannel[index])),
      0,
    );
    const normalizedRmse = Math.sqrt(squaredError / browserChannel.length) / rms(offlineChannel);
    return { maxAbsoluteError, normalizedRmse };
  });
  const activeRms = rms(browserResult.channels[0].slice(1024, renderRequest.endFrame));
  const preNotePeak = Math.max(...browserResult.channels[0].slice(0, renderRequest.startFrame).map(Math.abs));
  const releasedRms = rms(browserResult.channels[0].slice(-8192));
  console.log(
    JSON.stringify({ activeRms, metrics, preNotePeak, releasedRms, sampleRate: browserResult.sampleRate }),
  );

  expect(browserResult.channels).toHaveLength(2);
  expect(browserResult.channels.flat().every(Number.isFinite)).toBe(true);
  expect(browserResult.channels.every((channel) => rms(channel) > 0.0001)).toBe(true);
  expect(metrics.every((metric) => metric.maxAbsoluteError < 0.000001)).toBe(true);
  expect(metrics.every((metric) => metric.normalizedRmse < 0.00001)).toBe(true);
  expect(preNotePeak).toBeLessThan(0.000001);
  expect(activeRms).toBeGreaterThan(releasedRms * 2);
});
