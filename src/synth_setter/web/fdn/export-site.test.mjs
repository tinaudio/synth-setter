import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { exportSite } from "./export-site.mjs";

async function fakeBundle(root, name, files) {
  const directory = path.join(root, name);
  await mkdir(path.join(directory, "vendor"), { recursive: true });
  const manifest = { schemaVersion: 1, paramSpecName: "pyfdn_n8_mono_householder", files: {} };
  for (const [key, filename] of Object.entries(files)) {
    await writeFile(path.join(directory, filename), `${key} bytes`);
    manifest.files[key] = key === filename ? { sha256: "0".repeat(64) } : { path: filename, sha256: "0".repeat(64) };
  }
  await writeFile(path.join(directory, "manifest.json"), JSON.stringify(manifest));
  return directory;
}

async function fakeOrt(root) {
  const directory = path.join(root, "ort-dist");
  await mkdir(directory, { recursive: true });
  for (const name of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm", "ort.node.min.js"]) {
    await writeFile(path.join(directory, name), name);
  }
  return directory;
}

test("exported site contains the page, ports, runtimes, and both bundles", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "fdn-site-"));
  const modelDirectory = await fakeBundle(root, "model", { "frontend.onnx": "frontend.onnx", "conditioning.onnx": "conditioning.onnx", "velocity.onnx": "velocity.onnx" });
  const faustDirectory = await fakeBundle(root, "faust", { dsp: "dsp.wasm" });
  const outputDirectory = path.join(root, "site");
  const result = await exportSite({ modelDirectory, faustDirectory, outputDirectory, ortDirectory: await fakeOrt(root) });
  assert.equal(result.paramSpecName, "pyfdn_n8_mono_householder");
  for (const relative of [
    "fdn/index.html",
    "fdn/app.mjs",
    "fdn/sketch.mjs",
    "fdn/author.mjs",
    "fdn/fixtures/octave_bands.json",
    "rk4.mjs",
    "guidance.mjs",
    "ort/ort.wasm.min.mjs",
    "ort/ort-wasm-simd-threaded.wasm",
    "faust/runtime.mjs",
    "faust/vendor/faustwasm.mjs",
    "faust/vendor/package.json",
    "faust/manifest.json",
    "faust/dsp.wasm",
    "model/manifest.json",
    "model/velocity.onnx",
  ]) {
    assert.ok((await stat(path.join(outputDirectory, relative))).isFile(), relative);
  }
  assert.match(await readFile(path.join(outputDirectory, "index.html"), "utf8"), /fdn\/index\.html/);
  await assert.rejects(stat(path.join(outputDirectory, "ort", "ort.node.min.js")));
  await assert.rejects(stat(path.join(outputDirectory, "fdn", "sketch.test.mjs")));
});

test("export refuses a model bundle whose manifest lacks a graph", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "fdn-site-"));
  const modelDirectory = await fakeBundle(root, "model", { "frontend.onnx": "frontend.onnx", "velocity.onnx": "velocity.onnx" });
  const faustDirectory = await fakeBundle(root, "faust", { dsp: "dsp.wasm" });
  await assert.rejects(
    exportSite({ modelDirectory, faustDirectory, outputDirectory: path.join(root, "site"), ortDirectory: await fakeOrt(root) }),
    /files\.conditioning\.onnx/,
  );
});

test("export refuses a missing Faust artifact directory", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "fdn-site-"));
  const modelDirectory = await fakeBundle(root, "model", { "frontend.onnx": "frontend.onnx", "conditioning.onnx": "conditioning.onnx", "velocity.onnx": "velocity.onnx" });
  await assert.rejects(
    exportSite({ modelDirectory, faustDirectory: path.join(root, "absent"), outputDirectory: path.join(root, "site"), ortDirectory: await fakeOrt(root) }),
    /Faust artifact directory is missing/,
  );
});
