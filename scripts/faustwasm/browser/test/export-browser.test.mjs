import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { mkdtemp } from "node:fs/promises";

import { exportBrowserBundle } from "../export-browser.mjs";

const execFileAsync = promisify(execFile);

async function makeArtifact(root) {
  const artifactDirectory = path.join(root, "artifact");
  await mkdir(artifactDirectory);
  await writeFile(
    path.join(artifactDirectory, "manifest.json"),
    JSON.stringify({
      schemaVersion: 1,
      identity: "test-synth",
      faustwasmVersion: "0.18.3",
      mode: "mono",
      voices: 0,
      outputs: 2,
      parameters: [],
      files: {
        dsp: {
          path: "dsp.wasm",
          sha256: "336154bf67f765f8f75d16a0accee61b5ee5f6a75b2a2905703df913bd550f3e",
          metadata: {},
        },
      },
    }),
  );
  await writeFile(path.join(artifactDirectory, "dsp.wasm"), "wasm");
  return artifactDirectory;
}

test("exportBrowserBundle_copies_artifact_and_local_player_assets", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const runtimePath = path.join(root, "runtime.mjs");
  const outputDirectory = path.join(root, "site");
  const faustLicensePath = path.join(root, "COPYING.txt");
  const faustModulePath = path.join(root, "faustwasm.mjs");
  await writeFile(
    runtimePath,
    'import "../../node_modules/@grame/faustwasm/dist/esm/index.js";\nexport const loadFaustArtifact = () => {};\n',
  );
  await writeFile(faustLicensePath, "LGPL license\n");
  await writeFile(faustModulePath, "export class FaustMonoDspGenerator {}\n");

  await exportBrowserBundle({
    artifactDirectory,
    faustLicensePath,
    faustModulePath,
    outputDirectory,
    runtimePath,
  });

  assert.equal(await readFile(path.join(outputDirectory, "dsp.wasm"), "utf8"), "wasm");
  assert.match(await readFile(path.join(outputDirectory, "index.html"), "utf8"), /Start audio/);
  const exportedRuntime = await readFile(path.join(outputDirectory, "runtime.mjs"), "utf8");
  assert.match(exportedRuntime, /loadFaustArtifact/);
  assert.match(exportedRuntime, /\.\/vendor\/faustwasm\.mjs/);
  assert.doesNotMatch(exportedRuntime, /node_modules/);
  assert.match(
    await readFile(path.join(outputDirectory, "vendor", "faustwasm.mjs"), "utf8"),
    /FaustMonoDspGenerator/,
  );
  assert.equal(
    await readFile(path.join(outputDirectory, "vendor", "FAUSTWASM-COPYING.txt"), "utf8"),
    "LGPL license\n",
  );
});

test("export_browser_cli_creates_relocatable_site_outside_checkout", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-cli-"));
  const artifactDirectory = await makeArtifact(root);
  const outputDirectory = path.join(root, "published", "player");
  const runtimePath = path.join(root, "runtime.mjs");
  await writeFile(runtimePath, "export const loadFaustArtifact = () => {};\n");

  const { stdout } = await execFileAsync(process.execPath, [
    path.resolve(import.meta.dirname, "..", "export-browser.mjs"),
    "--artifact",
    artifactDirectory,
    "--output",
    outputDirectory,
    "--runtime",
    runtimePath,
  ]);

  assert.match(stdout, /Browser bundle exported/);
  assert.match(await readFile(path.join(outputDirectory, "controller.mjs"), "utf8"), /AudioContext/);
  assert.equal(await readFile(path.join(outputDirectory, "dsp.wasm"), "utf8"), "wasm");
});

test("exportBrowserBundle_rejects_existing_output", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const outputDirectory = path.join(root, "site");
  await mkdir(outputDirectory);

  await assert.rejects(
    exportBrowserBundle({ artifactDirectory, outputDirectory }),
    /already exists/,
  );
});

test("exportBrowserBundle_rejects_corrupted_artifact", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  await writeFile(path.join(artifactDirectory, "dsp.wasm"), "corrupted");

  await assert.rejects(
    exportBrowserBundle({
      artifactDirectory,
      outputDirectory: path.join(root, "site"),
    }),
    /dsp\.wasm hash mismatch/,
  );
});

test("exportBrowserBundle_rejects_non_v1_manifest", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const manifestPath = path.join(artifactDirectory, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  manifest.schemaVersion = 2;
  await writeFile(manifestPath, JSON.stringify(manifest));

  await assert.rejects(
    exportBrowserBundle({
      artifactDirectory,
      faustModulePath: path.join(root, "missing-faustwasm.mjs"),
      outputDirectory: path.join(root, "site"),
      runtimePath: path.join(root, "missing-runtime.mjs"),
    }),
    /schemaVersion 1/,
  );
});

test("exportBrowserBundle_rejects_output_nested_in_artifact", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);

  await assert.rejects(
    exportBrowserBundle({
      artifactDirectory,
      faustModulePath: path.join(root, "faustwasm.mjs"),
      outputDirectory: path.join(artifactDirectory, "site"),
      runtimePath: path.join(root, "runtime.mjs"),
    }),
    /outside the artifact directory/,
  );
});
