import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { lstat, mkdir, readFile, rename, rm, symlink, writeFile } from "node:fs/promises";
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
      libfaustVersion: "2.88.0",
      compileOptions: "-ftz 2",
      sourceSha256: "c81d2c96157c96f8e0b8478321fe21d8cc1473d2478c9f6bdcc843593d18f70f",
      mode: "mono",
      voices: 0,
      outputs: 2,
      parameters: [],
      files: {
        dsp: {
          path: "dsp.wasm",
          sha256: "336154bf67f765f8f75d16a0accee61b5ee5f6a75b2a2905703df913bd550f3e",
        },
      },
      dspMeta: {
        name: "test-synth",
        inputs: 0,
        outputs: 2,
        ui: [],
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
  const relocatedDirectory = path.join(root, "relocated-player");
  await rename(outputDirectory, relocatedDirectory);
  await rm(artifactDirectory, { recursive: true });
  assert.equal(await readFile(path.join(relocatedDirectory, "dsp.wasm"), "utf8"), "wasm");
  assert.equal((await lstat(path.join(relocatedDirectory, "dsp.wasm"))).isFile(), true);
  assert.equal((await lstat(path.join(relocatedDirectory, "dsp.wasm"))).isSymbolicLink(), false);
});

test("exportBrowserBundle_ignores_unreferenced_symlinked_content", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const runtimePath = path.join(root, "runtime.mjs");
  const outputDirectory = path.join(root, "site");
  const faustLicensePath = path.join(root, "COPYING.txt");
  const faustModulePath = path.join(root, "faustwasm.mjs");
  const outsidePath = path.join(root, "outside.txt");
  await writeFile(runtimePath, "export const loadFaustArtifact = () => {};\n");
  await writeFile(faustLicensePath, "LGPL license\n");
  await writeFile(faustModulePath, "export class FaustMonoDspGenerator {}\n");
  await writeFile(outsidePath, "outside\n");
  await symlink(outsidePath, path.join(artifactDirectory, "unreferenced.txt"));

  await exportBrowserBundle({
    artifactDirectory,
    faustLicensePath,
    faustModulePath,
    outputDirectory,
    runtimePath,
  });

  await assert.rejects(lstat(path.join(outputDirectory, "unreferenced.txt")), { code: "ENOENT" });
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

test("exportBrowserBundle_rejects_manifest_without_identity", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = path.join(root, "artifact");
  await mkdir(artifactDirectory);
  await writeFile(
    path.join(artifactDirectory, "manifest.json"),
    JSON.stringify({ schemaVersion: 1, files: {} }),
  );

  await assert.rejects(
    exportBrowserBundle({
      artifactDirectory,
      outputDirectory: path.join(root, "site"),
    }),
    /manifest\.identity/,
  );
});

test("exportBrowserBundle_rejects_nonpositive_output_count", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const manifestPath = path.join(artifactDirectory, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  manifest.outputs = 0;
  await writeFile(manifestPath, JSON.stringify(manifest));

  await assert.rejects(
    exportBrowserBundle({ artifactDirectory, outputDirectory: path.join(root, "site") }),
    /manifest\.outputs: must be a positive integer/,
  );
});

test("exportBrowserBundle_rejects_invalid_parameter_entry", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const manifestPath = path.join(artifactDirectory, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  manifest.parameters = [
    {
      canonicalAddress: "/gain",
      kind: "discrete",
      max: 1,
      min: 0.5,
      wasmAddress: "/test-synth/gain",
    },
  ];
  await writeFile(manifestPath, JSON.stringify(manifest));

  await assert.rejects(
    exportBrowserBundle({ artifactDirectory, outputDirectory: path.join(root, "site") }),
    /manifest\.parameters\[0\]: discrete bounds must be integers/,
  );
});

test("exportBrowserBundle_rejects_missing_dsp_metadata", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const manifestPath = path.join(artifactDirectory, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  delete manifest.dspMeta;
  await writeFile(manifestPath, JSON.stringify(manifest));

  await assert.rejects(
    exportBrowserBundle({ artifactDirectory, outputDirectory: path.join(root, "site") }),
    /manifest\.dspMeta: must be an object/,
  );
});

test("exportBrowserBundle_rejects_poly_artifact_without_mixer_file", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const manifestPath = path.join(artifactDirectory, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  manifest.mode = "poly";
  manifest.voices = 1;
  await writeFile(manifestPath, JSON.stringify(manifest));

  await assert.rejects(
    exportBrowserBundle({ artifactDirectory, outputDirectory: path.join(root, "site") }),
    /manifest\.files\.mixer: must be an object/,
  );
});

test("exportBrowserBundle_rejects_artifact_file_beneath_symlinked_directory", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "faust-browser-export-"));
  const artifactDirectory = await makeArtifact(root);
  const outsideDirectory = path.join(root, "outside");
  await mkdir(outsideDirectory);
  await writeFile(path.join(outsideDirectory, "dsp.wasm"), "wasm");
  await symlink(outsideDirectory, path.join(artifactDirectory, "linked"));
  const manifestPath = path.join(artifactDirectory, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  manifest.files.dsp.path = "linked/dsp.wasm";
  await writeFile(manifestPath, JSON.stringify(manifest));

  await assert.rejects(
    exportBrowserBundle({
      artifactDirectory,
      outputDirectory: path.join(root, "site"),
    }),
    /symbolic link.*linked/,
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
    /manifest\.schemaVersion: requires 1/,
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
