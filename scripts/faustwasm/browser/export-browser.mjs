#!/usr/bin/env node

import { createHash } from "node:crypto";
import { cp, lstat, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const browserDirectory = path.dirname(fileURLToPath(import.meta.url));
const defaultRuntimePath = path.join(browserDirectory, "..", "runtime.mjs");
const defaultFaustModulePath = path.join(
  browserDirectory,
  "node_modules",
  "@grame",
  "faustwasm",
  "dist",
  "esm",
  "index.js",
);
const defaultFaustLicensePath = path.join(path.dirname(defaultFaustModulePath), "..", "..", "COPYING.txt");
const playerAssets = ["capture-worklet.js", "controller.mjs", "index.html", "styles.css"];

function isWithin(parent, candidate) {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

async function readManifest(artifactDirectory) {
  const manifestPath = path.join(artifactDirectory, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  if (manifest.schemaVersion !== 1) {
    throw new Error(`Browser export requires artifact schemaVersion 1, received ${manifest.schemaVersion}`);
  }
  return manifest;
}

async function requireAbsent(filePath) {
  try {
    await lstat(filePath);
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`Browser output already exists: ${filePath}`);
}

async function verifyArtifactFiles(artifactDirectory, files) {
  for (const file of Object.values(files)) {
    if (!file) continue;
    const filePath = path.resolve(artifactDirectory, file.path);
    if (!isWithin(artifactDirectory, filePath)) {
      throw new Error(`Artifact file path escapes its directory: ${file.path}`);
    }
    let bytes;
    try {
      bytes = await readFile(filePath);
    } catch (error) {
      throw new Error(`Artifact file is missing: ${file.path}`, { cause: error });
    }
    const actualHash = createHash("sha256").update(bytes).digest("hex");
    if (actualHash !== file.sha256) {
      throw new Error(`${file.path} hash mismatch: expected ${file.sha256}, received ${actualHash}`);
    }
  }
}

export async function exportBrowserBundle({
  artifactDirectory,
  faustLicensePath = defaultFaustLicensePath,
  faustModulePath = defaultFaustModulePath,
  outputDirectory,
  runtimePath = defaultRuntimePath,
}) {
  const manifest = await readManifest(artifactDirectory);
  await verifyArtifactFiles(artifactDirectory, manifest.files);
  if (isWithin(artifactDirectory, outputDirectory)) {
    throw new Error("Browser output must be outside the artifact directory");
  }
  await requireAbsent(outputDirectory);
  const runtimeSource = await readFile(runtimePath, "utf8");
  const exportedRuntime = runtimeSource.replaceAll(
    "../../node_modules/@grame/faustwasm/dist/esm/index.js",
    "./vendor/faustwasm.mjs",
  );
  if (exportedRuntime.includes("node_modules/@grame/faustwasm")) {
    throw new Error("Shared runtime uses an unsupported FaustWasm module path");
  }

  await cp(artifactDirectory, outputDirectory, { recursive: true });
  await Promise.all(
    playerAssets.map((asset) => cp(path.join(browserDirectory, asset), path.join(outputDirectory, asset))),
  );
  await writeFile(path.join(outputDirectory, "runtime.mjs"), exportedRuntime);
  await mkdir(path.join(outputDirectory, "vendor"));
  await Promise.all([
    cp(faustLicensePath, path.join(outputDirectory, "vendor", "FAUSTWASM-COPYING.txt")),
    cp(faustModulePath, path.join(outputDirectory, "vendor", "faustwasm.mjs")),
  ]);
}

function parseArguments(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const option = argv[index];
    const value = argv[index + 1];
    if (!option?.startsWith("--") || value === undefined) {
      throw new Error("Usage: export-browser.mjs --artifact DIR --output DIR [--runtime FILE] [--faust-module FILE]");
    }
    values.set(option, value);
  }
  if (!values.has("--artifact") || !values.has("--output")) {
    throw new Error("Usage: export-browser.mjs --artifact DIR --output DIR [--runtime FILE] [--faust-module FILE]");
  }
  return {
    artifactDirectory: values.get("--artifact"),
    faustModulePath: values.get("--faust-module") ?? defaultFaustModulePath,
    outputDirectory: values.get("--output"),
    runtimePath: values.get("--runtime") ?? defaultRuntimePath,
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  exportBrowserBundle(parseArguments(process.argv.slice(2)))
    .then(() => console.log("Browser bundle exported"))
    .catch((error) => {
      console.error(error.message);
      process.exitCode = 1;
    });
}
