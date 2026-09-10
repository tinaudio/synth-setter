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
const RESERVED_BUNDLE_PATHS = new Set([
  ...playerAssets,
  "manifest.json",
  "runtime.mjs",
  "vendor",
]);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const SUPPORTED_FAUSTWASM_VERSION = "0.18.3";

function manifestError(field, requirement) {
  throw new Error(`Invalid manifest.${field}: ${requirement}`);
}

function requireNonEmptyString(value, field) {
  if (typeof value !== "string" || value.trim() === "") {
    manifestError(field, "must be a non-empty string");
  }
}

function validateArtifactFile(file, field) {
  if (!file || typeof file !== "object" || Array.isArray(file)) {
    manifestError(field, "must be an object");
  }
  requireNonEmptyString(file.path, `${field}.path`);
  const normalizedPath = path.normalize(file.path);
  const firstSegment = normalizedPath.split(path.sep)[0];
  if (
    path.isAbsolute(file.path) ||
    normalizedPath === ".." ||
    normalizedPath.startsWith(`..${path.sep}`)
  ) {
    manifestError(`${field}.path`, "must stay within the artifact directory");
  }
  if (RESERVED_BUNDLE_PATHS.has(normalizedPath) || RESERVED_BUNDLE_PATHS.has(firstSegment)) {
    manifestError(`${field}.path`, "conflicts with a browser bundle asset");
  }
  if (typeof file.sha256 !== "string" || !SHA256_PATTERN.test(file.sha256)) {
    manifestError(`${field}.sha256`, "must be a lowercase SHA-256 digest");
  }
}

function validateDspMetadata(metadata, field, expectedOutputs) {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    manifestError(field, "must be an object");
  }
  requireNonEmptyString(metadata.name, `${field}.name`);
  if (!Number.isInteger(metadata.inputs) || metadata.inputs < 0) {
    manifestError(`${field}.inputs`, "must be a non-negative integer");
  }
  if (metadata.outputs !== expectedOutputs) {
    manifestError(`${field}.outputs`, "must match manifest.outputs");
  }
  if (!Array.isArray(metadata.ui)) manifestError(`${field}.ui`, "must be an array");
}

function validateParameter(parameter, index) {
  const field = `parameters[${index}]`;
  if (!parameter || typeof parameter !== "object" || Array.isArray(parameter)) {
    manifestError(field, "must be an object");
  }
  requireNonEmptyString(parameter.canonicalAddress, `${field}.canonicalAddress`);
  requireNonEmptyString(parameter.wasmAddress, `${field}.wasmAddress`);
  if (!Number.isFinite(parameter.min)) manifestError(`${field}.min`, "must be finite");
  if (!Number.isFinite(parameter.max)) manifestError(`${field}.max`, "must be finite");
  if (parameter.min > parameter.max) manifestError(field, "min must not exceed max");
  if (parameter.kind !== "continuous" && parameter.kind !== "discrete") {
    manifestError(`${field}.kind`, 'must be "continuous" or "discrete"');
  }
  if (parameter.kind === "discrete" && (!Number.isInteger(parameter.min) || !Number.isInteger(parameter.max))) {
    manifestError(field, "discrete bounds must be integers");
  }
}

function validateManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    throw new Error("Invalid manifest: must be an object");
  }
  if (manifest.schemaVersion !== 1) {
    manifestError("schemaVersion", `requires 1, received ${manifest.schemaVersion}`);
  }
  requireNonEmptyString(manifest.identity, "identity");
  if (manifest.faustwasmVersion !== SUPPORTED_FAUSTWASM_VERSION) {
    manifestError("faustwasmVersion", `must be ${SUPPORTED_FAUSTWASM_VERSION}`);
  }
  requireNonEmptyString(manifest.libfaustVersion, "libfaustVersion");
  if (typeof manifest.compileOptions !== "string") {
    manifestError("compileOptions", "must be a string");
  }
  if (typeof manifest.sourceSha256 !== "string" || !SHA256_PATTERN.test(manifest.sourceSha256)) {
    manifestError("sourceSha256", "must be a lowercase SHA-256 digest");
  }
  if (manifest.mode !== "mono" && manifest.mode !== "poly") {
    manifestError("mode", 'must be "mono" or "poly"');
  }
  if (!Number.isInteger(manifest.voices) || manifest.voices < 0) {
    manifestError("voices", "must be a non-negative integer");
  }
  if (manifest.mode === "mono" && manifest.voices !== 0) {
    manifestError("voices", "must be 0 for mono artifacts");
  }
  if (manifest.mode === "poly" && manifest.voices < 1) {
    manifestError("voices", "must be positive for poly artifacts");
  }
  if (!Number.isInteger(manifest.outputs) || manifest.outputs < 1) {
    manifestError("outputs", "must be a positive integer");
  }
  if (!Array.isArray(manifest.parameters)) {
    manifestError("parameters", "must be an array");
  }
  manifest.parameters.forEach(validateParameter);
  const canonicalAddresses = manifest.parameters.map((parameter) => parameter.canonicalAddress);
  const wasmAddresses = manifest.parameters.map((parameter) => parameter.wasmAddress);
  if (new Set(canonicalAddresses).size !== canonicalAddresses.length) {
    manifestError("parameters", "canonicalAddress values must be unique");
  }
  if (new Set(wasmAddresses).size !== wasmAddresses.length) {
    manifestError("parameters", "wasmAddress values must be unique");
  }
  if (!manifest.files || typeof manifest.files !== "object" || Array.isArray(manifest.files)) {
    manifestError("files", "must be an object");
  }
  const fileKinds = Object.keys(manifest.files);
  const unsupportedFileKind = fileKinds.find((kind) => !["dsp", "effect", "mixer"].includes(kind));
  if (unsupportedFileKind) manifestError(`files.${unsupportedFileKind}`, "is not supported");
  validateArtifactFile(manifest.files.dsp, "files.dsp");
  if (manifest.mode === "poly") {
    validateArtifactFile(manifest.files.mixer, "files.mixer");
  } else if (manifest.files.mixer || manifest.files.effect) {
    manifestError("files", "mono artifacts may contain only dsp");
  }
  if (manifest.files.effect) validateArtifactFile(manifest.files.effect, "files.effect");
  const filePaths = Object.values(manifest.files)
    .filter(Boolean)
    .map((file) => file.path);
  if (new Set(filePaths).size !== filePaths.length) {
    manifestError("files", "paths must be unique");
  }
  validateDspMetadata(manifest.dspMeta, "dspMeta", manifest.outputs);
  if (manifest.files.effect) {
    validateDspMetadata(manifest.effectMeta, "effectMeta", manifest.outputs);
  }
  if (!manifest.files.effect && manifest.effectMeta !== undefined) {
    manifestError("effectMeta", "requires files.effect");
  }
}

function isWithin(parent, candidate) {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

async function rejectSymbolicLink(filePath, displayPath) {
  const fileStat = await lstat(filePath);
  if (fileStat.isSymbolicLink()) {
    throw new Error(`Artifact path contains a symbolic link: ${displayPath}`);
  }
  return fileStat;
}

async function readManifest(artifactDirectory) {
  await rejectSymbolicLink(artifactDirectory, artifactDirectory);
  const manifestPath = path.join(artifactDirectory, "manifest.json");
  await rejectSymbolicLink(manifestPath, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  validateManifest(manifest);
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

async function readVerifiedArtifactFiles(artifactDirectory, files) {
  const verifiedFiles = new Map();
  for (const file of Object.values(files)) {
    if (!file) continue;
    const filePath = path.resolve(artifactDirectory, file.path);
    if (!isWithin(artifactDirectory, filePath)) {
      throw new Error(`Artifact file path escapes its directory: ${file.path}`);
    }
    try {
      let currentPath = path.resolve(artifactDirectory);
      for (const segment of path.relative(currentPath, filePath).split(path.sep)) {
        currentPath = path.join(currentPath, segment);
        const fileStat = await rejectSymbolicLink(currentPath, path.relative(artifactDirectory, currentPath));
        if (currentPath === filePath && !fileStat.isFile()) {
          throw new Error(`Artifact file is not a regular file: ${file.path}`);
        }
      }
      const bytes = await readFile(filePath);
      const actualHash = createHash("sha256").update(bytes).digest("hex");
      if (actualHash !== file.sha256) {
        throw new Error(`${file.path} hash mismatch: expected ${file.sha256}, received ${actualHash}`);
      }
      verifiedFiles.set(file.path, bytes);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      throw new Error(`Artifact file is missing: ${file.path}`, { cause: error });
    }
  }
  return verifiedFiles;
}

export async function exportBrowserBundle({
  artifactDirectory,
  faustLicensePath = defaultFaustLicensePath,
  faustModulePath = defaultFaustModulePath,
  outputDirectory,
  runtimePath = defaultRuntimePath,
}) {
  const manifest = await readManifest(artifactDirectory);
  const artifactFiles = await readVerifiedArtifactFiles(artifactDirectory, manifest.files);
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

  await mkdir(outputDirectory, { recursive: true });
  await writeFile(path.join(outputDirectory, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await Promise.all(
    [...artifactFiles].map(async ([relativePath, bytes]) => {
      const destination = path.join(outputDirectory, relativePath);
      await mkdir(path.dirname(destination), { recursive: true });
      await writeFile(destination, bytes);
    }),
  );
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
