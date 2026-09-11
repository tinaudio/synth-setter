#!/usr/bin/env node
// Assemble a self-contained static site: page, JS ports, ONNX Runtime Web, the FaustWasm
// runtime, one exported Faust artifact, and one exported model bundle.
import { cp, mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const fdnDirectory = path.dirname(fileURLToPath(import.meta.url));
const webDirectory = path.join(fdnDirectory, "..");
const packageRoot = path.join(webDirectory, "..");
const PAGE_ASSETS = ["index.html", "app.mjs", "audio.mjs", "author.mjs", "bundle.mjs", "decode.mjs", "dsp.mjs", "fft.mjs", "inference.mjs", "metrics.mjs", "noise.mjs", "patch.mjs", "render.mjs", "sketch.mjs", "wav.mjs"];
const MODEL_FILES = ["frontend.onnx", "conditioning.onnx", "velocity.onnx"];

async function requireDirectory(directory, label) {
  try {
    if (!(await stat(directory)).isDirectory()) throw new Error();
  } catch {
    throw new Error(`${label} directory is missing: ${directory}`);
  }
}

// Model manifests key files by filename; Faust artifact manifests carry an explicit path.
async function requireManifestFiles(directory, label, expectedKeys) {
  const manifest = JSON.parse(await readFile(path.join(directory, "manifest.json"), "utf8"));
  for (const key of expectedKeys) {
    const entry = manifest.files?.[key];
    if (!entry?.sha256) throw new Error(`${label} manifest lacks files.${key}`);
    await stat(path.join(directory, entry.path ?? key));
  }
  return manifest;
}

export async function exportSite({ modelDirectory, faustDirectory, outputDirectory, ortDirectory }) {
  await requireDirectory(modelDirectory, "model bundle");
  await requireDirectory(faustDirectory, "Faust artifact");
  const modelManifest = await requireManifestFiles(modelDirectory, "model bundle", MODEL_FILES);
  await requireManifestFiles(faustDirectory, "Faust artifact", ["dsp"]);
  const ort = ortDirectory ?? path.join(webDirectory, "node_modules", "onnxruntime-web", "dist");
  await requireDirectory(ort, "onnxruntime-web dist");

  await mkdir(path.join(outputDirectory, "fdn", "fixtures"), { recursive: true });
  for (const asset of PAGE_ASSETS) {
    await cp(path.join(fdnDirectory, asset), path.join(outputDirectory, "fdn", asset));
  }
  await cp(path.join(fdnDirectory, "fixtures", "octave_bands.json"), path.join(outputDirectory, "fdn", "fixtures", "octave_bands.json"));
  for (const shared of ["rk4.mjs", "guidance.mjs"]) {
    await cp(path.join(webDirectory, shared), path.join(outputDirectory, shared));
  }
  await writeFile(
    path.join(outputDirectory, "index.html"),
    '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=fdn/index.html">\n',
  );
  await mkdir(path.join(outputDirectory, "ort"), { recursive: true });
  for (const entry of await readdir(ort)) {
    if (/^ort(\.wasm\.min\.mjs|-wasm-simd-threaded(\.(asyncify|jsep|jspi))?\.(mjs|wasm))$/.test(entry)) {
      await cp(path.join(ort, entry), path.join(outputDirectory, "ort", entry));
    }
  }
  await mkdir(path.join(outputDirectory, "faust", "vendor"), { recursive: true });
  await cp(path.join(packageRoot, "faustwasm", "runtime.mjs"), path.join(outputDirectory, "faust", "runtime.mjs"));
  await cp(path.join(packageRoot, "faustwasm", "vendor", "faustwasm.mjs"), path.join(outputDirectory, "faust", "vendor", "faustwasm.mjs"));
  await cp(path.join(packageRoot, "faustwasm", "vendor", "package.json"), path.join(outputDirectory, "faust", "vendor", "package.json"));
  await cp(faustDirectory, path.join(outputDirectory, "faust"), { recursive: true });
  await cp(modelDirectory, path.join(outputDirectory, "model"), { recursive: true });
  return { pageUrl: "fdn/index.html", paramSpecName: modelManifest.paramSpecName };
}

function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!flag?.startsWith("--") || value === undefined) throw new Error(`Usage: --model DIR --faust DIR --output DIR [--ort DIR]`);
    options[flag.slice(2)] = value;
  }
  for (const required of ["model", "faust", "output"]) {
    if (!options[required]) throw new Error(`--${required} is required`);
  }
  return options;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const options = parseArguments(process.argv.slice(2));
  exportSite({
    modelDirectory: path.resolve(options.model),
    faustDirectory: path.resolve(options.faust),
    outputDirectory: path.resolve(options.output),
    ortDirectory: options.ort ? path.resolve(options.ort) : undefined,
  })
    .then((result) => console.log(`Site exported; open ${result.pageUrl} for ${result.paramSpecName}`))
    .catch((error) => {
      console.error(error.message);
      process.exit(1);
    });
}
