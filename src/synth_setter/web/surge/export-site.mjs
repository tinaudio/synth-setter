import {
  cp,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const web = path.dirname(directory);
const poster = path.join(web, "poster");
const ort = path.join(web, "node_modules", "onnxruntime-web", "dist");

async function canonicalProspectivePath(destination) {
  const absolute = path.resolve(destination);
  let ancestor = absolute;
  while (true) {
    try {
      return path.resolve(
        await realpath(ancestor),
        path.relative(ancestor, absolute),
      );
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      const parent = path.dirname(ancestor);
      if (parent === ancestor) throw error;
      ancestor = parent;
    }
  }
}

export async function exportSite({ model, engine, output, comparison }) {
  const destination = await canonicalProspectivePath(output);
  for (const source of [
    model,
    engine,
    ort,
    ...(comparison ? [comparison] : []),
  ]) {
    const relative = path.relative(await realpath(source), destination);
    if (
      !path.isAbsolute(relative) &&
      relative !== ".." &&
      !relative.startsWith(`..${path.sep}`)
    ) {
      throw new Error("output must not be inside a copied source tree");
    }
  }
  const models = [model];
  if (comparison) {
    const catalog = JSON.parse(
      await readFile(path.join(comparison, "catalog.json"), "utf8"),
    );
    if (
      !Array.isArray(catalog.models) ||
      catalog.models.length !== 4 ||
      new Set(catalog.models.map((entry) => entry.id)).size !== 4 ||
      catalog.models.some((entry) => !/^[a-z0-9-]+$/.test(entry.id))
    )
      throw new Error("invalid four-model catalog");
    models.push(
      ...catalog.models.map((entry) => path.join(comparison, entry.id)),
    );
  }
  for (const root of models) {
    const manifest = JSON.parse(
      await readFile(path.join(root, "manifest.json"), "utf8"),
    );
    for (const name of [
      "frontend.onnx",
      "conditioning.onnx",
      "velocity.onnx",
      "preset.fxp",
      ...(manifest.sketch ? ["sketch.onnx"] : []),
    ])
      await stat(path.join(root, name));
  }
  for (const [root, files] of [
    [
      engine,
      [
        "manifest.json",
        "surge-host.mjs",
        "surge-host.wasm",
        "surge-xt.clap.wasm",
      ],
    ],
  ]) {
    for (const file of files) await stat(path.join(root, file));
  }
  try {
    await stat(output);
    throw new Error(`output already exists: ${output}`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await mkdir(path.dirname(output), { recursive: true });
  const staging = await mkdtemp(
    path.join(path.dirname(output), ".surge-site-"),
  );
  try {
    for (const [source, destination] of [
      ["index.html", "index.html"],
      ["poster-demo.css", "poster-demo.css"],
      ["poster.mjs", "poster.mjs"],
      ["styles.css", "styles.css"],
      ["LICENSE-template-MPL-2.0.txt", "LICENSE-poster-template-MPL-2.0.txt"],
    ]) {
      await cp(path.join(poster, source), path.join(staging, destination));
    }
    for (const name of ["_headers", "_redirects"])
      await cp(path.join(directory, name), path.join(staging, name));
    await mkdir(path.join(staging, "surge"));
    for (const name of [
      "index.html",
      "app.mjs",
      "start.mjs",
      "comparison.mjs",
      "worker.mjs",
      "decode.mjs",
      "author.mjs",
      "audio.mjs",
      "spectrogram.mjs",
    ]) {
      await cp(path.join(directory, name), path.join(staging, "surge", name));
    }
    await mkdir(path.join(staging, "fdn"));
    for (const name of ["noise.mjs", "dsp.mjs", "fft.mjs"])
      await cp(path.join(web, "fdn", name), path.join(staging, "fdn", name));
    for (const name of ["rk4.mjs", "guidance.mjs"])
      await cp(path.join(web, name), path.join(staging, name));
    await cp(model, path.join(staging, "model"), { recursive: true });
    if (comparison)
      await cp(comparison, path.join(staging, "models"), { recursive: true });
    await cp(engine, path.join(staging, "engine"), { recursive: true });
    await cp(
      path.join(web, "..", "surgewasm", "runtime.mjs"),
      path.join(staging, "engine", "runtime.mjs"),
    );
    await cp(ort, path.join(staging, "ort"), { recursive: true });
    await rename(staging, output);
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const args = process.argv.slice(2);
  const options = {};
  for (let index = 0; index < args.length; index += 2) {
    if (
      !["--model", "--engine", "--output", "--comparison"].includes(
        args[index],
      ) ||
      !args[index + 1]
    )
      throw new Error(
        "Usage: --model DIR --engine DIR --output DIR [--comparison DIR]",
      );
    options[args[index].slice(2)] = path.resolve(args[index + 1]);
  }
  if (!["model", "engine", "output"].every((name) => options[name]))
    throw new Error("--model, --engine and --output are required");
  await exportSite(options);
  console.log(
    `Static poster and Surge evaluation: ${options.output}/index.html`,
  );
}
