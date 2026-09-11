import { cp, mkdir, rename, rm, mkdtemp, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const web = path.dirname(directory);

export async function exportSite({ model, engine, output }) {
  for (const [root, files] of [
    [
      model,
      [
        "manifest.json",
        "frontend.onnx",
        "sketch.onnx",
        "conditioning.onnx",
        "velocity.onnx",
        "preset.fxp",
      ],
    ],
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
    await mkdir(path.join(staging, "surge"));
    for (const name of [
      "index.html",
      "app.mjs",
      "worker.mjs",
      "decode.mjs",
      "author.mjs",
      "audio.mjs",
    ]) {
      await cp(path.join(directory, name), path.join(staging, "surge", name));
    }
    await mkdir(path.join(staging, "fdn"));
    await cp(
      path.join(web, "fdn", "noise.mjs"),
      path.join(staging, "fdn", "noise.mjs"),
    );
    for (const name of ["rk4.mjs", "guidance.mjs"])
      await cp(path.join(web, name), path.join(staging, name));
    await cp(model, path.join(staging, "model"), { recursive: true });
    await cp(engine, path.join(staging, "engine"), { recursive: true });
    await cp(
      path.join(web, "..", "surgewasm", "runtime.mjs"),
      path.join(staging, "engine", "runtime.mjs"),
    );
    await cp(
      path.join(web, "node_modules", "onnxruntime-web", "dist"),
      path.join(staging, "ort"),
      { recursive: true },
    );
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
      !["--model", "--engine", "--output"].includes(args[index]) ||
      !args[index + 1]
    )
      throw new Error("Usage: --model DIR --engine DIR --output DIR");
    options[args[index].slice(2)] = path.resolve(args[index + 1]);
  }
  if (Object.keys(options).length !== 3)
    throw new Error("--model, --engine and --output are required");
  await exportSite(options);
  console.log(`Static Surge evaluation: ${options.output}/surge/index.html`);
}
