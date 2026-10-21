import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import {
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

const execute = promisify(execFile);

const modelFiles = [
  "manifest.json",
  "frontend.onnx",
  "sketch.onnx",
  "conditioning.onnx",
  "velocity.onnx",
  "preset.fxp",
];
const engineFiles = [
  "manifest.json",
  "surge-host.mjs",
  "surge-host.wasm",
  "surge-xt.clap.wasm",
];

for (const destination of ["model", "engine", "alias"]) {
  test(`site export rejects ${destination} descendants before writing to sources`, async () => {
    const root = await mkdtemp(join(tmpdir(), "surge-site-containment-"));
    try {
      const model = join(root, "model");
      const engine = join(root, "engine");
      await mkdir(model);
      await mkdir(engine);
      await symlink(model, join(root, "alias"), "dir");
      await assert.rejects(
        execute(process.execPath, [
          fileURLToPath(new URL("./export-site.mjs", import.meta.url)),
          "--model",
          model,
          "--engine",
          engine,
          "--output",
          join(root, destination, "new", "site"),
        ]),
        (error) => {
          assert.match(
            error.stderr,
            /output must not be inside a copied source tree/,
          );
          return true;
        },
      );
      assert.deepEqual(await readdir(model), []);
      assert.deepEqual(await readdir(engine), []);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
}

test("site export makes the poster the root landing page with attribution", async () => {
  const root = await mkdtemp(join(tmpdir(), "surge-poster-export-"));
  try {
    const model = join(root, "model");
    const engine = join(root, "engine");
    const output = join(root, "site");
    await mkdir(model);
    await mkdir(engine);
    await Promise.all(
      modelFiles.map((name) => writeFile(join(model, name), "model")),
    );
    await Promise.all(
      engineFiles.map((name) => writeFile(join(engine, name), "engine")),
    );

    await execute(process.execPath, [
      fileURLToPath(new URL("./export-site.mjs", import.meta.url)),
      "--model",
      model,
      "--engine",
      engine,
      "--output",
      output,
    ]);

    assert.match(
      await readFile(join(output, "index.html"), "utf8"),
      /Load interactive demo/,
    );
    assert.match(
      await readFile(join(output, "poster-demo.css"), "utf8"),
      /demo-frame/,
    );
    assert.equal(
      await readFile(join(output, "styles.css"), "utf8"),
      await readFile(
        fileURLToPath(new URL("../poster/styles.css", import.meta.url)),
        "utf8",
      ),
    );
    assert.match(
      await readFile(
        join(output, "LICENSE-poster-template-MPL-2.0.txt"),
        "utf8",
      ),
      /Mozilla Public License Version 2\.0/,
    );
    assert.match(
      await readFile(join(output, "surge", "index.html"), "utf8"),
      /Surge music-sketch evaluation/,
    );
    const redirects = await readFile(join(output, "_redirects"), "utf8");
    assert.doesNotMatch(redirects, /^\/\s/m);
    assert.match(
      redirects,
      /^\/model\/velocity\.onnx https:\/\/.*\.r2\.dev\/model\/velocity\.onnx 302$/m,
    );
    assert.match(
      await readFile(join(output, "_headers"), "utf8"),
      /Cross-Origin-Embedder-Policy: require-corp/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
