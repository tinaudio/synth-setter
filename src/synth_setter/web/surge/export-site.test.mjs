import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readdir, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

const execute = promisify(execFile);

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
          "--model", model,
          "--engine", engine,
          "--output", join(root, destination, "new", "site"),
        ]),
        (error) => {
          assert.match(error.stderr, /output must not be inside a copied source tree/);
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
