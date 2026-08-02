import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import {
  ManagerLocal,
  Package,
  RegistryType,
} from "@open-audio-stack/core";

assert.notEqual(process.env.NODE_TLS_REJECT_UNAUTHORIZED, "0");

const [root, lockPath] = process.argv.slice(2);
const lock = JSON.parse(await readFile(lockPath, "utf8"));
const [packageReference] = Object.keys(lock);
const versionSeparator = packageReference.lastIndexOf("@");
const packageName = packageReference.slice(0, versionSeparator);
const version = packageReference.slice(versionSeparator + 1);
const artifacts = lock[packageReference].artifacts;

const manager = new ManagerLocal(RegistryType.Plugins, {
  appDir: path.join(root, "app"),
  artifactLockPath: lockPath,
  pluginsDir: path.join(root, "plugins"),
});
manager.packages.set(
  packageName,
  new Package(packageName, {
    [version]: {
      files: artifacts.map((artifact) => ({
        ...artifact,
        systems: artifact.systems.map((system) => ({ type: system })),
      })),
    },
  }),
);

await manager.install(packageName, version);
