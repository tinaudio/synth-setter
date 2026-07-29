#!/usr/bin/env node

import assert from "node:assert/strict";
import { exec, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:https";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";
import sudoPrompt from "@vscode/sudo-prompt";
import {
  getArchitecture,
  getSystem,
  ManagerLocal,
  Package,
  RegistryType,
} from "@open-audio-stack/core";

const projectRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const patchScript = path.join(projectRoot, "scripts/studiorack/patch-core.mjs");
const managerSourceFixture = path.join(
  projectRoot,
  "tests/fixtures/studiorack/ManagerLocal.patch-preconditions.txt",
);
const packageReference = "example/synth@1.2.3";

/**
 * @typedef {object} ArtifactOptions
 * @property {string} [sha256] Artifact digest.
 * @property {string} [type] Studiorack artifact type.
 * @property {string} url Artifact download URL.
 */

/**
 * @typedef {object} LockedArtifact
 * @property {string[]} architectures Supported CPU architectures.
 * @property {string} sha256 Expected artifact digest.
 * @property {string[]} systems Supported operating systems.
 * @property {string} type Studiorack artifact type.
 * @property {string} url Artifact download URL.
 */

/**
 * Build a host-compatible artifact fixture.
 * @param {ArtifactOptions} options Artifact fields to override.
 * @returns {LockedArtifact} Artifact for the current host platform.
 */
function artifact({ sha256 = "a".repeat(64), type = "archive", url }) {
  return {
    architectures: [getArchitecture()],
    sha256,
    systems: [getSystem()],
    type,
    url,
  };
}

/**
 * @typedef {object} ArtifactLockOptions
 * @property {LockedArtifact[]} artifacts Artifacts pinned for the fixture package.
 * @property {string} filename Lock filename under the temporary root.
 * @property {string} root Temporary fixture root.
 */

/**
 * Write an artifact-lock fixture under the temporary root.
 * @param {ArtifactLockOptions} options Lock contents and destination.
 * @returns {Promise<string>} Absolute lock-file path.
 */
async function writeArtifactLock({ artifacts, filename, root }) {
  const lock = path.join(root, filename);
  await writeFile(
    lock,
    JSON.stringify({
      [packageReference]: { artifacts },
    }),
  );
  return lock;
}

/**
 * Build a valid artifact-lock document.
 * @returns {Record<string, {artifacts: LockedArtifact[]}>} Fixture keyed by package reference.
 */
function validArtifactLock() {
  return {
    [packageReference]: {
      artifacts: [
        artifact({ url: "https://example.test/locked.zip" }),
      ],
    },
  };
}

/**
 * Assert that ManagerLocal rejects an artifact-lock document.
 * @param {unknown} document Artifact-lock document or raw JSON text.
 * @param {RegExp} expected Expected rejection pattern.
 * @returns {Promise<void>} Promise resolved after rejection is verified.
 */
async function assertArtifactLockRejected(document, expected) {
  await withTemporaryRoot(async (root) => {
    const lock = path.join(root, "studiorack.lock.json");
    await writeFile(
      lock,
      typeof document === "string" ? document : JSON.stringify(document),
    );
    const pluginsDir = path.join(root, "plugins");
    const manager = new ManagerLocal(RegistryType.Plugins, {
      appDir: path.join(root, "app"),
      artifactLockPath: lock,
      pluginsDir,
    });
    manager.packages.set(
      "example/synth",
      new Package("example/synth", {
        "1.2.3": {
          files: [
            {
              ...artifact({ url: "https://example.test/locked.zip" }),
              systems: [{ type: getSystem() }],
            },
          ],
        },
      }),
    );
    await mkdir(path.join(pluginsDir, "VST3/example/synth/1.2.3"), {
      recursive: true,
    });

    await assert.rejects(manager.install("example/synth", "1.2.3"), expected);
  });
}

/**
 * @typedef {object} ManagerFixtureOptions
 * @property {string} [lockedSha256] Digest stored in the lock.
 * @property {string} lockedUrl URL stored in the lock.
 * @property {string} [liveSha256] Digest exposed by the registry fixture.
 * @property {string} liveUrl URL exposed by the registry fixture.
 * @property {string} [type] Studiorack artifact type.
 */

/**
 * Build a manager with independent locked and live artifact metadata.
 * @param {string} root Temporary fixture root.
 * @param {ManagerFixtureOptions} options Locked and live artifact fields.
 * @returns {Promise<{manager: ManagerLocal, pluginsDir: string}>} Manager and plugin root.
 */
async function managerFixture(
  root,
  {
    lockedSha256 = "a".repeat(64),
    lockedUrl,
    liveSha256 = "a".repeat(64),
    liveUrl,
    type = "archive",
  },
) {
  const lock = await writeArtifactLock({
    artifacts: [artifact({ sha256: lockedSha256, type, url: lockedUrl })],
    filename: "studiorack.lock.json",
    root,
  });
  const pluginsDir = path.join(root, "plugins");
  const manager = new ManagerLocal(RegistryType.Plugins, {
    appDir: path.join(root, "app"),
    artifactLockPath: lock,
    pluginsDir,
  });
  manager.packages.set(
    "example/synth",
    new Package("example/synth", {
      "1.2.3": {
        files: [
          {
            ...artifact({ sha256: liveSha256, type, url: liveUrl }),
            systems: [{ type: getSystem() }],
          },
        ],
      },
    }),
  );
  return { manager, pluginsDir };
}

/**
 * Encode a one-package registry as a data URL.
 * @param {string} url Artifact URL exposed by the registry.
 * @param {string} [type] Studiorack artifact type.
 * @returns {string} Data URL containing the registry document.
 */
function registryDataUrl(url, type = "installer") {
  const file = {
    ...artifact({ type, url }),
    systems: [{ type: getSystem() }],
  };
  const registry = {
    apps: {},
    plugins: {
      "example/synth": {
        versions: {
          "1.2.3": {
            author: "synth-setter",
            changes: "test fixture",
            date: "2026-01-01T00:00:00.000Z",
            description: "elevated boundary fixture",
            files: [{ ...file, size: 8 }],
            image: "https://example.test/plugin.jpg",
            license: "mit",
            name: "Example Synth",
            tags: ["synth"],
            type: "instrument",
            url: "https://example.test/plugin",
          },
        },
      },
    },
    presets: {},
    projects: {},
  };
  return `data:application/json,${encodeURIComponent(JSON.stringify(registry))}`;
}

/**
 * @typedef {object} ElevatedManagerFixtureOptions
 * @property {string} childUrl Artifact URL exposed to the admin child.
 * @property {string} configuredLockUrl Artifact URL stored in the configured lock.
 * @property {string} parentUrl Artifact URL exposed to the parent manager.
 */

/**
 * Build a manager that crosses the real elevated-child boundary.
 * @param {string} root Temporary fixture root.
 * @param {ElevatedManagerFixtureOptions} options Parent, child, and lock URLs.
 * @returns {Promise<ManagerLocal>} Configured parent manager.
 */
async function elevatedManagerFixture(
  root,
  { childUrl, configuredLockUrl, parentUrl },
) {
  const appDir = path.join(root, "app");
  await mkdir(appDir, { recursive: true });
  await writeFile(
    path.join(appDir, "config.json"),
    JSON.stringify({
      registries: [{ name: "child", url: registryDataUrl(childUrl) }],
    }),
  );
  const configuredLock = await writeArtifactLock({
    artifacts: [
      artifact({ type: "installer", url: configuredLockUrl }),
    ],
    filename: "configured.lock.json",
    root,
  });
  const manager = new ManagerLocal(RegistryType.Plugins, {
    appDir,
    artifactLockPath: configuredLock,
    pluginsDir: path.join(root, "plugins"),
  });
  manager.packages.set(
    "example/synth",
    new Package("example/synth", {
      "1.2.3": {
        files: [
          {
            ...artifact({ type: "installer", url: parentUrl }),
            systems: [{ type: getSystem() }],
          },
        ],
      },
    }),
  );
  return manager;
}

/**
 * Run a body with elevation redirected to a local admin child.
 * @param {() => Promise<void>} body Assertions that trigger elevation.
 * @returns {Promise<number>} Number of child launches.
 */
async function withLocalAdminChild(body) {
  const originalExec = sudoPrompt.exec;
  let launches = 0;
  sudoPrompt.exec = (command, _options, callback) => {
    launches += 1;
    exec(command, { env: process.env }, callback);
  };
  try {
    await body();
    return launches;
  } finally {
    sudoPrompt.exec = originalExec;
  }
}

/**
 * Run a body inside a disposable fixture root.
 * @param {(root: string) => Promise<void>} body Fixture operations.
 * @returns {Promise<void>} Promise resolved after cleanup.
 */
async function withTemporaryRoot(body) {
  const root = await mkdtemp(path.join(tmpdir(), "synth-setter-studiorack-"));
  try {
    await body(root);
  } finally {
    await rm(root, { force: true, recursive: true });
  }
}

/**
 * Run a body against a disposable HTTPS artifact server.
 * @param {Buffer} archive Archive bytes served by the preferred endpoint.
 * @param {(server: {baseUrl: string, requests: (string | undefined)[]}) => Promise<void>} body Server assertions.
 * @returns {Promise<void>} Promise resolved after server cleanup.
 */
async function withArtifactServer(archive, body) {
  const certificateRoot = await mkdtemp(
    path.join(tmpdir(), "synth-setter-studiorack-tls-"),
  );
  const keyPath = path.join(certificateRoot, "key.pem");
  const certificatePath = path.join(certificateRoot, "certificate.pem");
  const certificate = spawnSync(
    "openssl",
    [
      "req",
      "-x509",
      "-newkey",
      "rsa:2048",
      "-nodes",
      "-subj",
      "/CN=127.0.0.1",
      "-keyout",
      keyPath,
      "-out",
      certificatePath,
      "-days",
      "1",
    ],
    { encoding: "utf8" },
  );
  assert.equal(certificate.status, 0, certificate.stderr);

  const requests = [];
  const server = createServer(
    {
      cert: await readFile(certificatePath),
      key: await readFile(keyPath),
    },
    (request, response) => {
      requests.push(request.url);
      if (request.url === "/preferred.zip") {
        response.writeHead(200, { "content-type": "application/zip" });
        response.end(archive);
        return;
      }
      response.writeHead(500);
      response.end("installer must not be selected");
    },
  );
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const previousTlsSetting = process.env.NODE_TLS_REJECT_UNAUTHORIZED;
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";
  try {
    const address = server.address();
    assert(address && typeof address !== "string");
    await body({
      baseUrl: `https://127.0.0.1:${address.port}`,
      requests,
    });
  } finally {
    if (previousTlsSetting === undefined)
      delete process.env.NODE_TLS_REJECT_UNAUTHORIZED;
    else process.env.NODE_TLS_REJECT_UNAUTHORIZED = previousTlsSetting;
    await new Promise((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
    await rm(certificateRoot, { force: true, recursive: true });
  }
}

/**
 * Build the upstream filesystem-helper patch fixture.
 * @returns {string} Unpatched helper source.
 */
function helperSource() {
  return (
    "else if (isTarFile) {\n" +
    "    return await tar.extract({ file: filePath, cwd: dirPath });\n" +
    "}\n" +
    "const pkgs = dirRead(path.join(mountPoint, '**', '*.pkg'));\n" +
    "if (fileExists(path.join(f, 'Contents', 'Info.plist'))) {\n"
  );
}

/**
 * Patch isolated helper, manager, and admin source fixtures.
 * @param {string} root Temporary fixture root.
 * @param {string} adminSource Unpatched admin source.
 * @param {string} [helperText] Unpatched helper source.
 * @returns {Promise<{admin: string, helper: string, manager: string, result: import("node:child_process").SpawnSyncReturns<string>}>} Fixture paths and patch result.
 */
async function patchFixture(root, adminSource, helperText = helperSource()) {
  const helper = path.join(root, "file.js");
  const manager = path.join(root, "ManagerLocal.js");
  const admin = path.join(root, "admin.js");
  await Promise.all([
    writeFile(helper, helperText),
    copyFile(managerSourceFixture, manager),
    writeFile(admin, adminSource),
  ]);
  const result = spawnSync(
    process.execPath,
    [patchScript, helper, manager, admin],
    {
      encoding: "utf8",
    },
  );
  return { admin, helper, manager, result };
}

test("patched core accepts a matching lock before installed-state handling", async () => {
  await withTemporaryRoot(async (root) => {
    const url = "https://example.test/locked.zip";
    const { manager, pluginsDir } = await managerFixture(root, {
      lockedUrl: url,
      liveUrl: url,
    });
    await mkdir(path.join(pluginsDir, "VST3/example/synth/1.2.3"), {
      recursive: true,
    });

    const installed = await manager.install("example/synth", "1.2.3");

    assert.equal(installed.installed, true);
  });
});

test("patched core gives the explicit environment lock precedence", async () => {
  await withTemporaryRoot(async (root) => {
    const liveUrl = "https://example.test/environment-locked.zip";
    const { manager, pluginsDir } = await managerFixture(root, {
      lockedUrl: "https://example.test/config-locked.zip",
      liveUrl,
    });
    const environmentLock = await writeArtifactLock({
      artifacts: [artifact({ url: liveUrl })],
      filename: "environment.lock.json",
      root,
    });
    await mkdir(path.join(pluginsDir, "VST3/example/synth/1.2.3"), {
      recursive: true,
    });
    const previousLock = process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK;
    process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK = environmentLock;
    try {
      const installed = await manager.install("example/synth", "1.2.3");
      assert.equal(installed.installed, true);
    } finally {
      if (previousLock === undefined)
        delete process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK;
      else
        process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK = previousLock;
    }
  });
});

test("patched core rejects registry drift before installed-state handling", async () => {
  await withTemporaryRoot(async (root) => {
    const { manager, pluginsDir } = await managerFixture(root, {
      lockedUrl: "https://example.test/locked.zip",
      liveUrl: "https://127.0.0.1:9/changed.zip",
    });
    await mkdir(path.join(pluginsDir, "VST3/example/synth/1.2.3"), {
      recursive: true,
    });

    await assert.rejects(
      manager.install("example/synth", "1.2.3"),
      new RegExp(`artifact lock mismatch for ${packageReference}`),
    );
  });
});

test("patched core rejects SHA-only registry drift", async () => {
  await withTemporaryRoot(async (root) => {
    const url = "https://example.test/locked.zip";
    const { manager, pluginsDir } = await managerFixture(root, {
      lockedSha256: "a".repeat(64),
      lockedUrl: url,
      liveSha256: "b".repeat(64),
      liveUrl: url,
    });
    await mkdir(path.join(pluginsDir, "VST3/example/synth/1.2.3"), {
      recursive: true,
    });

    await assert.rejects(
      manager.install("example/synth", "1.2.3"),
      new RegExp(`artifact lock mismatch for ${packageReference}`),
    );
  });
});

test("patched core preserves non-host artifacts in a matching lock", async () => {
  await withTemporaryRoot(async (root) => {
    const hostArtifact = artifact({
      url: "https://example.test/host.zip",
    });
    const otherArtifact = {
      ...artifact({ url: "https://example.test/other.zip" }),
      systems: [getSystem() === "linux" ? "mac" : "linux"],
    };
    const lock = await writeArtifactLock({
      artifacts: [hostArtifact, otherArtifact],
      filename: "studiorack.lock.json",
      root,
    });
    const pluginsDir = path.join(root, "plugins");
    const manager = new ManagerLocal(RegistryType.Plugins, {
      appDir: path.join(root, "app"),
      artifactLockPath: lock,
      pluginsDir,
    });
    manager.packages.set(
      "example/synth",
      new Package("example/synth", {
        "1.2.3": {
          files: [{ ...hostArtifact, systems: [{ type: getSystem() }] }],
        },
      }),
    );
    await mkdir(path.join(pluginsDir, "VST3/example/synth/1.2.3"), {
      recursive: true,
    });

    const installed = await manager.install("example/synth", "1.2.3");

    assert.equal(installed.installed, true);
  });
});

test("patched core rejects malformed artifact-lock JSON", async () => {
  await assertArtifactLockRejected("{", /invalid artifact lock JSON/);
});

test("patched core rejects a non-object artifact lock", async () => {
  await assertArtifactLockRejected([], /artifact lock root must be an object/);
});

test("patched core rejects an empty package name", async () => {
  await assertArtifactLockRejected(
    { "@1.2.3": { artifacts: [artifact({ url: "https://example.test/a" })] } },
    /package reference "@1\.2\.3" must contain a nonempty package and version/,
  );
});

test("patched core rejects an empty package version", async () => {
  await assertArtifactLockRejected(
    { "example/synth@": { artifacts: [artifact({ url: "https://example.test/a" })] } },
    /package reference "example\/synth@" must contain a nonempty package and version/,
  );
});

test("patched core rejects extra package entry fields", async () => {
  const lock = validArtifactLock();
  lock[packageReference].source = "registry";

  await assertArtifactLockRejected(
    lock,
    /example\/synth@1\.2\.3 must have exactly fields: artifacts/,
  );
});

test("patched core rejects an empty artifacts list", async () => {
  await assertArtifactLockRejected(
    { [packageReference]: { artifacts: [] } },
    /example\/synth@1\.2\.3\.artifacts must be a nonempty array/,
  );
});

test("patched core rejects extra artifact fields", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].size = 123;

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\] must have exactly fields: architectures, sha256, systems, type, url/,
  );
});

test("patched core rejects empty artifact platform lists", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].architectures = [];

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.architectures must be a nonempty array/,
  );
});

test("patched core rejects unsupported artifact architectures", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].architectures = ["mips64"];

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.architectures contains unsupported value "mips64"/,
  );
});

test("patched core rejects unsupported artifact systems", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].systems = ["freebsd"];

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.systems contains unsupported value "freebsd"/,
  );
});

test("patched core rejects unsupported artifact types", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].type = "image";

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.type contains unsupported value "image"/,
  );
});

test("patched core rejects non-HTTPS artifact URLs", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].url = "http://example.test/a";

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.url must be a valid HTTPS URL/,
  );
});

test("patched core rejects malformed artifact URLs", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].url = "https://";

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.url must be a valid HTTPS URL/,
  );
});

test("patched core rejects non-string artifact URLs", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].url = ["https://example.test/a"];

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.url must be a valid HTTPS URL/,
  );
});

test("patched core rejects uppercase SHA-256 digests", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].sha256 = "A".repeat(64);

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.sha256 must be 64 lowercase hexadecimal characters/,
  );
});

test("patched core rejects short SHA-256 digests", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].sha256 = "a".repeat(63);

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.sha256 must be 64 lowercase hexadecimal characters/,
  );
});

test("patched core rejects duplicate platform values", async () => {
  const lock = validArtifactLock();
  lock[packageReference].artifacts[0].systems = ["linux", "linux"];

  await assertArtifactLockRejected(
    lock,
    /artifacts\[0\]\.systems contains duplicate value "linux"/,
  );
});

test("patched core rejects duplicate artifacts", async () => {
  const lock = validArtifactLock();
  const original = lock[packageReference].artifacts[0];
  lock[packageReference].artifacts.push({
    url: original.url,
    type: original.type,
    systems: [...original.systems],
    sha256: original.sha256,
    architectures: [...original.architectures],
  });

  await assertArtifactLockRejected(
    lock,
    /example\/synth@1\.2\.3\.artifacts contains duplicate artifact at index 1/,
  );
});

test("patched core installs only the preferred archive from a mixed lock", async () => {
  await withTemporaryRoot(async (root) => {
    const zip = new AdmZip();
    zip.addFile(
      "Preferred.vst3/Contents/x86_64-linux/Preferred.so",
      Buffer.from("preferred archive"),
    );
    const archive = zip.toBuffer();
    const archiveSha256 = createHash("sha256").update(archive).digest("hex");

    await withArtifactServer(archive, async ({ baseUrl, requests }) => {
      const archiveArtifact = artifact({
        sha256: archiveSha256,
        url: `${baseUrl}/preferred.zip`,
      });
      const installerArtifact = artifact({
        sha256: "b".repeat(64),
        type: "installer",
        url: `${baseUrl}/forbidden.deb`,
      });
      const lock = await writeArtifactLock({
        artifacts: [installerArtifact, archiveArtifact],
        filename: "studiorack.lock.json",
        root,
      });
      const pluginsDir = path.join(root, "plugins");
      const manager = new ManagerLocal(RegistryType.Plugins, {
        appDir: path.join(root, "app"),
        artifactLockPath: lock,
        pluginsDir,
      });
      manager.packages.set(
        "example/synth",
        new Package("example/synth", {
          "1.2.3": {
            files: [installerArtifact, archiveArtifact].map((file) => ({
              ...file,
              systems: [{ type: getSystem() }],
            })),
          },
        }),
      );

      const installed = await manager.install("example/synth", "1.2.3");

      assert.equal(installed.installed, true);
      assert.deepEqual(requests, ["/preferred.zip"]);
      assert.equal(
        await readFile(
          path.join(
            pluginsDir,
            "VST3/example/synth/1.2.3/Preferred.vst3/Contents/x86_64-linux/Preferred.so",
          ),
          "utf8",
        ),
        "preferred archive",
      );
    });
  });
});

test("elevated install hands configured lock to the real admin child", async () => {
  await withTemporaryRoot(async (root) => {
    const configuredUrl = "https://example.test/configured.AppImage";
    const manager = await elevatedManagerFixture(root, {
      childUrl: "https://example.test/child-drift.AppImage",
      configuredLockUrl: configuredUrl,
      parentUrl: configuredUrl,
    });
    const previousLock = process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK;
    delete process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK;
    try {
      const launches = await withLocalAdminChild(async () => {
        await assert.rejects(
          manager.install("example/synth", "1.2.3"),
          /runCliAsAdmin: admin command reported error: artifact lock mismatch/,
        );
      });
      assert.equal(launches, 1);
    } finally {
      if (previousLock !== undefined)
        process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK = previousLock;
    }
  });
});

test("elevated admin child keeps environment lock precedence", async () => {
  await withTemporaryRoot(async (root) => {
    const environmentUrl = "https://example.test/environment.AppImage";
    const manager = await elevatedManagerFixture(root, {
      childUrl: "https://example.test/configured.AppImage",
      configuredLockUrl: "https://example.test/configured.AppImage",
      parentUrl: environmentUrl,
    });
    const environmentLock = await writeArtifactLock({
      artifacts: [artifact({ type: "installer", url: environmentUrl })],
      filename: "environment.lock.json",
      root,
    });
    const previousLock = process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK;
    process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK = environmentLock;
    try {
      const launches = await withLocalAdminChild(async () => {
        await assert.rejects(
          manager.install("example/synth", "1.2.3"),
          /runCliAsAdmin: admin command reported error: artifact lock mismatch/,
        );
      });
      assert.equal(launches, 1);
    } finally {
      if (previousLock === undefined)
        delete process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK;
      else
        process.env.SYNTH_SETTER_STUDIORACK_ARTIFACT_LOCK = previousLock;
    }
  });
});

test("admin patch forwards artifactLockPath through ManagerLocal config", async () => {
  await withTemporaryRoot(async (root) => {
    const before =
      "const manager = new ManagerLocal(args.type, { appDir: args.appDir });";
    const { admin, result } = await patchFixture(root, before);

    assert.equal(result.status, 0, result.stderr);
    assert.match(
      await readFile(admin, "utf8"),
      /artifactLockPath: args\.artifactLockPath/,
    );
  });
});

test("admin patch fails closed when its forwarding precondition is missing", async () => {
  await withTemporaryRoot(async (root) => {
    const incompatible = "const manager = createManager(args);";
    const originalHelper = helperSource();
    const originalManager = await readFile(managerSourceFixture, "utf8");
    const { admin, helper, manager, result } = await patchFixture(
      root,
      incompatible,
      originalHelper,
    );

    assert.notEqual(result.status, 0);
    assert.match(
      result.stderr,
      /expected admin artifact lock forwarding source was not found/,
    );
    assert.equal(await readFile(admin, "utf8"), incompatible);
    assert.equal(await readFile(helper, "utf8"), originalHelper);
    assert.equal(await readFile(manager, "utf8"), originalManager);
  });
});

test("patch fails closed when an applied source marker is ambiguous", async () => {
  await withTemporaryRoot(async (root) => {
    const adminSource =
      "const manager = new ManagerLocal(args.type, { appDir: args.appDir });";
    const ambiguousHelper = helperSource()
      .replace(
        "if (fileExists(path.join(f, 'Contents', 'Info.plist'))) {",
        "if (path.extname(f).toLowerCase() === '.vst3' || " +
          "fileExists(path.join(f, 'Contents', 'Info.plist'))) {",
      )
      .concat("path.extname(f).toLowerCase() === '.vst3';\n");
    const originalManager = await readFile(managerSourceFixture, "utf8");
    const { admin, helper, manager, result } = await patchFixture(
      root,
      adminSource,
      ambiguousHelper,
    );

    assert.notEqual(result.status, 0);
    assert.match(
      result.stderr,
      /ambiguous Linux VST3 bundle detection marker in .*file\.js: found 2 matches/,
    );
    assert.equal(await readFile(admin, "utf8"), adminSource);
    assert.equal(await readFile(helper, "utf8"), ambiguousHelper);
    assert.equal(await readFile(manager, "utf8"), originalManager);
  });
});
