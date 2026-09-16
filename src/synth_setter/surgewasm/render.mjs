import { createHash } from "node:crypto";
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:http";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { encodeStereoWav } from "../web/surge/audio.mjs";
import { renderSurge } from "./runtime.mjs";

const ARTIFACT_CONTENT_TYPES = new Map([
  ["manifest.json", "application/json"],
  ["surge-host.mjs", "text/javascript"],
  ["surge-host.wasm", "application/wasm"],
  ["surge-xt.clap.wasm", "application/wasm"],
]);
const NANOSECONDS_PER_SECOND = 1_000_000_000;

const requireAbsent = async (path) => {
  try {
    await lstat(path);
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`output directory already exists: ${path}`);
};

const startBundleServer = async (bundleDirectory) => {
  const server = createServer(async (request, response) => {
    const pathname = new URL(request.url, "http://localhost").pathname;
    const filename = pathname.slice(1);
    const contentType = ARTIFACT_CONTENT_TYPES.get(filename);
    if (
      request.method !== "GET" ||
      !contentType ||
      pathname !== `/${filename}`
    ) {
      response.writeHead(404).end();
      return;
    }
    try {
      const bytes = await readFile(join(bundleDirectory, filename));
      response.setHeader("Content-Type", contentType);
      response.end(bytes);
    } catch (error) {
      response.destroy(error);
    }
  });
  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolveListen();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string")
    throw new Error("temporary HTTP server has no TCP address");
  return { root: new URL(`http://127.0.0.1:${address.port}/`), server };
};

const closeServer = (server) =>
  new Promise((resolveClose, reject) => {
    server.close((error) => (error ? reject(error) : resolveClose()));
  });

const renderRequests = async ({
  requests,
  root,
  loader,
  preset,
  stagingDirectory,
}) => {
  const samples = [];
  let engineCommit;
  let rendererVersion;
  for (const [sample, request] of requests.entries()) {
    const started = process.hrtime.bigint();
    const audio = await renderSurge({ ...request, root, loader, preset });
    const renderSeconds =
      Number(process.hrtime.bigint() - started) / NANOSECONDS_PER_SECOND;
    engineCommit ??= audio.engineCommit;
    rendererVersion ??= audio.version;
    if (
      audio.engineCommit !== engineCommit ||
      audio.version !== rendererVersion
    ) {
      throw new Error(
        "Surge renderer provenance changed within one request batch",
      );
    }

    const filename = `sample_${String(sample).padStart(2, "0")}.wav`;
    await writeFile(
      join(stagingDirectory, filename),
      new Uint8Array(encodeStereoWav([audio.left, audio.right], request.sampleRate)),
    );
    samples.push({
      sample,
      filename,
      sampleRate: request.sampleRate,
      frames: request.frames,
      renderSeconds,
      parameters: request.parameters,
    });
  }
  return { engineCommit, rendererVersion, samples };
};

const main = async (args) => {
  if (args.length !== 4) {
    throw new Error(
      "usage: node render.mjs BUNDLE_DIRECTORY PRESET_FXP REQUESTS_JSON OUTPUT_DIRECTORY",
    );
  }
  const [bundleArgument, presetPath, requestsPath, outputArgument] = args;
  const bundleDirectory = resolve(bundleArgument);
  const outputDirectory = resolve(outputArgument);
  await requireAbsent(outputDirectory);

  const requests = JSON.parse(await readFile(requestsPath, "utf8"));
  if (!Array.isArray(requests) || requests.length === 0) {
    throw new Error("requests must be a non-empty array");
  }
  const preset = new Uint8Array(await readFile(presetPath));
  const presetSha256 = createHash("sha256").update(preset).digest("hex");
  const outputParent = dirname(outputDirectory);
  await mkdir(outputParent, { recursive: true });
  const stagingDirectory = await mkdtemp(
    join(outputParent, `.${basename(outputDirectory)}-staging-`),
  );
  let published = false;
  try {
    // Keep integrity checks and imports bound to one snapshot during source rebuilds.
    const snapshotDirectory = join(stagingDirectory, ".engine");
    await mkdir(snapshotDirectory);
    for (const filename of ARTIFACT_CONTENT_TYPES.keys()) {
      await writeFile(join(snapshotDirectory, filename), await readFile(join(bundleDirectory, filename)));
    }
    const loader = async (options) => {
      const factory = (
        await import(pathToFileURL(join(snapshotDirectory, "surge-host.mjs")).href)
      ).default;
      return factory(options);
    };
    const { root, server } = await startBundleServer(snapshotDirectory);
    let rendered;
    try {
      rendered = await renderRequests({
        requests,
        root,
        loader,
        preset,
        stagingDirectory,
      });
    } finally {
      await closeServer(server);
    }
    const report = {
      source: "surge-wasm",
      engineCommit: rendered.engineCommit,
      rendererVersion: rendered.rendererVersion,
      presetSha256,
      sampleCount: rendered.samples.length,
      rows: rendered.samples,
    };
    await writeFile(
      join(stagingDirectory, "report.json"),
      `${JSON.stringify(report, null, 2)}\n`,
    );
    await rm(snapshotDirectory, { recursive: true });
    await requireAbsent(outputDirectory);
    await rename(stagingDirectory, outputDirectory);
    published = true;
  } finally {
    if (!published)
      await rm(stagingDirectory, { recursive: true, force: true });
  }
};

await main(process.argv.slice(2));
