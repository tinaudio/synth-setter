import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import {
  cp,
  mkdtemp,
  mkdir,
  readFile,
  readdir,
  readlink,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execute = promisify(execFile);
const bundlePath = process.env.SURGE_WASM_BUNDLE;
if (!bundlePath)
  throw new Error("SURGE_WASM_BUNDLE must name a bundle produced by build.mjs");

const repository = fileURLToPath(new URL("../../../", import.meta.url));
const renderScript = fileURLToPath(new URL("./render.mjs", import.meta.url));
const presetPath = join(repository, "presets", "surge-base.fxp");
const python = process.env.PYTHON ?? join(repository, ".venv", "bin", "python");
const request = {
  parameters: [],
  note: 60,
  velocity: 100,
  noteStart: 0,
  noteEnd: 0.5,
  sampleRate: 44100,
  frames: 44100,
};

const baselineParameters = [
  { id: 301, name: "A Filter Configuration", value: 0.038 },
  { id: 306, name: "A Filter 1 Type", value: 0.0655 },
  { id: 308, name: "A Filter 1 Cutoff", value: 0.8 },
  { id: 274, name: "A Osc 1 Route", value: 0.1265 },
];

let temporaryDirectory;

before(async () => {
  temporaryDirectory = await mkdtemp(join(tmpdir(), "surge-wasm-render-test-"));
});

after(async () => {
  if (temporaryDirectory)
    await rm(temporaryDirectory, { recursive: true, force: true });
});

const runCli = async (requests, outputName) => {
  const requestsPath = join(temporaryDirectory, `${outputName}.json`);
  const outputPath = join(temporaryDirectory, outputName);
  await writeFile(requestsPath, JSON.stringify(requests));
  await execute(
    "node",
    [renderScript, bundlePath, presetPath, requestsPath, outputPath],
    {
      cwd: repository,
      timeout: 120_000,
    },
  );
  return outputPath;
};

const consumeWavs = async (paths) => {
  const program = `
import json
import numpy as np
import soundfile as sf
import sys
rows = []
for path in sys.argv[1:]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    rows.append({
        "sampleRate": sample_rate,
        "shape": list(audio.shape),
        "finite": bool(np.isfinite(audio).all()),
        "channelPeaks": np.abs(audio).max(axis=0).tolist(),
        "energy": float(np.square(audio).sum()),
        "spectralCentroid": float(
            np.average(
                np.fft.rfftfreq(len(audio), 1 / sample_rate),
                weights=np.abs(np.fft.rfft(audio, axis=0)).sum(axis=1),
            )
        ),
    })
print(json.dumps(rows))
`;
  const { stdout } = await execute(python, ["-c", program, ...paths], {
    cwd: repository,
  });
  return JSON.parse(stdout);
};

test("renderCli_baseline_and_closed_cutoff_publish_consumable_audio_and_report", async () => {
  const closedParameters = baselineParameters.map((parameter) =>
    parameter.id === 308 ? { ...parameter, value: 0 } : parameter,
  );
  const outputPath = await runCli(
    [
      { ...request, parameters: baselineParameters },
      { ...request, parameters: closedParameters },
    ],
    "successful-output",
  );

  assert.deepEqual((await readdir(outputPath)).sort(), [
    "report.json",
    "sample_00.wav",
    "sample_01.wav",
  ]);
  const audio = await consumeWavs([
    join(outputPath, "sample_00.wav"),
    join(outputPath, "sample_01.wav"),
  ]);
  assert.deepEqual(
    audio.map(({ sampleRate, shape, finite }) => ({
      sampleRate,
      shape,
      finite,
    })),
    [
      { sampleRate: 44100, shape: [44100, 2], finite: true },
      { sampleRate: 44100, shape: [44100, 2], finite: true },
    ],
  );
  assert.ok(audio[0].channelPeaks.every((peak) => peak > 1e-5));
  assert.ok(audio[1].channelPeaks.every((peak) => peak > 0));
  assert.ok(
    audio[1].spectralCentroid < audio[0].spectralCentroid * 0.5,
    `cutoff did not attenuate high frequencies: ${JSON.stringify(audio)}`,
  );

  const report = JSON.parse(
    await readFile(join(outputPath, "report.json"), "utf8"),
  );
  const presetSha256 = createHash("sha256")
    .update(await readFile(presetPath))
    .digest("hex");
  assert.equal(report.source, "surge-wasm");
  assert.equal(report.engineCommit, "dd68c74346c828ef25bd6504867936648161b7a7");
  assert.equal(report.rendererVersion, "Surge XT 1.4.HEAD.dd68c743");
  assert.equal(report.presetSha256, presetSha256);
  assert.equal(report.sampleCount, 2);
  assert.deepEqual(
    report.rows.map(
      ({ sample, filename, sampleRate, frames, parameters }) => ({
        sample,
        filename,
        sampleRate,
        frames,
        parameters,
      }),
    ),
    [
      {
        sample: 0,
        filename: "sample_00.wav",
        sampleRate: 44100,
        frames: 44100,
        parameters: baselineParameters,
      },
      {
        sample: 1,
        filename: "sample_01.wav",
        sampleRate: 44100,
        frames: 44100,
        parameters: closedParameters,
      },
    ],
  );
  assert.ok(
    report.rows.every(
      ({ renderSeconds }) =>
        Number.isFinite(renderSeconds) && renderSeconds > 0,
    ),
  );
});

test("renderCli_accepts_unextended_native_FX_parameter_identity", async () => {
  const output = await runCli([
    { ...request, parameters: [...baselineParameters, { id: 12, name: "FX A1 Time", value: 0.5 }] },
  ], "native-fx-label");
  const [audio] = await consumeWavs([join(output, "sample_00.wav")]);
  assert.equal(audio.finite, true);
  assert.ok(audio.channelPeaks.every((peak) => peak > 1e-5));
});

for (const name of ["FX A2 Time", "FX A1 Wrong"]) {
  test(`renderCli_rejects_incorrect_FX_identity_${name}`, async () => {
    await assert.rejects(
      runCli([{ ...request, parameters: [{ id: 12, name, value: 0.5 }] }], `wrong-${name.replaceAll(" ", "-")}`),
      /Surge parameter 12 is named/,
    );
  });
}

test("renderCli_invalid_later_request_leaves_no_published_output", async () => {
  const requestsPath = join(temporaryDirectory, "invalid.json");
  const outputPath = join(temporaryDirectory, "invalid-output");
  const invalid = {
    ...request,
    parameters: [{ id: 308, name: "A Filter 1 Cutoff", value: 2 }],
  };
  await writeFile(requestsPath, JSON.stringify([request, invalid]));

  await assert.rejects(
    execute(
      "node",
      [renderScript, bundlePath, presetPath, requestsPath, outputPath],
      {
        cwd: repository,
        timeout: 120_000,
      },
    ),
    /normalized to \[0, 1\]/,
  );
  await assert.rejects(readdir(outputPath), { code: "ENOENT" });
});

test("renderCli_rejects_tampered_loader_before_executing_it", async () => {
  const badBundle = join(temporaryDirectory, "tampered-bundle");
  await cp(bundlePath, badBundle, { recursive: true });
  await writeFile(join(badBundle, "surge-host.mjs"), 'throw new Error("EXECUTED_TAMPERED_FACTORY");');
  const requestsPath = join(temporaryDirectory, "tampered-request.json");
  await writeFile(requestsPath, JSON.stringify([request]));
  const output = join(temporaryDirectory, "tampered-output");
  await assert.rejects(
    execute("node", [renderScript, badBundle, presetPath, requestsPath, output], { cwd: repository }),
    /Surge artifact digest mismatch: surge-host.mjs/,
  );
  await assert.rejects(readdir(output), { code: "ENOENT" });
});

test("renderCli_uses_verified_snapshot_when_source_loader_is_replaced", async () => {
  const changingBundle = join(temporaryDirectory, "changing-bundle");
  await cp(bundlePath, changingBundle, { recursive: true });
  const loaderPath = join(changingBundle, "surge-host.mjs");
  const originalPath = join(temporaryDirectory, "original-loader.mjs");
  await writeFile(originalPath, await readFile(loaderPath));
  await rm(loaderPath);
  await execute("mkfifo", [loaderPath]);
  const requestsPath = join(temporaryDirectory, "changing-request.json");
  await writeFile(requestsPath, JSON.stringify([request]));
  const output = join(temporaryDirectory, "changing-output");
  const replaceAfterOpen = `
from pathlib import Path
import sys
path = Path(sys.argv[1])
original = Path(sys.argv[2]).read_bytes()
with path.open("wb") as stream:
    path.unlink()
    path.write_text('throw new Error("UNVERIFIED_FACTORY_EXECUTED");')
    stream.write(original)
`;
  await Promise.all([
    execute(python, ["-c", replaceAfterOpen, loaderPath, originalPath], { timeout: 30000 }),
    execute("node", [renderScript, changingBundle, presetPath, requestsPath, output], { cwd: repository, timeout: 30000 }),
  ]);
  assert.deepEqual((await readdir(output)).sort(), ["report.json", "sample_00.wav"]);
});

test("renderCli_rejects_fractional_sample_rate", async () => {
  const requestsPath = join(temporaryDirectory, "fractional-rate.json");
  await writeFile(requestsPath, JSON.stringify([{ ...request, sampleRate: 44100.5 }]));
  const output = join(temporaryDirectory, "fractional-output");
  await assert.rejects(
    execute("node", [renderScript, bundlePath, presetPath, requestsPath, output], { cwd: repository }),
    /sampleRate must be an integer/,
  );
});

test("renderCli_refuses_existing_dangling_output_symlink", async () => {
  const output = join(temporaryDirectory, "dangling-output");
  const target = join(temporaryDirectory, "missing-target");
  await symlink(target, output);
  const requestsPath = join(temporaryDirectory, "dangling-request.json");
  await writeFile(requestsPath, JSON.stringify([request]));
  await assert.rejects(
    execute("node", [renderScript, bundlePath, presetPath, requestsPath, output], { cwd: repository }),
    /output directory already exists/,
  );
  assert.equal(await readlink(output), target);
});

test("renderCli_empty_request_list_is_rejected", async () => {
  const outputPath = join(temporaryDirectory, "empty-output");

  await assert.rejects(
    runCli([], "empty-output"),
    /requests must be a non-empty array/,
  );
  await assert.rejects(readdir(outputPath), { code: "ENOENT" });
});

test("renderCli_existing_destination_is_refused", async () => {
  const outputPath = join(temporaryDirectory, "existing-output");
  await mkdir(outputPath);

  await assert.rejects(
    runCli([request], "existing-output"),
    /output directory already exists/,
  );
  assert.deepEqual(await readdir(outputPath), []);
});
