import assert from "node:assert/strict";
import test from "node:test";
import { canonicalPatch } from "./patch.mjs";
import { gaussianNoise } from "./noise.mjs";
import { encodeWav } from "./wav.mjs";

const native = {
  delays: Int32Array.from([400, 500, 600, 700, 800, 900, 1000, 1100]),
  inputMatrix: Float64Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
  outputMatrix: Float64Array.from([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]),
  directMatrix: 0.25,
  rtDcSeconds: 1.5,
  rtNyquistSeconds: 0.75,
};

test("canonical patch names every householder coordinate the way the param spec does", () => {
  const patch = canonicalPatch(native);
  assert.equal(Object.keys(patch).length, 27);
  assert.equal(patch["delays.3"], 700);
  assert.equal(patch["input_matrix.2.0"], 0.3);
  assert.equal(patch["output_matrix.0.7"], -0.8);
  assert.equal(patch["direct_matrix.0.0"], 0.25);
  assert.equal(patch["post_delay.rt_dc_seconds"], 1.5);
  assert.equal(patch["post_delay.rt_nyquist_seconds"], 0.75);
});

test("seeded gaussian noise is deterministic and roughly standard normal", () => {
  const first = gaussianNoise(17, 4096);
  const second = gaussianNoise(17, 4096);
  assert.deepEqual(first, second);
  const mean = first.reduce((sum, value) => sum + value, 0) / first.length;
  const variance = first.reduce((sum, value) => sum + (value - mean) ** 2, 0) / first.length;
  assert.ok(Math.abs(mean) < 0.1);
  assert.ok(Math.abs(variance - 1) < 0.1);
  assert.notDeepEqual(gaussianNoise(18, 4096), first);
});

test("WAV encoding writes a 16-bit mono RIFF header with the sample count", () => {
  const bytes = encodeWav(Float64Array.from([0, 0.5, -0.5, 1]), 44100);
  const view = new DataView(bytes.buffer);
  assert.equal(String.fromCharCode(...bytes.subarray(0, 4)), "RIFF");
  assert.equal(String.fromCharCode(...bytes.subarray(8, 12)), "WAVE");
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 44100);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 8);
  assert.equal(view.getInt16(46, true), 16384);
  assert.equal(view.getInt16(50, true), 32767);
});
