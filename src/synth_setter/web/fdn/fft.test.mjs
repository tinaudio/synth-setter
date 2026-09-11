import assert from "node:assert/strict";
import test from "node:test";
import { powerSpectrum } from "./fft.mjs";

function naivePowerSpectrum(frame) {
  const bins = Math.floor(frame.length / 2) + 1;
  const power = new Float64Array(bins);
  for (let bin = 0; bin < bins; bin++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < frame.length; n++) {
      const angle = (-2 * Math.PI * bin * n) / frame.length;
      re += frame[n] * Math.cos(angle);
      im += frame[n] * Math.sin(angle);
    }
    power[bin] = re * re + im * im;
  }
  return power;
}

function ramp(length) {
  return Float64Array.from({ length }, (_, index) => Math.sin(index * 0.37) + index / length);
}

for (const length of [8, 441, 1024, 1102]) {
  test(`power spectrum of length ${length} matches a direct DFT`, () => {
    const frame = ramp(length);
    const actual = powerSpectrum(frame);
    const expected = naivePowerSpectrum(frame);
    assert.equal(actual.length, Math.floor(length / 2) + 1);
    const scale = Math.max(...expected);
    for (let bin = 0; bin < expected.length; bin++) {
      assert.ok(Math.abs(actual[bin] - expected[bin]) < 1e-9 * scale, `bin ${bin}`);
    }
  });
}

test("impulse has a flat unit power spectrum", () => {
  const frame = new Float64Array(16);
  frame[0] = 1;
  assert.deepEqual(Array.from(powerSpectrum(frame)), new Array(9).fill(1));
});
