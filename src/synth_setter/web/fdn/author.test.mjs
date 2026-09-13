import assert from "node:assert/strict";
import test from "node:test";
import { AUTHOR_DEFAULTS, authorReverbSketch, BAND_CENTRES_HZ } from "./author.mjs";
import { extractReverbSketch, SKETCH_INTERVALS } from "./sketch.mjs";
import { syntheticImpulseResponse } from "./synthetic.mjs";

const SAMPLE_RATE = 44100;
const FRAMES = 4 * SAMPLE_RATE;
const contract = { sampleRate: SAMPLE_RATE, frames: FRAMES };
const row = (sketch, index) => Array.from(sketch.subarray(index * SKETCH_INTERVALS, (index + 1) * SKETCH_INTERVALS));

test("authored sketch has the extractor's shape, dtype, and range", () => {
  const sketch = authorReverbSketch(AUTHOR_DEFAULTS, contract);
  assert.ok(sketch instanceof Float32Array);
  assert.equal(sketch.length, 10 * SKETCH_INTERVALS);
  assert.ok(sketch.every((value) => value >= -1 && value <= 1));
  assert.equal(BAND_CENTRES_HZ.length, 8);
});

test("authored decay rows match the extractor on a broadband exponential decay", () => {
  // Amplitude envelope exp(-t/tau) decays 60 dB of energy in tau * ln(1e6) / 2 seconds.
  const tau = 0.12;
  const rt60 = (tau * Math.log(1e6)) / 2;
  const response = syntheticImpulseResponse({ seed: 3, samples: FRAMES, decay_seconds: tau }, SAMPLE_RATE);
  const extracted = extractReverbSketch(response, SAMPLE_RATE);
  const authored = authorReverbSketch({ ...AUTHOR_DEFAULTS, rt60Seconds: new Array(8).fill(rt60) }, contract);
  // The lowest band tolerates more: its 62.5 Hz bandpass rings for tens of milliseconds.
  for (let band = 0; band < 8; band++) {
    const difference = row(authored, band).map((value, k) => Math.abs(value - row(extracted, band)[k]));
    const mean = difference.reduce((sum, value) => sum + value, 0) / difference.length;
    assert.ok(Math.max(...difference) < (band === 0 ? 0.25 : 0.1), `band ${band} max ${Math.max(...difference)}`);
    assert.ok(mean < 0.08, `band ${band} mean ${mean}`);
  }
});

test("longer RT60 keeps more energy at every interval and rows never rise", () => {
  const short = authorReverbSketch({ ...AUTHOR_DEFAULTS, rt60Seconds: new Array(8).fill(0.3) }, contract);
  const long = authorReverbSketch({ ...AUTHOR_DEFAULTS, rt60Seconds: new Array(8).fill(3.0) }, contract);
  for (let band = 0; band < 8; band++) {
    const shortRow = row(short, band);
    const longRow = row(long, band);
    shortRow.forEach((value, k) => assert.ok(longRow[k] >= value, `band ${band} interval ${k}`));
    for (let k = 1; k < SKETCH_INTERVALS; k++) assert.ok(longRow[k] <= longRow[k - 1] + 1e-9);
  }
});

test("pre-delay holds the decay at full energy until it elapses", () => {
  const delayed = authorReverbSketch({ ...AUTHOR_DEFAULTS, preDelayMs: 60 }, contract);
  const prompt = authorReverbSketch({ ...AUTHOR_DEFAULTS, preDelayMs: 0 }, contract);
  assert.ok(Math.abs(row(delayed, 0)[0] - 1) < 1e-6);
  assert.ok(row(prompt, 0)[5] < row(delayed, 0)[5]);
});

test("echo density row starts sparse and settles at the diffuse reference", () => {
  const sketch = authorReverbSketch({ ...AUTHOR_DEFAULTS, mixingTimeMs: 80, initialDensity: 0.2 }, contract);
  const density = row(sketch, 8);
  assert.ok(density[0] < -0.4);
  assert.ok(Math.abs(density[SKETCH_INTERVALS - 1]) < 1e-6);
  for (let k = 1; k < SKETCH_INTERVALS; k++) assert.ok(density[k] >= density[k - 1] - 1e-9);
});

test("spectral flatness row ramps linearly between its endpoints", () => {
  const sketch = authorReverbSketch({ ...AUTHOR_DEFAULTS, flatnessStart: 0.2, flatnessEnd: 0.8 }, contract);
  const flatness = row(sketch, 9);
  assert.ok(Math.abs(flatness[0] - (2 * 0.2 - 1)) < 1e-6);
  assert.ok(Math.abs(flatness[SKETCH_INTERVALS - 1] - (2 * 0.8 - 1)) < 1e-6);
  assert.ok(Math.abs(flatness[16] - 0.0129) < 0.01);
});

test("out-of-range controls are rejected", () => {
  assert.throws(() => authorReverbSketch({ ...AUTHOR_DEFAULTS, rt60Seconds: new Array(8).fill(0) }, contract), /rt60/);
  assert.throws(() => authorReverbSketch({ ...AUTHOR_DEFAULTS, rt60Seconds: new Array(7).fill(1) }, contract), /8/);
  assert.throws(() => authorReverbSketch({ ...AUTHOR_DEFAULTS, initialDensity: 1.5 }, contract), /initialDensity/);
  assert.throws(() => authorReverbSketch({ ...AUTHOR_DEFAULTS, flatnessEnd: -0.1 }, contract), /flatness/);
});
