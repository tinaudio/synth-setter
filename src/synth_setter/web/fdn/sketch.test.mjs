import assert from "node:assert/strict";
import test from "node:test";
import golden from "./fixtures/golden.json" with { type: "json" };
import { extractReverbSketch } from "./sketch.mjs";
import { syntheticImpulseResponse } from "./synthetic.mjs";

const SAMPLE_RATE = golden.sample_rate;

function maxAbsDifference(actual, expected) {
  return Math.max(...expected.map((value, index) => Math.abs(actual[index] - value)));
}

test("sketch of the golden target matches the Python extractor", () => {
  const response = syntheticImpulseResponse(golden.target.recipe, SAMPLE_RATE);
  const sketch = extractReverbSketch(response, SAMPLE_RATE);
  assert.equal(sketch.length, 10 * 32);
  assert.ok(maxAbsDifference(sketch, golden.target.sketch.flat()) < 1e-6);
});

test("sketch of the golden prediction matches the Python extractor", () => {
  const response = syntheticImpulseResponse(golden.pred.recipe, SAMPLE_RATE);
  const sketch = extractReverbSketch(response, SAMPLE_RATE);
  assert.ok(maxAbsDifference(sketch, golden.pred.sketch.flat()) < 1e-6);
});

test("sketch values are float32 and clipped to [-1, 1]", () => {
  const response = syntheticImpulseResponse(golden.target.recipe, SAMPLE_RATE);
  const sketch = extractReverbSketch(response, SAMPLE_RATE);
  assert.ok(sketch instanceof Float32Array);
  assert.ok(sketch.every((value) => value >= -1 && value <= 1));
});

test("silent response is rejected", () => {
  assert.throws(() => extractReverbSketch(new Float64Array(176400), SAMPLE_RATE), /non-zero/);
});

test("response shorter than the analysis window is rejected", () => {
  assert.throws(() => extractReverbSketch(new Float64Array(512).fill(0.1), SAMPLE_RATE), /1024/);
});

test("unsupported sample rate is rejected", () => {
  const response = syntheticImpulseResponse(golden.target.recipe, SAMPLE_RATE);
  assert.throws(() => extractReverbSketch(response, 48000), /44100/);
});
