import assert from "node:assert/strict";
import test from "node:test";
import golden from "./fixtures/golden.json" with { type: "json" };
import { evaluateImpulseResponses } from "./metrics.mjs";
import { syntheticImpulseResponse } from "./synthetic.mjs";

const SAMPLE_RATE = golden.sample_rate;
const target = syntheticImpulseResponse(golden.target.recipe, SAMPLE_RATE);
const pred = syntheticImpulseResponse(golden.pred.recipe, SAMPLE_RATE);

function assertClose(actual, expected, relative) {
  assert.ok(
    Math.abs(actual - expected) <= relative * Math.abs(expected),
    `expected ${actual} within ${relative} of ${expected}`,
  );
}

test("multi-scale spectral distance matches Python", () => {
  const metrics = evaluateImpulseResponses(target, pred, SAMPLE_RATE);
  assertClose(metrics.mss, golden.metrics.mss, 1e-5);
});

test("octave energy-decay RMSE matches Python", () => {
  const metrics = evaluateImpulseResponses(target, pred, SAMPLE_RATE);
  assertClose(metrics.octave_edc_rmse_db, golden.metrics.octave_edc_rmse_db, 1e-6);
});

test("octave RT60 log RMSE matches Python", () => {
  const metrics = evaluateImpulseResponses(target, pred, SAMPLE_RATE);
  assertClose(metrics.octave_rt60_log_rmse, golden.metrics.octave_rt60_log_rmse, 1e-6);
});

test("T30 percentage error matches Python", () => {
  const metrics = evaluateImpulseResponses(target, pred, SAMPLE_RATE);
  assertClose(metrics.t30_mape, golden.metrics.t30_mape, 1e-6);
});

test("C50 absolute error matches Python", () => {
  const metrics = evaluateImpulseResponses(target, pred, SAMPLE_RATE);
  assertClose(metrics.c50_mae_db, golden.metrics.c50_mae_db, 1e-6);
});

test("identical responses score zero on every metric", () => {
  const metrics = evaluateImpulseResponses(target, target, SAMPLE_RATE);
  assert.deepEqual(metrics, {
    mss: 0,
    octave_edc_rmse_db: 0,
    octave_rt60_log_rmse: 0,
    t30_mape: 0,
    c50_mae_db: 0,
  });
});

test("length mismatch is rejected", () => {
  assert.throws(() => evaluateImpulseResponses(target, pred.subarray(0, 1000), SAMPLE_RATE), /length/);
});

test("non-finite prediction is rejected", () => {
  const broken = Float64Array.from(pred);
  broken[10] = Infinity;
  assert.throws(() => evaluateImpulseResponses(target, broken, SAMPLE_RATE), /finite/);
});

test("responses shorter than the energy-decay window are rejected", () => {
  const short = new Float64Array(1024).fill(0.1);
  assert.throws(() => evaluateImpulseResponses(short, short, SAMPLE_RATE), /4096/);
});
