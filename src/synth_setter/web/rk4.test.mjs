import assert from "node:assert/strict";
import test from "node:test";
import { integrateRK4 } from "./rk4.mjs";

test("one RK4 step integrates an exponential field", async () => {
  const result = await integrateRK4({
    field: async (x) => x,
    noise: new Float32Array([1]),
    steps: 1,
  });
  assert.ok(Math.abs(result[0] - 2.7083333333) < 1e-6);
});

test("multiple RK4 steps advance a time-dependent field to the endpoint", async () => {
  const result = await integrateRK4({
    field: async (_x, t) => new Float32Array([2 * t]),
    noise: new Float32Array([1]),
    steps: 4,
  });
  assert.ok(Math.abs(result[0] - 2) < 1e-6);
});

test("20000 RK4 steps integrate a constant field to its endpoint", async () => {
  const result = await integrateRK4({
    field: async () => new Float32Array([1]),
    noise: new Float32Array([0]),
    steps: 20000,
  });
  assert.ok(Math.abs(result[0] - 1) < 2e-4);
});

test("steps beyond 20000 are rejected", async () => {
  await assert.rejects(integrateRK4({field: async (x) => x, noise: new Float32Array([1]), steps: 20001}), /steps/);
});

test("zero integration steps are rejected", async () => {
  await assert.rejects(integrateRK4({field: async (x) => x, noise: new Float32Array([1]), steps: 0}), /steps/);
});

test("nonfinite field output cannot become a prediction", async () => {
  await assert.rejects(integrateRK4({field: async () => new Float32Array([NaN]), noise: new Float32Array([1]), steps: 1}), /finite/);
});
