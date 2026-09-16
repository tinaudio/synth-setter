import assert from "node:assert/strict";
import test from "node:test";
import { branchWeights } from "./guidance.mjs";

test("both mode reproduces the three-branch content/sketch guidance", () => {
  assert.deepEqual(Array.from(branchWeights("both", 2, 3)), [-2, 1, 0, 2]);
});

test("mel-only mode never weights a sketch-conditioned branch", () => {
  assert.deepEqual(Array.from(branchWeights("mel_only", 2, 3)), [-1, 0, 2, 0]);
});

test("sketch-only mode never weights a content-conditioned branch", () => {
  assert.deepEqual(Array.from(branchWeights("sketch_only", 2, 3)), [-2, 3, 0, 0]);
});

test("unconditional mode ignores both strengths", () => {
  assert.deepEqual(Array.from(branchWeights("unconditional", 2, 3)), [1, 0, 0, 0]);
});

test("unknown modes are rejected", () => {
  assert.throws(() => branchWeights("content", 2, 3), /mode/);
});
