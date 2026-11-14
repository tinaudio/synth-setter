import assert from "node:assert/strict";
import test from "node:test";
import { sketchKeep } from "./sketch_groups.mjs";

const NAMES = ["loudness", "centroid", "pitch"];

test("keeping every group yields an all-ones float mask in checkpoint order", () => {
  assert.deepEqual(Array.from(sketchKeep(NAMES, ["pitch", "loudness", "centroid"])), [1, 1, 1]);
});

test("a subset zeroes only the groups the page left unchecked", () => {
  assert.deepEqual(Array.from(sketchKeep(NAMES, ["pitch"])), [0, 0, 1]);
});

test("an empty selection keeps only the positional encoding", () => {
  assert.deepEqual(Array.from(sketchKeep(NAMES, [])), [0, 0, 0]);
});

test("unknown group names are rejected instead of ignored", () => {
  assert.throws(() => sketchKeep(NAMES, ["tempo"]), /tempo/);
});

test("duplicate group names are rejected", () => {
  assert.throws(() => sketchKeep(NAMES, ["pitch", "pitch"]), /duplicate/);
});
