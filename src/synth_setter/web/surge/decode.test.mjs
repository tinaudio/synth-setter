import assert from "node:assert/strict";
import test from "node:test";
import { decodeParameters } from "./decode.mjs";

const manifest = {
  encodedWidth: 7,
  parameters: [
    {
      name: "gain",
      kind: "continuous",
      target: "synth",
      offset: 0,
      width: 1,
      min: 0,
      max: 1,
      id: 12,
      nativeName: "Gain",
    },
    {
      name: "wave",
      kind: "onehot",
      target: "synth",
      offset: 1,
      width: 3,
      values: [0, 0.5, 1],
      id: 13,
      nativeName: "Wave",
    },
    {
      name: "pitch",
      kind: "integer",
      target: "note",
      offset: 4,
      width: 1,
      min: 48,
      max: 72,
    },
    {
      name: "note_start_and_end",
      kind: "window",
      target: "note",
      offset: 5,
      width: 2,
      max: 4,
    },
  ],
};

test("model coordinates decode to synth values and native MIDI timing", () => {
  const result = decodeParameters(
    Float32Array.of(0, -1, 1, -1, 0, -0.5, 0.5),
    manifest,
  );
  assert.deepEqual(result.synth, { gain: 0.5, wave: 0.5 });
  assert.deepEqual(result.note, { pitch: 60, note_start_and_end: [1, 3] });
});

test("onehot saturation ties choose the first coordinate like the native decoder", () => {
  const result = decodeParameters(
    Float32Array.of(5, 3, 8, -1, -5, -3, 4),
    manifest,
  );
  assert.deepEqual(result.synth, { gain: 1, wave: 0 });
  assert.deepEqual(result.note, { pitch: 48, note_start_and_end: [0, 4] });
});

test("nonfinite predictions cannot reach the synth", () => {
  assert.throws(
    () => decodeParameters([NaN, 0, 0, 0, 0, 0, 0], manifest),
    /finite/,
  );
});

test("truncated predictions cannot silently omit controls", () => {
  assert.throws(() => decodeParameters([0], manifest), /width/);
});
