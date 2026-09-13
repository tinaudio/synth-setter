import assert from "node:assert/strict";
import test from "node:test";
import { authorMusicSketch } from "./author.mjs";

test("authored note occupies its MIDI activation bin only inside the note window", () => {
  const sketch = authorMusicSketch({
    pitch: 60,
    start: 1,
    end: 3,
    loudness: -0.5,
    centroid: 0,
  });
  assert.equal(sketch.length, 386 * 32);
  assert.equal(sketch[(2 + 60 * 3) * 32 + 8], 1);
  assert.equal(sketch[(2 + 60 * 3) * 32 + 24], 0);
  assert.equal(sketch[(2 + 61 * 3) * 32 + 8], 0);
  assert.equal(sketch[8], -0.5);
  assert.equal(sketch[0], -1);
});

test("authored invalid note range is rejected rather than clipped", () => {
  assert.throws(
    () =>
      authorMusicSketch({
        pitch: 128,
        start: 0,
        end: 1,
        loudness: 0,
        centroid: 0,
      }),
    /pitch/,
  );
});
