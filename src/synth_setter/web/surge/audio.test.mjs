import assert from "node:assert/strict";
import test from "node:test";
import { encodeStereoWav, audioMetrics } from "./audio.mjs";

test("stereo WAV interleaves channels and retains the sample rate", () => {
  const wav = encodeStereoWav(
    [Float32Array.of(0.25, -0.5), Float32Array.of(-0.25, 0.5)],
    44100,
  );
  const view = new DataView(wav);
  assert.equal(view.getUint16(22, true), 2);
  assert.equal(view.getUint32(24, true), 44100);
  assert.equal(view.getUint32(40, true), 16);
  assert.equal(view.getFloat32(44, true), 0.25);
  assert.equal(view.getFloat32(48, true), -0.25);
  assert.equal(view.getFloat32(52, true), -0.5);
});

test("audio comparison reports known RMS error", () => {
  assert.equal(
    audioMetrics(
      [
        [0, 0],
        [0, 0],
      ],
      [
        [1, -1],
        [1, -1],
      ],
    ).waveformRmse,
    1,
  );
});
