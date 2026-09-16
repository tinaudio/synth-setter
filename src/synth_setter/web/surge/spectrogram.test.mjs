import assert from "node:assert/strict";
import test from "node:test";
import { stereoSpectrogram } from "./spectrogram.mjs";

test("spectrogram locates a known tone at its calibrated full-scale level", () => {
  const tone = Float32Array.from({ length: 8192 }, (_, n) => Math.sin(2 * Math.PI * 64 * n / 2048));
  const frame = stereoSpectrogram([tone, tone])[4];
  assert.equal(frame.indexOf(Math.max(...frame)), 64);
  assert.ok(Math.abs(frame[64]) < 1e-5);
});

test("spectrogram shared scale retains twenty decibels of attenuation", () => {
  const tone = Float32Array.from({ length: 8192 }, (_, n) => 0.1 * Math.sin(2 * Math.PI * 64 * n / 2048));
  assert.ok(Math.abs(stereoSpectrogram([tone, tone])[4][64] + 20) < 1e-5);
});

test("spectrogram averages channel powers without cancelling opposite phases", () => {
  const left = Float32Array.from({ length: 8192 }, (_, n) => Math.sin(2 * Math.PI * 64 * n / 2048));
  const right = Float32Array.from(left, (value) => -value);
  assert.ok(Math.abs(stereoSpectrogram([left, right])[4][64]) < 1e-5);
});

test("silent spectrogram is finite at the display floor", () => {
  const silence = new Float32Array(2048);
  assert.ok(stereoSpectrogram([silence, silence]).every((frame) => frame.every((value) => value === -100)));
});
