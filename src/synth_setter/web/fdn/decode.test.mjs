import assert from "node:assert/strict";
import test from "node:test";
import golden from "./fixtures/golden.json" with { type: "json" };
import { decodeHouseholderRow, HOUSEHOLDER_WIDTH } from "./decode.mjs";

const decode = golden.decode;

test("golden row decodes to the Python renderer-native values", () => {
  const native = decodeHouseholderRow(Float32Array.from(decode.model_row));
  assert.deepEqual(Array.from(native.delays), decode.delays);
  assert.deepEqual(Array.from(native.inputMatrix), decode.input_matrix);
  assert.deepEqual(Array.from(native.outputMatrix), decode.output_matrix);
  assert.equal(native.directMatrix, decode.direct_matrix);
  assert.equal(native.rtDcSeconds, decode.rt_dc_seconds);
  assert.equal(native.rtNyquistSeconds, decode.rt_nyquist_seconds);
});

test("row width is the householder spec width", () => {
  assert.equal(HOUSEHOLDER_WIDTH, 27);
  assert.equal(decode.model_row.length, HOUSEHOLDER_WIDTH);
});

test("delays saturate at the native bounds for overshooting predictions", () => {
  const row = new Float32Array(HOUSEHOLDER_WIDTH);
  row[0] = -3;
  row[7] = 3;
  const native = decodeHouseholderRow(row);
  assert.equal(native.delays[0], 400);
  assert.equal(native.delays[7], 1200);
});

test("wrong width is rejected", () => {
  assert.throws(() => decodeHouseholderRow(new Float32Array(26)), /27/);
});

test("non-finite prediction is rejected", () => {
  const row = new Float32Array(HOUSEHOLDER_WIDTH);
  row[3] = NaN;
  assert.throws(() => decodeHouseholderRow(row), /finite/);
});
