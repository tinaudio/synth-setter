// Port of decode_model_output for the pyfdn_n8_mono_householder spec: model space [-1, 1]
// saturates onto encoded [0, 1], then each field maps linearly onto its native bounds.
export const HOUSEHOLDER_WIDTH = 27;
const ORDER = 8;
const DELAY_MIN = 400;
const DELAY_MAX = 1200;
const RT_MIN_SECONDS = 0.1;
const RT_MAX_SECONDS = 4.0;

function encoded(row, index) {
  return Math.min(Math.max((row[index] + 1) / 2, 0), 1);
}

function linear(row, index, min, max) {
  return min + encoded(row, index) * (max - min);
}

function slice(row, offset, length, min, max) {
  return Float64Array.from({ length }, (_, i) => linear(row, offset + i, min, max));
}

export function decodeHouseholderRow(row) {
  if (row.length !== HOUSEHOLDER_WIDTH) throw new Error(`model row must have ${HOUSEHOLDER_WIDTH} values`);
  if (!Array.from(row).every(Number.isFinite)) throw new Error("model row must contain only finite values");
  return {
    delays: Int32Array.from(slice(row, 0, ORDER, DELAY_MIN, DELAY_MAX), Math.round),
    inputMatrix: slice(row, ORDER, ORDER, -1, 1),
    outputMatrix: slice(row, 2 * ORDER, ORDER, -1, 1),
    directMatrix: linear(row, 3 * ORDER, -1, 1),
    rtDcSeconds: linear(row, 3 * ORDER + 1, RT_MIN_SECONDS, RT_MAX_SECONDS),
    rtNyquistSeconds: linear(row, 3 * ORDER + 2, RT_MIN_SECONDS, RT_MAX_SECONDS),
  };
}
