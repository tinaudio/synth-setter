// Canonical parameter addresses follow ParamSpec.native_names(): arrays expand in C order.
export function canonicalPatch(native) {
  const patch = {};
  native.delays.forEach((value, index) => (patch[`delays.${index}`] = value));
  native.inputMatrix.forEach((value, index) => (patch[`input_matrix.${index}.0`] = value));
  native.outputMatrix.forEach((value, index) => (patch[`output_matrix.0.${index}`] = value));
  patch["direct_matrix.0.0"] = native.directMatrix;
  patch["post_delay.rt_dc_seconds"] = native.rtDcSeconds;
  patch["post_delay.rt_nyquist_seconds"] = native.rtNyquistSeconds;
  return patch;
}
