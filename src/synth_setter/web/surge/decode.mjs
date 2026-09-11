export function decodeParameters(row, manifest) {
  if (row.length !== manifest.encodedWidth)
    throw new Error("prediction width differs from model contract");
  if (!Array.from(row).every(Number.isFinite))
    throw new Error("prediction must be finite");
  const encoded = Array.from(row, (value) =>
    Math.max(0, Math.min(1, (value + 1) / 2)),
  );
  const result = { synth: {}, note: {}, parameters: [] };
  let offset = 0;
  for (const parameter of manifest.parameters) {
    const { name, kind, target, width } = parameter;
    if (
      parameter.offset !== offset ||
      !Number.isInteger(width) ||
      width < 1 ||
      offset + width > row.length
    ) {
      throw new Error("invalid parameter span");
    }
    const values = encoded.slice(offset, offset + width);
    let value;
    if (kind === "onehot" && parameter.values.length === width) {
      value = parameter.values[values.indexOf(Math.max(...values))];
    } else if (kind === "continuous" && width === 1) {
      value = parameter.min + values[0] * (parameter.max - parameter.min);
    } else if (kind === "integer" && width === 1) {
      value =
        parameter.min +
        Math.floor(values[0] * (parameter.max - parameter.min) + 0.5);
    } else if (kind === "window" && width === 2) {
      value = values.map((entry) => entry * parameter.max);
    } else {
      throw new Error(`unsupported parameter encoding: ${name}`);
    }
    if (target !== "synth" && target !== "note")
      throw new Error("invalid parameter target");
    result[target][name] = value;
    if (target === "synth")
      result.parameters.push({
        id: parameter.id,
        name: parameter.nativeName,
        value,
      });
    offset += width;
  }
  if (offset !== row.length)
    throw new Error("parameter spans do not cover prediction width");
  return result;
}
