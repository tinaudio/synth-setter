// Float32 keep mask in checkpoint group order: ONNX Runtime Web feeds float tensors,
// and the conditioning graph treats values above 0.5 as "keep".
export function sketchKeep(names, enabled) {
  const seen = new Set();
  for (const name of enabled) {
    if (!names.includes(name)) throw new Error(`unknown sketch group ${name}`);
    if (seen.has(name)) throw new Error(`duplicate sketch group ${name}`);
    seen.add(name);
  }
  return Float32Array.from(names, (name) => (seen.has(name) ? 1 : 0));
}

// Checkpoint order, not page order, so provenance matches the native selector.
export function appliedGroups(names, enabled) {
  const keep = sketchKeep(names, enabled);
  return names.filter((_, index) => keep[index] === 1);
}
