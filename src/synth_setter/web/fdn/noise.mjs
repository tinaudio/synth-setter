// Seeded standard-normal draws (mulberry32 + Box-Muller) so a browser run is reproducible and
// its exact noise can be replayed through the PyTorch sampler for parity checks.
function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianNoise(seed, length) {
  if (!Number.isInteger(seed) || seed < 0) throw new Error("seed must be a non-negative integer");
  const uniform = mulberry32(seed);
  const noise = new Float32Array(length);
  for (let index = 0; index < length; index += 2) {
    const u1 = Math.max(uniform(), Number.EPSILON);
    const u2 = uniform();
    const radius = Math.sqrt(-2 * Math.log(u1));
    noise[index] = radius * Math.cos(2 * Math.PI * u2);
    if (index + 1 < length) noise[index + 1] = radius * Math.sin(2 * Math.PI * u2);
  }
  return noise;
}
