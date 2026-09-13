function finiteVector(value, width) {
  if (!(value instanceof Float32Array) || value.length !== width || !value.every(Number.isFinite)) {
    throw new Error("Flow state must be a finite float32 vector of the expected width");
  }
  return value;
}

function advance(x, velocity, dt) {
  finiteVector(velocity, x.length);
  return Float32Array.from(x, (value, i) => value + dt * velocity[i]);
}

export async function integrateRK4({ field, noise, steps, onStep = () => {} }) {
  if (!Number.isInteger(steps) || steps < 1 || steps > 20000) {
    throw new Error("steps must be an integer between 1 and 20000");
  }
  if (!noise.length) throw new Error("Flow noise cannot be empty");
  let x = finiteVector(noise.slice(), noise.length);
  const dt = Math.fround(1 / steps);
  let t = 0;
  for (let step = 0; step < steps; step++) {
    const k1 = finiteVector(await field(x, t), x.length);
    const k2 = finiteVector(await field(advance(x, k1, dt / 2), Math.fround(t + dt / 2)), x.length);
    const k3 = finiteVector(await field(advance(x, k2, dt / 2), Math.fround(t + dt / 2)), x.length);
    const k4 = finiteVector(await field(advance(x, k3, dt), Math.fround(t + dt)), x.length);
    x = Float32Array.from(x, (value, i) => value + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    finiteVector(x, noise.length);
    t = Math.fround(t + dt);
    onStep(step + 1, steps);
  }
  return x;
}
