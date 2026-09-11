// Mirror of synth_setter.tools.browser_fdn_fixtures.synthetic_impulse_response: the golden
// fixtures carry only this integer recipe, so both languages rebuild the same float64 signal.
const LCG_MULTIPLIER = 1664525;
const LCG_INCREMENT = 1013904223;
const LCG_MODULUS = 2 ** 32;

export function syntheticImpulseResponse({ seed, samples, decay_seconds: decaySeconds }, sampleRate) {
  let state = seed % LCG_MODULUS;
  const response = new Float64Array(samples);
  for (let index = 0; index < samples; index++) {
    state = (Math.imul(LCG_MULTIPLIER, state) + LCG_INCREMENT) >>> 0;
    const uniform = state / LCG_MODULUS;
    response[index] = (2 * uniform - 1) * 0.5 * Math.exp(-(index / sampleRate) / decaySeconds);
  }
  response[0] = 1;
  return response;
}
