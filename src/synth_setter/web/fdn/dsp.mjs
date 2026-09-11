import { powerSpectrum } from "./fft.mjs";

// scipy.signal.sosfilt: cascaded direct-form II transposed biquads from zero state.
export function sosfilt(sos, input) {
  let signal = Float64Array.from(input);
  for (const [b0, b1, b2, a0, a1, a2] of sos) {
    const output = new Float64Array(signal.length);
    let z1 = 0;
    let z2 = 0;
    for (let n = 0; n < signal.length; n++) {
      const x = signal[n];
      const y = (b0 * x + z1) / a0;
      z1 = b1 * x - a1 * y + z2;
      z2 = b2 * x - a2 * y;
      output[n] = y;
    }
    signal = output;
  }
  return signal;
}

// Schroeder backward integration: EDC[t] = Σ_{τ≥t} x[τ]².
export function energyDecayCurve(signal) {
  const curve = new Float64Array(signal.length);
  let total = 0;
  for (let n = signal.length - 1; n >= 0; n--) {
    total += signal[n] * signal[n];
    curve[n] = total;
  }
  return curve;
}

function cosineWindow(length, alpha, periodic) {
  const denominator = periodic ? length : length - 1;
  return Float64Array.from({ length }, (_, n) => alpha - (1 - alpha) * Math.cos((2 * Math.PI * n) / denominator));
}

// numpy.hanning is symmetric; scipy.signal.get_window(fftbins=True) windows are periodic.
export const hann = (length, { periodic }) => cosineWindow(length, 0.5, periodic);
export const hamming = (length, { periodic }) => cosineWindow(length, 0.54, periodic);

// Power spectrogram frames as librosa/torch produce them; `center` pads n_fft/2 zeros each side.
export function framePowerSpectra(signal, { frameLength, hop, window, center }) {
  const padded = center ? new Float64Array(signal.length + frameLength) : signal;
  if (center) padded.set(signal, Math.floor(frameLength / 2));
  const frameCount = Math.floor((padded.length - frameLength) / hop) + 1;
  const frames = [];
  const frame = new Float64Array(frameLength);
  for (let index = 0; index < frameCount; index++) {
    const start = index * hop;
    for (let n = 0; n < frameLength; n++) frame[n] = padded[start + n] * window[n];
    frames.push(powerSpectrum(frame));
  }
  return frames;
}

export function assertFiniteSignal(signal, name) {
  if (!(signal instanceof Float64Array) || signal.length === 0) {
    throw new Error(`${name} must be a non-empty Float64Array`);
  }
  if (!signal.every(Number.isFinite)) throw new Error(`${name} must contain only finite values`);
}
