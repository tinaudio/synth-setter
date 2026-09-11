// Port of synth_setter.features.pyfdn_controls.extract_reverb_sketch (10 × 32 controls).
import octaveBands from "./fixtures/octave_bands.json" with { type: "json" };
import { assertFiniteSignal, energyDecayCurve, framePowerSpectra, hann, sosfilt } from "./dsp.mjs";

export const SKETCH_CONTROLS = 10;
export const SKETCH_INTERVALS = 32;
const ANALYSIS_WINDOW = 1024;
const ANALYSIS_HOP = 128;
export const EDC_FLOOR_DB = -60;
const LOG_HEAD_FRACTION = 0.005;
const LOG_RANGE_RATIO = 200;
// erfc(1/sqrt(2)): the Gaussian reference density of Abel & Huang.
const GAUSSIAN_REFERENCE = 0.31731050786291415;
const TINY = 2.2250738585072014e-308;

function peakNormalized(ir, sampleRate) {
  assertFiniteSignal(ir, "impulse response");
  if (sampleRate !== octaveBands.sample_rate) {
    throw new Error(`reverb sketch supports only ${octaveBands.sample_rate} Hz responses`);
  }
  if (ir.length < ANALYSIS_WINDOW) throw new Error("impulse response must contain at least 1024 samples");
  let peak = 0;
  for (const value of ir) peak = Math.max(peak, Math.abs(value));
  if (peak === 0) throw new Error("impulse response must have non-zero energy");
  return Float64Array.from(ir, (value) => value / peak);
}

// Sample edges s_k = round(f_k N) with f_0 = 0 and f_k = 0.005 · 200^((k-1)/31).
export function logTimeEdges(samples) {
  const edges = new Int32Array(SKETCH_INTERVALS + 1);
  for (let k = 1; k <= SKETCH_INTERVALS; k++) {
    const fraction = k === SKETCH_INTERVALS ? 1 : LOG_HEAD_FRACTION * LOG_RANGE_RATIO ** ((k - 1) / (SKETCH_INTERVALS - 1));
    edges[k] = Math.round(fraction * samples);
  }
  for (let k = 1; k <= SKETCH_INTERVALS; k++) {
    if (edges[k] <= edges[k - 1]) throw new Error("log-time sample edges must be strictly increasing");
  }
  return edges;
}

export function poolSampleTrack(track, edges) {
  const pooled = new Float64Array(SKETCH_INTERVALS);
  for (let k = 0; k < SKETCH_INTERVALS; k++) {
    let sum = 0;
    for (let n = edges[k]; n < edges[k + 1]; n++) sum += track[n];
    pooled[k] = sum / (edges[k + 1] - edges[k]);
  }
  return pooled;
}

export function frameCenters(samples) {
  const count = Math.floor((samples - ANALYSIS_WINDOW) / ANALYSIS_HOP) + 1;
  return Int32Array.from({ length: count }, (_, index) => index * ANALYSIS_HOP + ANALYSIS_WINDOW / 2);
}

export function poolFrameTrack(track, centers, edges) {
  const totals = new Float64Array(SKETCH_INTERVALS);
  const counts = new Int32Array(SKETCH_INTERVALS);
  for (let index = 0; index < centers.length; index++) {
    let interval = 0;
    while (interval + 1 < edges.length && edges[interval + 1] <= centers[index]) interval++;
    totals[interval] += track[index];
    counts[interval] += 1;
  }
  if (counts.some((count) => count === 0)) throw new Error("every log-time interval must receive frames");
  return totals.map((total, k) => total / counts[k]);
}

function octaveEdcTracks(response, edges) {
  return octaveBands.sketch.map((sos) => {
    const curve = energyDecayCurve(sosfilt(sos, response));
    const reference = curve[0];
    if (!(reference > 0)) throw new Error("every octave band must have non-zero energy");
    // Clip to [-60, 0] dB per sample before pooling, as the Python extractor does.
    const normalized = Float64Array.from(curve, (energy) => {
      const db = 10 * Math.log10(Math.max(energy / reference, TINY));
      return 1 + Math.min(Math.max(db, EDC_FLOOR_DB), 0) / 30;
    });
    return poolSampleTrack(normalized, edges);
  });
}

// pyFDN.echo_density: sparse Abel-Huang density every analysis hop, linearly interpolated.
function echoDensity(response) {
  const length = response.length;
  const half = ANALYSIS_WINDOW / 2;
  const window = hann(ANALYSIS_WINDOW, { periodic: false });
  const windowSum = window.reduce((sum, value) => sum + value, 0);
  const sparseIndex = [];
  for (let n = 0; n < length; n += ANALYSIS_HOP) sparseIndex.push(n);
  if (sparseIndex[sparseIndex.length - 1] !== length - 1) sparseIndex.push(length - 1);
  const sparse = sparseIndex.map((center) => {
    let start;
    let windowOffset;
    let count;
    if (center <= half) {
      start = 0;
      count = center + half;
      windowOffset = ANALYSIS_WINDOW - count;
    } else if (center <= length - half - 1) {
      start = center - half;
      count = ANALYSIS_WINDOW;
      windowOffset = 0;
    } else {
      start = center - half;
      count = length - start;
      windowOffset = 0;
    }
    let energy = 0;
    for (let i = 0; i < count; i++) {
      const value = response[start + i];
      energy += (window[windowOffset + i] / windowSum) * value * value;
    }
    const threshold = Math.sqrt(energy);
    let density = 0;
    for (let i = 0; i < count; i++) {
      if (Math.abs(response[start + i]) > threshold) density += window[windowOffset + i] / windowSum;
    }
    return density / GAUSSIAN_REFERENCE;
  });
  const dense = new Float64Array(length);
  let segment = 0;
  for (let n = 0; n < length; n++) {
    while (segment + 1 < sparseIndex.length - 1 && sparseIndex[segment + 1] <= n) segment++;
    const left = sparseIndex[segment];
    const right = sparseIndex[segment + 1];
    if (n >= right) dense[n] = sparse[segment + 1];
    else dense[n] = sparse[segment] + ((sparse[segment + 1] - sparse[segment]) * (n - left)) / (right - left);
  }
  return dense;
}

// Diffuse density 1 maps to 0 so the reference sits at the centre of model space.
export const normalizeEchoDensity = (density) => (2 * density) / (1 + density) - 1;

function echoDensityTrack(response, edges) {
  const centers = frameCenters(response.length);
  const dense = echoDensity(response);
  const perFrame = Float64Array.from(centers, (center) => dense[center]);
  return poolFrameTrack(perFrame, centers, edges).map(normalizeEchoDensity);
}

function spectralFlatnessTrack(response, edges) {
  const spectra = framePowerSpectra(response, {
    frameLength: ANALYSIS_WINDOW,
    hop: ANALYSIS_HOP,
    window: hann(ANALYSIS_WINDOW, { periodic: false }),
    center: false,
  });
  const flatness = Float64Array.from(spectra, (power) => {
    let arithmetic = 0;
    let logSum = 0;
    for (const value of power) {
      arithmetic += value;
      logSum += Math.log(Math.max(value, TINY));
    }
    arithmetic /= power.length;
    const geometric = Math.exp(logSum / power.length);
    const ratio = arithmetic > 0 ? geometric / arithmetic : 0;
    return Math.min(Math.max(ratio, 0), 1);
  });
  const centers = frameCenters(response.length);
  return poolFrameTrack(flatness, centers, edges).map((value) => 2 * Math.min(Math.max(value, 0), 1) - 1);
}

export function extractReverbSketch(ir, sampleRate) {
  const response = peakNormalized(ir, sampleRate);
  const edges = logTimeEdges(response.length);
  const rows = [...octaveEdcTracks(response, edges), echoDensityTrack(response, edges), spectralFlatnessTrack(response, edges)];
  const sketch = new Float32Array(SKETCH_CONTROLS * SKETCH_INTERVALS);
  rows.forEach((row, index) => {
    if (!row.every(Number.isFinite)) throw new Error("reverb sketch must contain only finite values");
    sketch.set(row.map((value) => Math.min(Math.max(value, -1), 1)), index * SKETCH_INTERVALS);
  });
  return sketch;
}
