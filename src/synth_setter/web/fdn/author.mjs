// Author a pyfdn_reverb sketch from a few physical controls instead of a recording. Each row
// evaluates a closed form on the extractor's own log-time grid, so an authored sketch lands
// where an extracted one would for a single-slope decay.
import {
  EDC_FLOOR_DB,
  frameCenters,
  logTimeEdges,
  normalizeEchoDensity,
  poolFrameTrack,
  poolSampleTrack,
  SKETCH_CONTROLS,
  SKETCH_INTERVALS,
} from "./sketch.mjs";

export const BAND_CENTRES_HZ = [62.5, 125, 250, 500, 1000, 2000, 4000, 8000];
// RT60 bounds mirror the pyFDN spec's DC and Nyquist reverberation-time range.
export const RT60_RANGE_SECONDS = [0.1, 4.0];

export const AUTHOR_DEFAULTS = Object.freeze({
  rt60Seconds: [1.6, 1.5, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4],
  preDelayMs: 0,
  mixingTimeMs: 40,
  initialDensity: 0.5,
  flatnessStart: 0.3,
  flatnessEnd: 0.6,
});

function requireUnit(value, name) {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(`${name} must lie in [0, 1]`);
  return value;
}

function requireNonNegative(value, name) {
  if (!Number.isFinite(value) || value < 0) throw new Error(`${name} must be non-negative`);
  return value;
}

function validate(params) {
  const rt60 = params.rt60Seconds;
  if (!Array.isArray(rt60) || rt60.length !== BAND_CENTRES_HZ.length) {
    throw new Error(`rt60Seconds must hold ${BAND_CENTRES_HZ.length} band values`);
  }
  for (const value of rt60) {
    if (!Number.isFinite(value) || value < RT60_RANGE_SECONDS[0] || value > RT60_RANGE_SECONDS[1]) {
      throw new Error(`rt60Seconds must lie in [${RT60_RANGE_SECONDS.join(", ")}] seconds`);
    }
  }
  requireNonNegative(params.preDelayMs, "preDelayMs");
  requireNonNegative(params.mixingTimeMs, "mixingTimeMs");
  requireUnit(params.initialDensity, "initialDensity");
  requireUnit(params.flatnessStart, "flatnessStart");
  requireUnit(params.flatnessEnd, "flatnessEnd");
}

// Energy decays 60 dB per RT60 once the pre-delay has elapsed; clipped and mapped like the extractor.
function decayRow(rt60, preDelaySamples, frames, sampleRate, edges) {
  const track = new Float64Array(frames);
  for (let n = 0; n < frames; n++) {
    const elapsed = Math.max(0, n - preDelaySamples) / sampleRate;
    const db = Math.max(EDC_FLOOR_DB, (-60 * elapsed) / rt60);
    track[n] = 1 + db / 30;
  }
  return poolSampleTrack(track, edges);
}

// Density rises from the initial value to the diffuse reference (1.0) at the mixing time.
function densityRow(params, centers, edges, sampleRate) {
  const mixingSamples = (params.mixingTimeMs / 1000) * sampleRate;
  const perFrame = Float64Array.from(centers, (center) => {
    const progress = mixingSamples > 0 ? Math.min(1, center / mixingSamples) : 1;
    return params.initialDensity + (1 - params.initialDensity) * progress;
  });
  return poolFrameTrack(perFrame, centers, edges).map(normalizeEchoDensity);
}

// Flatness ramps across the log-time intervals themselves, so the endpoints land exactly.
function flatnessRow(params) {
  return Float64Array.from({ length: SKETCH_INTERVALS }, (_, k) => {
    const progress = k / (SKETCH_INTERVALS - 1);
    return 2 * (params.flatnessStart + (params.flatnessEnd - params.flatnessStart) * progress) - 1;
  });
}

export function authorReverbSketch(params, { sampleRate, frames }) {
  validate(params);
  const edges = logTimeEdges(frames);
  const centers = frameCenters(frames);
  const preDelaySamples = (params.preDelayMs / 1000) * sampleRate;
  const rows = [
    ...params.rt60Seconds.map((rt60) => decayRow(rt60, preDelaySamples, frames, sampleRate, edges)),
    densityRow(params, centers, edges, sampleRate),
    flatnessRow(params),
  ];
  const sketch = new Float32Array(SKETCH_CONTROLS * SKETCH_INTERVALS);
  rows.forEach((row, index) => sketch.set(row.map((value) => Math.min(Math.max(value, -1), 1)), index * SKETCH_INTERVALS));
  return sketch;
}
