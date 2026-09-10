// Ports of the pyFDN impulse-response metrics from synth_setter.evaluation.compute_audio_metrics.
import octaveBands from "./fixtures/octave_bands.json" with { type: "json" };
import { assertFiniteSignal, energyDecayCurve, framePowerSpectra, hann, sosfilt } from "./dsp.mjs";

// (window ms, hop ms, mel bins) resolutions of compute_mel_specs.
const MEL_PARAMS = [
  [10, 5, 32],
  [25, 10, 64],
  [100, 50, 128],
];
const DB_TOP = 80;
const DB_AMIN = 1e-10;
const EDC_WINDOW = 4096;
const EDC_HOP = 1024;
const EDC_FLOOR_DB = -45;
const EDC_BAND_EDGES_HZ = [44, 88, 177, 354, 707, 1414, 2828, 5657, 11314];
const RT60_FIT_DB = 30;
const C50_MS = 50;
const TINY = 2.2250738585072014e-308;

const hzToMel = (hz) => {
  const minLogHz = 1000;
  const linear = hz / (200 / 3);
  if (hz < minLogHz) return linear;
  return 15 + Math.log(hz / minLogHz) / (Math.log(6.4) / 27);
};
const melToHz = (mel) => {
  if (mel < 15) return mel * (200 / 3);
  return 1000 * Math.exp((Math.log(6.4) / 27) * (mel - 15));
};

// librosa.filters.mel with slaney norm; librosa stores the basis as float32.
function melFilterbank(sampleRate, frameLength, melBins) {
  const bins = Math.floor(frameLength / 2) + 1;
  const fftHz = Float64Array.from({ length: bins }, (_, k) => (k * sampleRate) / frameLength);
  const maxMel = hzToMel(sampleRate / 2);
  const melHz = Float64Array.from({ length: melBins + 2 }, (_, m) => melToHz((m * maxMel) / (melBins + 1)));
  const basis = [];
  for (let m = 0; m < melBins; m++) {
    const norm = 2 / (melHz[m + 2] - melHz[m]);
    basis.push(
      Float32Array.from(fftHz, (hz) => {
        const lower = (hz - melHz[m]) / (melHz[m + 1] - melHz[m]);
        const upper = (melHz[m + 2] - hz) / (melHz[m + 2] - melHz[m + 1]);
        return Math.max(0, Math.min(lower, upper)) * norm;
      }),
    );
  }
  return basis;
}

// librosa.power_to_db(ref=np.max, amin=1e-10, top_db=80) over the whole spectrogram.
function powerToDbRefMax(spectrogram) {
  let reference = 0;
  for (const column of spectrogram) for (const value of column) reference = Math.max(reference, value);
  const referenceDb = 10 * Math.log10(Math.max(DB_AMIN, reference));
  let peakDb = -Infinity;
  const db = spectrogram.map((column) =>
    column.map((value) => {
      const level = 10 * Math.log10(Math.max(DB_AMIN, value)) - referenceDb;
      peakDb = Math.max(peakDb, level);
      return level;
    }),
  );
  return db.map((column) => column.map((level) => Math.max(level, peakDb - DB_TOP)));
}

function logMelSpectrogram(signal, sampleRate, [windowMs, hopMs, melBins]) {
  const frameLength = Math.floor((windowMs * sampleRate) / 1000);
  const hop = Math.floor((hopMs * sampleRate) / 1000);
  const spectra = framePowerSpectra(signal, {
    frameLength,
    hop,
    window: hann(frameLength, { periodic: true }),
    center: true,
  });
  const basis = melFilterbank(sampleRate, frameLength, melBins);
  const mel = spectra.map((power) =>
    Float64Array.from(basis, (weights) => {
      let sum = 0;
      for (let k = 0; k < weights.length; k++) sum += weights[k] * power[k];
      return sum;
    }),
  );
  return powerToDbRefMax(mel);
}

export function mss(target, pred, sampleRate) {
  let distance = 0;
  for (const params of MEL_PARAMS) {
    const a = logMelSpectrogram(target, sampleRate, params);
    const b = logMelSpectrogram(pred, sampleRate, params);
    let sum = 0;
    let count = 0;
    for (let frame = 0; frame < a.length; frame++) {
      for (let bin = 0; bin < a[frame].length; bin++) {
        sum += Math.abs(a[frame][bin] - b[frame][bin]);
        count += 1;
      }
    }
    distance += sum / count;
  }
  return distance / MEL_PARAMS.length;
}

// pyFDN MatchEnergyDecay: per-band STFT Schroeder curves in dB, RMS over target frames above floor.
function bandEnergyDecayDb(signal, sampleRate) {
  const spectra = framePowerSpectra(signal, {
    frameLength: EDC_WINDOW,
    hop: EDC_HOP,
    window: hann(EDC_WINDOW, { periodic: true }),
    center: false,
  });
  const bins = spectra[0].length;
  const binHz = Float64Array.from({ length: bins }, (_, k) => (k * sampleRate) / EDC_WINDOW);
  const bands = [];
  for (let band = 0; band + 1 < EDC_BAND_EDGES_HZ.length; band++) {
    const [lo, hi] = [EDC_BAND_EDGES_HZ[band], EDC_BAND_EDGES_HZ[band + 1]];
    const power = Float64Array.from(spectra, (frame) => {
      let sum = 0;
      for (let k = 0; k < bins; k++) if (binHz[k] >= lo && binHz[k] < hi) sum += frame[k];
      return sum;
    });
    const curve = new Float64Array(power.length);
    let total = 0;
    for (let frame = power.length - 1; frame >= 0; frame--) {
      total += power[frame];
      curve[frame] = total;
    }
    bands.push(curve.map((energy) => 10 * Math.log10(energy / (curve[0] + TINY) + TINY)));
  }
  return bands;
}

export function octaveEdcRmseDb(target, pred, sampleRate) {
  const reference = bandEnergyDecayDb(target, sampleRate);
  const candidate = bandEnergyDecayDb(pred, sampleRate);
  let sum = 0;
  let count = 0;
  reference.forEach((curve, band) => {
    curve.forEach((level, frame) => {
      if (level > EDC_FLOOR_DB) {
        sum += (candidate[band][frame] - level) ** 2;
        count += 1;
      }
    });
  });
  if (count === 0) throw new Error("the target never rises above the energy-decay floor");
  return Math.sqrt(sum / count);
}

// pyFDN._rt_from_ir: linear fit from -5 dB over `decayDb`, extrapolated to 60 dB; 0 when unfit.
function reverberationTime(bandSignal, sampleRate, decayDb) {
  const energy = energyDecayCurve(bandSignal);
  let lastPositive = -1;
  let positiveCount = 0;
  for (let n = 0; n < energy.length; n++) {
    if (energy[n] > 0) {
      lastPositive = n;
      positiveCount += 1;
    }
  }
  if (positiveCount < 2) return 0;
  const db = new Float64Array(lastPositive + 1);
  const reference = 10 * Math.log10(energy[0]);
  let minimum = Infinity;
  for (let n = 0; n <= lastPositive; n++) {
    db[n] = 10 * Math.log10(energy[n]) - reference;
    minimum = Math.min(minimum, db[n]);
  }
  let fitDb = decayDb;
  if (-minimum - 5 < decayDb) fitDb = -minimum;
  const start = db.findIndex((level) => level < -5);
  if (start < 0) return 0;
  let end = db.findIndex((level) => level < db[start] - fitDb);
  if (end < 0) end = db.length;
  const size = end - start;
  if (size < 2) return 0;
  // Ordinary least-squares slope of (t, level - level[start]).
  let meanT = 0;
  let meanY = 0;
  for (let i = 0; i < size; i++) {
    meanT += i / sampleRate;
    meanY += db[start + i] - db[start];
  }
  meanT /= size;
  meanY /= size;
  let covariance = 0;
  let variance = 0;
  for (let i = 0; i < size; i++) {
    const dt = i / sampleRate - meanT;
    covariance += dt * (db[start + i] - db[start] - meanY);
    variance += dt * dt;
  }
  const slope = covariance / variance;
  return slope >= 0 ? 0 : -60 / slope;
}

function bandReverberationTimes(signal, sampleRate, bank, decayDb) {
  return bank.map((sos) => reverberationTime(sosfilt(sos, signal), sampleRate, decayDb));
}

export function octaveRt60LogRmse(target, pred, sampleRate) {
  const a = bandReverberationTimes(target, sampleRate, octaveBands.sketch, RT60_FIT_DB);
  const b = bandReverberationTimes(pred, sampleRate, octaveBands.sketch, RT60_FIT_DB);
  let sum = 0;
  let count = 0;
  a.forEach((rt, band) => {
    if (rt > 0 && b[band] > 0) {
      sum += (Math.log(b[band]) - Math.log(rt)) ** 2;
      count += 1;
    }
  });
  if (count === 0) throw new Error("no valid paired octave-band RT60 estimates");
  return Math.sqrt(sum / count);
}

export function t30Mape(target, pred, sampleRate) {
  const a = bandReverberationTimes(target, sampleRate, octaveBands.metrics, RT60_FIT_DB);
  const b = bandReverberationTimes(pred, sampleRate, octaveBands.metrics, RT60_FIT_DB);
  let sum = 0;
  let count = 0;
  a.forEach((t30, band) => {
    if (t30 > 0 && b[band] > 0) {
      sum += Math.abs(b[band] - t30) / t30;
      count += 1;
    }
  });
  if (count === 0) throw new Error("no valid paired octave-band T30 estimates");
  return (100 * sum) / count;
}

function clarity50(bandSignal, sampleRate) {
  const boundary = Math.floor((C50_MS * sampleRate) / 1000);
  let early = 0;
  let late = 0;
  bandSignal.forEach((value, n) => {
    if (n < boundary) early += value * value;
    else late += value * value;
  });
  return 10 * Math.log10(early / (late + 1e-32));
}

export function c50MaeDb(target, pred, sampleRate) {
  let sum = 0;
  for (const sos of octaveBands.metrics) {
    sum += Math.abs(clarity50(sosfilt(sos, pred), sampleRate) - clarity50(sosfilt(sos, target), sampleRate));
  }
  return sum / octaveBands.metrics.length;
}

export function evaluateImpulseResponses(target, pred, sampleRate) {
  assertFiniteSignal(target, "target");
  assertFiniteSignal(pred, "prediction");
  if (target.length !== pred.length) throw new Error("target and prediction must have the same length");
  if (sampleRate !== octaveBands.sample_rate) {
    throw new Error(`impulse-response metrics support only ${octaveBands.sample_rate} Hz`);
  }
  return {
    mss: mss(target, pred, sampleRate),
    octave_edc_rmse_db: octaveEdcRmseDb(target, pred, sampleRate),
    octave_rt60_log_rmse: octaveRt60LogRmse(target, pred, sampleRate),
    t30_mape: t30Mape(target, pred, sampleRate),
    c50_mae_db: c50MaeDb(target, pred, sampleRate),
  };
}
