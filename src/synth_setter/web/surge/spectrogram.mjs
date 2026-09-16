import { framePowerSpectra, hann } from "../fdn/dsp.mjs";

const FFT_LENGTH = 2048;
const HOP = 512;
const FLOOR_DB = -100;

export function stereoSpectrogram(channels) {
  if (channels.length !== 2 || !channels[0].length || channels[0].length !== channels[1].length)
    throw new Error("spectrogram requires equal non-empty stereo channels");
  if (!channels.every((channel) => channel.every(Number.isFinite)))
    throw new Error("spectrogram requires finite audio");
  const window = hann(FFT_LENGTH, { periodic: true });
  const options = { frameLength: FFT_LENGTH, hop: HOP, window, center: true };
  const left = framePowerSpectra(channels[0], options);
  const right = framePowerSpectra(channels[1], options);
  const windowSum = window.reduce((sum, value) => sum + value, 0);
  return left.map((frame, time) => Float32Array.from(frame, (power, bin) => {
    // Interior bins include their negative-frequency partner; DC and Nyquist do not.
    const scale = bin === 0 || bin === FFT_LENGTH / 2 ? 1 : 4;
    const normalized = (power + right[time][bin]) * 0.5 * scale / windowSum ** 2;
    return Math.max(FLOOR_DB, Math.min(0, 10 * Math.log10(Math.max(1e-10, normalized))));
  }));
}

export function spectrogramCanvas(channels, title, sampleRate) {
  const frames = stereoSpectrogram(channels);
  const canvas = document.createElement("canvas");
  canvas.width = frames.length;
  canvas.height = 256;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", `${title} spectrogram`);
  const context = canvas.getContext("2d");
  const image = context.createImageData(canvas.width, canvas.height);
  for (let time = 0; time < canvas.width; time++) {
    for (let bin = 0; bin < canvas.height; bin++) {
      const frequency = 20 * (sampleRate / 40) ** (bin / (canvas.height - 1));
      const sourceBin = Math.min(FFT_LENGTH / 2, Math.round(frequency * FFT_LENGTH / sampleRate));
      const level = (frames[time][sourceBin] - FLOOR_DB) / -FLOOR_DB;
      const offset = ((canvas.height - 1 - bin) * canvas.width + time) * 4;
      image.data[offset] = Math.round(255 * level);
      image.data[offset + 1] = Math.round(255 * level ** 2);
      image.data[offset + 2] = Math.round(100 * level);
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  const figure = document.createElement("figure");
  const caption = document.createElement("figcaption");
  caption.textContent = `0–${(channels[0].length / sampleRate).toFixed(2)} s →; 20 Hz–${(sampleRate / 2000).toFixed(2)} kHz ↑ (logarithmic). Shared −100 to 0 dBFS scale; stereo mean power, Hann 2048 / hop 512.`;
  figure.append(canvas, caption);
  return figure;
}
