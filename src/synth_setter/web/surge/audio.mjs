export function encodeStereoWav(channels, sampleRate) {
  if (
    channels.length !== 2 ||
    channels[0].length !== channels[1].length ||
    channels[0].length === 0
  )
    throw new Error("expected equal nonempty stereo channels");
  const frames = channels[0].length;
  const buffer = new ArrayBuffer(44 + frames * 8);
  const view = new DataView(buffer);
  const text = (offset, value) =>
    [...value].forEach((char, index) =>
      view.setUint8(offset + index, char.charCodeAt(0)),
    );
  text(0, "RIFF");
  view.setUint32(4, buffer.byteLength - 8, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 8, true);
  view.setUint16(32, 8, true);
  view.setUint16(34, 32, true);
  text(36, "data");
  view.setUint32(40, frames * 8, true);
  for (let frame = 0; frame < frames; frame++) {
    for (let channel = 0; channel < 2; channel++) {
      const value = channels[channel][frame];
      if (!Number.isFinite(value)) throw new Error("audio must be finite");
      view.setFloat32(44 + frame * 8 + channel * 4, value, true);
    }
  }
  return buffer;
}

export function audioMetrics(target, prediction) {
  let targetPower = 0,
    predictionPower = 0,
    errorPower = 0,
    peak = 0;
  const frames = target[0].length;
  if (
    target.length !== 2 ||
    prediction.length !== 2 ||
    !frames ||
    target[1].length !== frames ||
    prediction.some((channel) => channel.length !== frames)
  )
    throw new Error("audio shapes differ");
  for (let channel = 0; channel < 2; channel++) {
    for (let frame = 0; frame < frames; frame++) {
      const expected = target[channel][frame],
        actual = prediction[channel][frame];
      if (!Number.isFinite(expected) || !Number.isFinite(actual))
        throw new Error("audio must be finite");
      targetPower += expected * expected;
      predictionPower += actual * actual;
      errorPower += (expected - actual) ** 2;
      peak = Math.max(peak, Math.abs(actual));
    }
  }
  return {
    targetRms: Math.sqrt(targetPower / (2 * frames)),
    predictionRms: Math.sqrt(predictionPower / (2 * frames)),
    waveformRmse: Math.sqrt(errorPower / (2 * frames)),
    predictionPeak: peak,
  };
}

export async function decodeStereo(buffer, sampleRate, frames) {
  const context = new OfflineAudioContext(2, frames, sampleRate);
  const decoded = await context.decodeAudioData(buffer.slice(0));
  const source = context.createBufferSource();
  source.buffer = decoded;
  source.connect(context.destination);
  source.start();
  const rendered = await context.startRendering();
  const samples = new Float32Array(2 * frames);
  for (let channel = 0; channel < 2; channel++)
    samples.set(
      rendered
        .getChannelData(channel)
        .map((value) => Math.max(-1, Math.min(1, value))),
      channel * frames,
    );
  return samples;
}
