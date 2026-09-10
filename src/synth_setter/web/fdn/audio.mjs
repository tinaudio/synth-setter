// Decode an uploaded file to the checkpoint's grid: mono, 44.1 kHz, exactly `frames` samples.
export async function decodeToContract(arrayBuffer, { sampleRate, frames }) {
  const decoder = new OfflineAudioContext(1, frames, sampleRate);
  const decoded = await decoder.decodeAudioData(arrayBuffer.slice(0));
  // Rendering through the offline context resamples and downmixes to the contract rate.
  const source = decoder.createBufferSource();
  source.buffer = decoded;
  source.connect(decoder.destination);
  source.start(0);
  const rendered = await decoder.startRendering();
  const samples = Float64Array.from(rendered.getChannelData(0), (value) => Math.max(-1, Math.min(1, value)));
  return { samples, sourceSampleRate: decoded.sampleRate, sourceChannels: decoded.numberOfChannels, sourceFrames: decoded.length };
}
