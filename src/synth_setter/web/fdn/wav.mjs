// 16-bit PCM mono RIFF/WAVE for playback and download of rendered responses.
export function encodeWav(samples, sampleRate) {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(bytes.buffer);
  const ascii = (offset, text) => [...text].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + index * 2, Math.round(clipped * 32767), true);
  });
  return bytes;
}
