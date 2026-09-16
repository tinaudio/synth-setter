export function authorMusicSketch({ pitch, start, end, loudness, centroid }) {
  if (!Number.isInteger(pitch) || pitch < 0 || pitch > 127)
    throw new Error("pitch must be a MIDI integer in [0,127]");
  if (
    ![start, end, loudness, centroid].every(Number.isFinite) ||
    start < 0 ||
    end > 4 ||
    end <= start ||
    Math.abs(loudness) > 1 ||
    Math.abs(centroid) > 1
  ) {
    throw new Error("invalid authored sketch range");
  }
  const sketch = new Float32Array(386 * 32);
  for (let frame = 0; frame < 32; frame++) {
    const active = frame / 8 >= start && frame / 8 < end;
    sketch[frame] = active ? loudness : -1;
    sketch[32 + frame] = active ? centroid : -1;
    // The trained PESTO profile has three activation bins per MIDI semitone.
    sketch[(2 + pitch * 3) * 32 + frame] = active ? 1 : 0;
  }
  return sketch;
}
