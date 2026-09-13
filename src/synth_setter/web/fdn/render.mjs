// Offline FaustWasm render of one decoded householder row: a fresh DSP instance per render so
// the in-DSP impulse fires at frame 0, exactly like pyFDN's impulse excitation.
import { applyCanonicalPatch, createOfflineSynth, renderNote } from "../faust/runtime.mjs";
import { canonicalPatch } from "./patch.mjs";

const BLOCK_SIZE = 128;

export async function renderImpulseResponse(artifact, native, { sampleRate, frames }) {
  const synth = await createOfflineSynth(artifact, { sampleRate, blockSize: BLOCK_SIZE });
  applyCanonicalPatch(synth, artifact.manifest, canonicalPatch(native));
  const [channel] = renderNote(synth, { frames, note: 60, velocity: 100, startFrame: 0, endFrame: frames });
  return Float64Array.from(channel);
}
