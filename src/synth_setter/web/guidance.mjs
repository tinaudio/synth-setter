const MODES = new Set(["both", "mel_only", "sketch_only", "unconditional"]);

// Mirrors synth_setter.models.flow_onnx.branch_weights: weights over the
// unconditional, sketch-only, content-only, and full velocity branches.
export function branchWeights(mode, contentCfg, sketchCfg) {
  if (!MODES.has(mode)) throw new Error(`unknown conditioning mode ${mode}`);
  if (![contentCfg, sketchCfg].every(Number.isFinite)) throw new Error("guidance must be finite");
  if (mode === "both") return Float32Array.of(1 - sketchCfg, sketchCfg - contentCfg, 0, contentCfg);
  if (mode === "mel_only") return Float32Array.of(1 - contentCfg, 0, contentCfg, 0);
  if (mode === "sketch_only") return Float32Array.of(1 - sketchCfg, sketchCfg, 0, 0);
  return Float32Array.of(1, 0, 0, 0);
}
