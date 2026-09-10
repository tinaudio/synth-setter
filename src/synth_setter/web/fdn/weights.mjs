// Weights over the velocity graph's four branches [unconditional, sketch_only, content_only, full].
// Mode "both" reproduces the three-branch CFG of vst_flow_matching_module.multi_cfg_velocity:
// u + s·(sketch − u) + c·(full − sketch).
export const CONDITIONING_MODES = ["both", "mel_only", "sketch_only", "unconditional"];

export function branchWeights(mode, contentCfg, sketchCfg) {
  for (const strength of [contentCfg, sketchCfg]) {
    if (!Number.isFinite(strength) || strength < 0) throw new Error("guidance strengths must be finite and non-negative");
  }
  const [c, s] = [contentCfg, sketchCfg];
  switch (mode) {
    case "both":
      return Float32Array.from([1 - s, s - c, 0, c]);
    case "mel_only":
      return Float32Array.from([1 - c, 0, c, 0]);
    case "sketch_only":
      return Float32Array.from([1 - s, s, 0, 0]);
    case "unconditional":
      return Float32Array.from([1, 0, 0, 0]);
    default:
      throw new Error(`unknown conditioning mode: ${mode}`);
  }
}
