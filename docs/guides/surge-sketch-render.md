# Render a Surge sketch match

Use the installed CLI to infer one Surge Simple patch from two audio files:

```bash
synth-setter-sketch-render \
  --guide-audio path/to/guide.wav \
  --reference-audio path/to/reference.wav \
  --content-cfg-strength 2 \
  --sketch-cfg-strength 3
```

`--content-cfg-strength` controls reference-mel guidance, while
`--sketch-cfg-strength` controls sketch guidance. Both accept finite nonnegative values,
including zero. Omit either flag to use its checkpoint value; legacy checkpoints without
a sketch value use the effective content strength.

`--reference-audio` supplies the model's normalized mel/timbre conditioning.
`--guide-audio` supplies loudness, spectral-centroid, and pitch sketch controls.

Both inputs accept mono or stereo audio at any sample rate supported by Pedalboard; the CLI resamples,
up-mixes, pads or trims, and prepares them on the model's four-second 44.1 kHz grid.
Input samples must be finite and within `[-1, 1]`; guide loudness remains at source
amplitude to match the stored training renders.

The command uses the immutable `flow_sketch_prelim` checkpoint and matching dataset
statistics. R2 downloads are SHA-256 verified and cached under the XDG synth-setter
cache. On Linux, Surge rendering automatically runs under the packaged headless X11
wrapper. Set `SYNTH_SETTER_PLUGIN_PATH` when Surge XT is not available at the managed
`plugins/Surge XT.vst3` alias.

Each invocation creates a unique directory under `outputs/synth-setter-sketch-render/` and
retains these files locally. Set `SYNTH_SETTER_SKETCH_OUTPUT_ROOT` to use another
local root.

- `pred.wav`: four-second stereo Surge render;
- `guide.wav`: grid-fitted guide input;
- `ref.wav`: grid-fitted reference input;
- `params.csv`: decoded Surge and note parameters;
- `manifest.json`: checkpoint, statistics, render, and destination provenance.

The same artifacts upload to a unique prefix under
`r2://intermediate-data/eval/synth-setter-sketch-render/`. Set
`SYNTH_SETTER_SKETCH_UPLOAD_PREFIX` to use another `r2://` prefix. The command prints
the local path to stderr and the final R2 URI as the last stdout line. If that upload
fails after inference, resume it from the retained directory:

```bash
synth-setter-sketch-render --retry-upload outputs/synth-setter-sketch-render/<run_id>
```
