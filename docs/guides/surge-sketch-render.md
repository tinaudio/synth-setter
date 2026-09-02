# Render a Surge sketch match

Use the checked-in third-party corpora for the primary smoke run:

```bash
uv run synth-setter-sketch-render --sketch-corpus nsynth_test --content-corpus esc50 --seed 0
```

File inputs use the same fixed model grid:

```bash
uv run synth-setter-sketch-render \
  --sketch-audio path/to/sketch.wav \
  --content-audio path/to/content.wav
```

File and corpus sources can be mixed independently:

```bash
uv run synth-setter-sketch-render \
  --sketch-audio path/to/sketch.wav \
  --content-corpus esc50 \
  --seed 17
```

Exactly one of `--sketch-audio` or `--sketch-corpus` is required, and exactly one
of `--content-audio` or `--content-corpus` is required. `nsynth_test` and `esc50`
refer directly to the checked-in third-party corpus configs; no Hydra group prefix
is needed.

`--content-audio` or `--content-corpus` supplies normalized mel/timbre conditioning.
`--sketch-audio` or `--sketch-corpus` supplies loudness, spectral-centroid, and pitch
sketch controls. Sources are decoded, resampled, up-mixed, padded or trimmed onto a
four-second, stereo 44.1 kHz grid. File samples must be finite and within `[-1, 1]`.

`--seed` defaults to `0`. It deterministically selects corpus rows and seeds the
model's local generator without changing process-global Python or Torch RNG state.
When both inputs use the same corpus, two distinct rows are selected without
replacement.

`--content-cfg-strength` controls content-mel guidance and
`--sketch-cfg-strength` controls sketch guidance. Both accept finite nonnegative
values, including zero. Omit either flag to use its checkpoint value; legacy
checkpoints without a sketch value use the effective content strength.

The command uses the immutable `flow_sketch_prelim` checkpoint and matching dataset
statistics. R2 downloads are SHA-256 verified and cached under the XDG synth-setter
cache. On Linux, Surge rendering automatically runs under the packaged headless X11
wrapper. Set `SYNTH_SETTER_PLUGIN_PATH` when Surge XT is not available at the managed
`plugins/Surge XT.vst3` alias.

Each invocation creates a unique directory under `outputs/synth-setter-sketch-render/`.
Set `SYNTH_SETTER_SKETCH_OUTPUT_ROOT` to use another local root. The retained files are:

- `pred.wav`: four-second stereo Surge render;
- `guide.wav`: grid-fitted sketch input;
- `ref.wav`: grid-fitted content input;
- `params.csv`: decoded Surge and note parameters;
- `manifest.json`: input, checkpoint, statistics, render, seed, and destination provenance.

File provenance records the resolved path and SHA-256 digest. Corpus provenance records
the config name, dataset URI, resolved Lance version, and row index.

The same artifacts upload to a unique prefix under
`r2://intermediate-data/eval/synth-setter-sketch-render/`. Set
`SYNTH_SETTER_SKETCH_UPLOAD_PREFIX` to use another `r2://` prefix. The command prints
the local path to stderr and the final R2 URI as the last stdout line. If upload fails
after inference, resume it from the retained directory:

```bash
uv run synth-setter-sketch-render --retry-upload outputs/synth-setter-sketch-render/<run_id>
```
