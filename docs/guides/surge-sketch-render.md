# Render a Surge sketch match

Use the installed CLI to infer one Surge Simple patch from two audio files:

```bash
synth-setter-clap --guide_audio path/to/guide.wav --ref_audio path/to/reference.wav
```

`ref_audio` supplies the model's normalized mel/timbre conditioning. `guide_audio`
supplies loudness, spectral-centroid, and pitch sketch controls. Both inputs accept
mono or stereo audio at any sample rate supported by Pedalboard; the CLI resamples,
up-mixes, pads or trims, and prepares them on the model's four-second 44.1 kHz grid.
Guide loudness remains at source amplitude to match the stored training renders.

The command uses the immutable `flow_sketch_prelim` checkpoint and matching dataset
statistics. R2 downloads are SHA-256 verified and cached under the XDG synth-setter
cache. On Linux, Surge rendering automatically runs under the packaged headless X11
wrapper. Set `SYNTH_SETTER_PLUGIN_PATH` when Surge XT is not available at the managed
`plugins/Surge XT.vst3` alias.

Each invocation creates a unique directory under `outputs/synth-setter-clap/` and
retains these files locally:

- `pred.wav`: four-second stereo Surge render;
- `guide.wav`: normalized guide input;
- `ref.wav`: normalized reference input;
- `params.csv`: decoded Surge and note parameters.

The same four files upload to a unique prefix under
`r2://intermediate-data/eval/synth-setter-clap/`. The command prints the local path to
stderr and the final R2 URI as the last stdout line.
