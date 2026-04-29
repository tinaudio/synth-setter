# Guide: Interactive Surge XT prediction & patch capture

> **Status**: Stable
> **Last Updated**: 2026-04-29
> **Source**: [`scripts/surge_xt_interactive.py`](../../scripts/surge_xt_interactive.py)

______________________________________________________________________

## What it is

`scripts/surge_xt_interactive.py` opens the Surge XT VST3 editor with
ML-predicted (or dataset-derived) parameters preloaded, streams real-time
audio so you can audition and tweak the patch by ear, lets you snapshot
patches by pressing `p`, and after the session renders those snapshots
into a labeled HDF5 training dataset.

It's a human-in-the-loop tool for producing high-quality (audio, params)
training pairs that random sampling can't reach.

## When to use it

- Auditioning a model's predicted parameters as audio — does row 42 of
  `pred-0.pt` actually sound like the target?
- Capturing curated patches by ear, then rendering them with the same
  pipeline the training data uses.
- Producing labeled training pairs from human sound design — patches
  that random sampling won't find but that the model needs to learn.

## Prerequisites

- Project Python env. `make install` plus `make install-surge-xt` covers
  everything; see [getting-started](../getting-started.md) for the full
  walkthrough.
- Surge XT VST3 at a known path (default `plugins/Surge XT.vst3` —
  satisfied by `make install-surge-xt`).
- A base preset file (default `presets/surge-base.vstpreset`).
- A working audio output device. The tool opens a real-time audio
  stream via `pedalboard.io.AudioStream`; headless environments without
  ALSA/PulseAudio cannot run it.
- *Optional*: a prediction tensor (`pred-*.pt` from `src/eval.py`) or
  an existing dataset (`*.h5`) to load parameters from.

## Quick start

Bare audition — open the editor on the base preset, no preloaded params:

```bash
python scripts/surge_xt_interactive.py
```

Audition a single prediction row (row index 0 inside `outputs/pred-0.pt`):

```bash
python scripts/surge_xt_interactive.py --pred outputs/pred-0.pt:0
```

Audition a row from an existing HDF5 dataset:

```bash
python scripts/surge_xt_interactive.py --dataset-ref outputs/test.h5:0
```

Record patches and render them into a fresh dataset:

```bash
python scripts/surge_xt_interactive.py \
    --pred outputs/pred-0.pt:0 \
    --output-dataset-path outputs/curated-patches.h5
```

`--pred` and `--dataset-ref` are mutually exclusive — passing both
raises `click.UsageError`.

## CLI reference

| Flag                    | Type               | Default                        | Notes                                                                                                                                                                                                                                                           |
| ----------------------- | ------------------ | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--plugin-path` / `-p`  | path               | `plugins/Surge XT.vst3`        | Path to VST3 plugin.                                                                                                                                                                                                                                            |
| `--preset-path` / `-r`  | path               | `presets/surge-base.vstpreset` | Base preset to load before applying any `--pred` / `--dataset-ref` params.                                                                                                                                                                                      |
| `--pred`                | `PATH:BATCH_IDX`   | unset                          | Prediction reference. When set, the predicted row is decoded and applied to the plugin before the editor opens. Example: `outputs/pred-0.pt:0`.                                                                                                                 |
| `--dataset-ref`         | `PATH:DATASET_IDX` | unset                          | Dataset reference. When set, the dataset row is decoded and applied to the plugin before the editor opens. Example: `outputs/test.h5:0`.                                                                                                                        |
| `--param-spec-name`     | str                | `surge_xt`                     | Parameter spec name (key into `param_specs`) used to decode prediction/dataset rows applied to the plugin and to enumerate which synth params are captured when recording patches.                                                                              |
| `--output-dataset-path` | path               | unset                          | HDF5 file to write recorded patches to. After the editor is closed, patches captured via the keyboard loop (press `p` to record, `q` to quit) are rendered through the plugin and written to this dataset via `src.data.vst.generate_vst_dataset.make_dataset`. |

Tip — the help strings above are quoted verbatim from the Click
decorators in `scripts/surge_xt_interactive.py`. Run
`python scripts/surge_xt_interactive.py --help` to confirm the current
text.

## The interactive session

Once the plugin loads, the editor window opens and three things happen
in parallel:

1. **Audio thread** — silence is routed through Surge XT
   (`play_audio`); the synth's own oscillators produce sound, so you
   hear whatever the current patch is doing. Resamples on the fly if
   `PLAYBACK_SAMPLE_RATE` differs from `SAMPLE_RATE`.
2. **Editor (main thread)** — the plugin's native GUI. Tweak knobs as
   you would in any host.
3. **Keyboard thread** — `keyboard_loop` reads keystrokes:
   - `p` — record the current values of every parameter named in
     `param_specs[--param-spec-name].synth_param_names` into an
     in-memory list.
   - `q` — set `stop_event` and exit.

Closing the editor window also sets `stop_event`, ending the audio
thread and signalling the keyboard thread to exit. Snapshots are
buffered in memory; nothing is written until the editor closes.

## Output dataset format

When `--output-dataset-path` is set, the recorded patches are rendered
through the plugin via
[`make_dataset`](../../src/data/vst/generate_vst_dataset.py) and
written to an HDF5 file with these datasets:

| Dataset       | Shape                                           | Dtype     | Notes                                                                          |
| ------------- | ----------------------------------------------- | --------- | ------------------------------------------------------------------------------ |
| `audio`       | `(N, 2, sample_rate * signal_duration_seconds)` | `float16` | Stereo waveform. Compressed with Blosc2.                                       |
| `mel_spec`    | `(N, 2, 128, 401)`                              | `float32` | Mel spectrogram per channel. Compressed with Blosc2.                           |
| `param_array` | `(N, P)`                                        | `float32` | Encoded params (`ParamSpec.encode` output) in `[0, 1]`. `P = len(param_spec)`. |

Where `N = len(synth_patches)`, `sample_rate = 44100`, and
`signal_duration_seconds = 4.0` (constants at the top of
`scripts/surge_xt_interactive.py`).

The audio attached attrs on the `audio` dataset record the rendering
config: `velocity`, `signal_duration_seconds`, `sample_rate`,
`channels`, `min_loudness`.

> **Use a fresh `--output-dataset-path` per session.** The current
> implementation does not support appending to a finalized HDF5 file:
> `make_dataset` creates fixed-size datasets (no `maxshape`), so
> re-running with an existing file will either fail with a
> `ValueError` from the fixed-params length check or fail on write
> with an out-of-bounds index. If you need to combine multiple
> sessions, write each one to its own file and concat downstream.

> **Note params are still randomized.** Only `fixed_synth_params_list`
> is passed to `make_dataset`; MIDI note, velocity, and timing remain
> sampled from `param_spec`. The same captured patch produces multiple
> rows with different note conditions if you record `p` more than once
> (the synth params are identical; the note params will differ).

## End-to-end workflow

```
                            ┌─────────────────────────┐
  src/eval.py  ──pred-*.pt──▶│                         │
                            │  surge_xt_interactive   │  ──┐
  *.h5 dataset ──row N─────▶│  (audition + capture)   │    │
                            └─────────────────────────┘    │
                                       │                    │
                              ▼ user presses 'p'            │
                            synth_patches: list[dict]        │
                                       │                    │
                                       ▼                    │
                            make_dataset(                    │
                                fixed_synth_params_list=…)   │
                                       │                    │
                                       ▼                    │
                            outputs/curated-patches.h5  ◀───┘
                            (audio, mel_spec, param_array)
                                       │
                                       ▼
                              downstream training
```

Worked example:

```bash
# 1. Generate predictions for some target audio (outside this guide).
python -m src.eval +experiment=surge/eval ckpt_path=...

# 2. Audition row 0 of the resulting predictions.
python scripts/surge_xt_interactive.py --pred outputs/pred-0.pt:0

# 3. When you find sounds you like, record them and produce a dataset.
python scripts/surge_xt_interactive.py \
    --pred outputs/pred-0.pt:0 \
    --output-dataset-path outputs/curated-patches.h5

# 4. Confirm the file:
python -c "import h5py; f = h5py.File('outputs/curated-patches.h5'); \
    print({k: f[k].shape for k in f})"
```

## Known limitations

These are accepted trade-offs, not bugs we plan to fix soon. Surface to
your teammates so they aren't blindsided.

- **0.5 s editor warm-up** — `_prepare_plugin` in `src/data/vst/core.py`
  shows the editor for a fixed
  [`_PREPARE_PLUGIN_SLEEP_SECONDS = 0.5`](../../src/data/vst/core.py)
  to let the plugin populate its full parameter dict before we apply
  params. On slow machines, parameter discovery may still be
  incomplete; the visible symptom is a `KeyError` from
  `set_params`. Workaround: bump the constant.
- **Plugin reloaded on every render in `make_dataset`** — `render_params`
  calls `load_plugin(plugin_path)` per sample
  ([`src/data/vst/core.py`](../../src/data/vst/core.py)). This is an
  intentional workaround for a silent / repeated-render bug surfaced
  during this branch's development; without per-call reloads, the
  plugin retained stale state. The cost is ~7 s of plugin-load
  overhead per sample, so capturing 100 patches at 4 s each takes
  more than ten minutes. Acceptable for human-scale capture; do not
  use this path for large pipeline runs.
- **Loudness-retry loop has no iteration cap** — `generate_sample`
  resamples params and re-renders until output integrated loudness
  exceeds `MAKE_DATASET_MIN_LOUDNESS = -50.0`
  ([`scripts/surge_xt_interactive.py`](../../scripts/surge_xt_interactive.py)).
  When `fixed_synth_params` is set, the params don't change between
  retries — so if the captured patch is near-silent, dataset
  generation hangs on that row. Workaround: only press `p` while you
  can hear the patch.
- **Blocking keyboard input** — `keyboard_loop` uses
  `click.getchar()`, which only checks `stop_event` between
  keystrokes. After the editor closes, you may need to press one key
  to let the script proceed to dataset rendering. Documented inline
  in [`scripts/surge_xt_interactive.py`](../../scripts/surge_xt_interactive.py).
- **No explicit lock on plugin parameters** — the audio thread reads
  the plugin's parameter state to render the next buffer at the same
  time the GUI thread may be writing it. `pedalboard` may handle this
  internally, but the contract isn't documented upstream. In practice
  this hasn't caused audible glitching, but a long session under load
  could.
- **No append support for `--output-dataset-path`** — see the warning
  in *Output dataset format* above.

## Troubleshooting

**`AudioStream` fails to open / no audio device.** You're running
headless or your default device is not configured. There's no public
CLI for offline rendering, but `render_to_wav` in
`scripts/surge_xt_interactive.py` is the library-level escape hatch
— write a small wrapper or call it from a notebook.

**Sample-rate mismatch.** `play_audio` resamples on the fly via
`StreamResampler` when `PLAYBACK_SAMPLE_RATE != SAMPLE_RATE`. If your
device only supports 48 kHz, edit `PLAYBACK_SAMPLE_RATE = 48000` at
the top of the script.

**`KeyError` from `record_patch`.** A param name in the spec isn't in
`plugin.parameters`. Likely causes: wrong `--param-spec-name` for the
loaded plugin, or the preset put the plugin into a state where some
params are hidden. Try `--param-spec-name surge_xt` against the
default Surge XT preset to rule out config issues.

**Prediction tensor shape mismatch.** `--pred` requires the second
dim of the loaded tensor to match `param_specs[--param-spec-name]` row
length. Print `pred_tensor.shape` and compare to
`len(param_specs[name])`.

**Dataset generation hangs after recording.** Likely the loudness
retry loop on a near-silent patch; see *Known limitations*.

**`ValueError: fixed_synth_params_list has length …`.** You re-ran
with an existing `--output-dataset-path`. Use a fresh path; see the
warning in *Output dataset format*.

## Related

- [`docs/design/eval-pipeline.md`](../design/eval-pipeline.md) —
  where `pred-*.pt` files come from.
- [`docs/glossary.md`](../glossary.md) — `param_spec`, VST, mel
  spectrogram.
- [`docs/design/data-pipeline.md`](../design/data-pipeline.md) —
  downstream consumer of the produced HDF5.
- [`docs/getting-started.md`](../getting-started.md) — env setup and
  Surge XT install.
