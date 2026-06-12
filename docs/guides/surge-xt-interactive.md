# Guide: Interactive Surge XT prediction & patch capture

> **Status**: Stable
> **Last Updated**: 2026-05-08
> **Source**: [`src/synth_setter/tools/surge_xt_interactive.py`](../../src/synth_setter/tools/surge_xt_interactive.py)

______________________________________________________________________

## What it is

`src/synth_setter/tools/surge_xt_interactive.py` opens the Surge XT VST3 editor with
ML-predicted (or dataset-derived) parameters preloaded, streams real-time
audio so you can audition and tweak the patch by ear, lets you snapshot
patches by pressing `p`, and after the session writes a directory
containing `train.h5` (plus optional `val.h5`/`test.h5`/`predict.h5`
siblings when `--checkpoint-path` is set).

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
- Surge XT VST3 at a known path (default `$SYNTH_SETTER_PLUGIN_PATH` when
  set and non-empty, else `plugins/Surge XT.vst3` — satisfied by
  `make install-surge-xt`).
- A base preset file. Selected automatically from
  `preset_paths[param_spec_name]` in `src/synth_setter/data/vst/__init__.py`
  (keyed by the value passed to `--param-spec-name`).
- A working audio output device. The tool opens a real-time audio
  stream via `pedalboard.io.AudioStream`; headless environments without
  ALSA/PulseAudio cannot run it.
- *Optional*: a prediction tensor (`pred-*.pt` from `src/synth_setter/cli/eval.py`) or
  an existing dataset (`*.h5`) to load parameters from.

## Quick start

`--param-spec-name` is required in every invocation; it selects the
parameter spec *and* the matching base preset (loaded by indexing
`preset_paths` with the value passed to `--param-spec-name`).

Bare audition — open the editor on the registry-selected base preset,
no preloaded params:

```bash
python -m synth_setter.tools.surge_xt_interactive --param-spec-name surge_xt
```

Audition a single prediction row (row index 0 inside `outputs/pred-0.pt`):

```bash
python -m synth_setter.tools.surge_xt_interactive \
    --param-spec-name surge_xt \
    --pred outputs/pred-0.pt:0
```

Audition a row from an existing HDF5 dataset:

```bash
python -m synth_setter.tools.surge_xt_interactive \
    --param-spec-name surge_xt \
    --dataset-ref outputs/test.h5:0
```

Record patches and render them into a fresh dataset directory:

```bash
python -m synth_setter.tools.surge_xt_interactive \
    --param-spec-name surge_xt \
    --pred outputs/pred-0.pt:0 \
    --output-dataset-dir-path outputs/curated-patches/
```

Render a deterministic test clip of the loaded patch to a WAV — useful
when no audio output device is available, and for reproducible audio
diffs of model predictions:

```bash
python -m synth_setter.tools.surge_xt_interactive \
    --param-spec-name surge_xt \
    --pred outputs/pred-0.pt:0 \
    --session-recording-path outputs/session.wav
```

When `--session-recording-path` is set, the live audio stream is
*replaced* by a fixed 10-second offline render: middle C held from
2 s to 4 s, with the surrounding silence capturing any release tail.
The render runs synchronously *before* the editor opens, so the WAV
depends only on the initially-loaded plugin state (preset + `--pred`
/ `--dataset-ref` params) and the same inputs always produce the
same WAV. After the render completes, the editor still opens and you
can still snapshot patches. No audio output device is required, but
the editor still needs a display. Combines freely with `--pred`,
`--dataset-ref`, and `--output-dataset-dir-path`.

`--pred` and `--dataset-ref` are mutually exclusive — passing both
raises `click.UsageError`.

## CLI reference

| Flag                        | Type               | Default                                                   | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------------- | ------------------ | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--plugin-path` / `-p`      | path               | `$SYNTH_SETTER_PLUGIN_PATH`, else `plugins/Surge XT.vst3` | Path to VST3 plugin. Defaults to `$SYNTH_SETTER_PLUGIN_PATH` or the in-repo bundle.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `--pred`                    | `PATH:BATCH_IDX`   | unset                                                     | Prediction reference. When set, the predicted row is decoded and applied to the plugin before the editor opens. Example: `outputs/pred-0.pt:0`.                                                                                                                                                                                                                                                                                                                                                                 |
| `--dataset-ref`             | `PATH:DATASET_IDX` | unset                                                     | Dataset reference. When set, the dataset row is decoded and applied to the plugin before the editor opens. Example: `outputs/test.h5:0`.                                                                                                                                                                                                                                                                                                                                                                        |
| `--param-spec-name`         | choice             | required                                                  | Parameter spec name — one of the keys registered in `src/synth_setter/data/vst/__init__.py` (`param_specs`). Selects which synth params are decoded from prediction/dataset rows, captured into recorded patches, and which base preset is loaded (the script indexes `preset_paths` with this value). There is no `--preset-path` flag — spec and preset travel together.                                                                                                                                      |
| `--output-dataset-dir-path` | path               | unset                                                     | Directory to create for the recorded patches. Must not already exist — `make_hdf5_dataset` writes fixed-size HDF5 datasets without `maxshape` and cannot append to existing files. After the editor is closed, patches captured via the keyboard loop (press `p` to record, `q` to quit) are rendered through the plugin and written to `train.h5` inside this directory via `synth_setter.data.vst.writers.make_hdf5_dataset` (plus `val.h5`/`test.h5`/`predict.h5` siblings when `--checkpoint-path` is set). |
| `--checkpoint-path`         | path               | unset                                                     | Optional checkpoint path to run standalone eval on after rendering captured patches. When set, triggers the `eval_patches` pipeline (`src/synth_setter/cli/eval.py mode=predict` → `predict_vst_audio.py` → `compute_audio_metrics.py`); see [`docs/design/eval-pipeline.md`](../design/eval-pipeline.md) for the full pipeline and `_METRIC_COLUMNS` in the script for the metric series produced.                                                                                                             |
| `--session-recording-path`  | path               | unset                                                     | Optional WAV file to render a deterministic test clip to. When set, the script renders a fixed `SESSION_RECORDING_DURATION_SECONDS` (10 s) WAV containing middle C from `NOTE_START` (2 s) to `NOTE_END` (4 s) through the loaded plugin and exits the audio thread. No live device output. Output depends only on plugin state (preset + `--pred` / `--dataset-ref` params) — same inputs always produce the same WAV. No-op when not set.                                                                     |

Tip — the help strings above are quoted verbatim from the Click
decorators in `src/synth_setter/tools/surge_xt_interactive.py`. Run
`python -m synth_setter.tools.surge_xt_interactive --help` to confirm the current
text.

## The interactive session

Once the plugin loads, the editor window opens and three things happen
in parallel:

1. **Audio thread** — silence is routed through Surge XT; the synth's
   own oscillators produce sound. Two modes:
   - Default (`play_audio`): writes plugin output to the default audio
     output device via `pedalboard.io.AudioStream`, resampling on the fly
     if `PLAYBACK_SAMPLE_RATE` differs from `SAMPLE_RATE`. You hear
     whatever the current patch is doing.
   - With `--session-recording-path` (`play_audio_recorded`): renders a
     deterministic `SESSION_RECORDING_DURATION_SECONDS` (10 s) clip
     (middle C from 2 s to 4 s) via a single `plugin.process(...)` call
     and writes it to the given WAV file. No live device output. Returns
     as soon as the offline render completes — the editor remains open
     for patch capture.
2. **Editor (main thread)** — the plugin's native GUI. Tweak knobs as
   you would in any host.
3. **Keyboard thread** — `keyboard_loop` reads keystrokes:
   - `p` — record the current values of every parameter named in
     `param_specs[--param-spec-name].synth_param_names` into an
     in-memory list.
   - `q` — set `stop_event` and exit.

Closing the editor window also sets `stop_event`, ending the audio
thread. The keyboard thread checks that event only between
`click.getchar()` calls, so it may not exit until another key is
pressed. Snapshots are buffered in memory; nothing is written until
the editor closes.

## Output dataset format

When `--output-dataset-dir-path` is set, the recorded patches are
rendered through the plugin via
[`make_hdf5_dataset`](../../src/synth_setter/data/vst/writers.py) and
written to `train.h5` inside that directory. With `--checkpoint-path`
also set, identical-content `val.h5`/`test.h5`/`predict.h5` siblings
are created next to `train.h5` so the eval pipeline has a `predict.h5`
to consume. Each file has these datasets:

| Dataset       | Shape                                           | Dtype     | Notes                                                                          |
| ------------- | ----------------------------------------------- | --------- | ------------------------------------------------------------------------------ |
| `audio`       | `(N, 2, sample_rate * signal_duration_seconds)` | `float16` | Stereo waveform. Compressed with Blosc2.                                       |
| `mel_spec`    | `(N, 2, 128, 401)`                              | `float32` | Mel spectrogram per channel. Compressed with Blosc2.                           |
| `param_array` | `(N, P)`                                        | `float32` | Encoded params (`ParamSpec.encode` output) in `[0, 1]`. `P = len(param_spec)`. |

Where `N = len(synth_patches)`, `sample_rate = 44100`, and
`signal_duration_seconds = 4.0` (constants at the top of
`src/synth_setter/tools/surge_xt_interactive.py`).

The audio attached attrs on the `audio` dataset record the rendering
config: `velocity`, `signal_duration_seconds`, `sample_rate`,
`channels`, `min_loudness`.

When `--checkpoint-path` is set, the `val.h5`/`test.h5`/`predict.h5`
siblings are byte-identical copies of `train.h5`; `predict.h5` is then
fed into the `eval_patches` pipeline.

> **Use a fresh `--output-dataset-dir-path` per session.** The script
> refuses an existing **directory**: `make_hdf5_dataset` creates fixed-size
> datasets (no `maxshape`), so re-running into an existing directory
> would either fail with a `ValueError` from the fixed-params length
> check or fail on write with an out-of-bounds index. If you need to
> combine multiple sessions, write each one to its own directory and
> concat downstream.

> **Note params are still randomized.** Only `fixed_synth_params_list`
> is passed to `make_hdf5_dataset`; MIDI note, velocity, and timing remain
> sampled from `param_spec`. The same captured patch produces multiple
> rows with different note conditions if you record `p` more than once
> (the synth params are identical; the note params will differ).

## End-to-end workflow

A `pred-*.pt` (from `src/synth_setter/cli/eval.py`) or an existing `*.h5` row supplies
the starting parameters; the live editor session captures user-curated
patches; on close, `make_hdf5_dataset` writes them to `train.h5` inside
`--output-dataset-dir-path` for downstream training. When
`--checkpoint-path` is set, the `eval_patches` function in
`src/synth_setter/tools/surge_xt_interactive.py` then runs the eval pipeline against
the captured patches — see its docstring for the predict → render →
metrics steps and their per-step validation.

Worked example:

```bash
# 1. Generate predictions for some target audio (outside this guide).
python -m synth_setter.cli.eval +experiment=surge/eval ckpt_path=...

# 2. Audition row 0 of the resulting predictions.
python -m synth_setter.tools.surge_xt_interactive \
    --param-spec-name surge_xt \
    --pred outputs/pred-0.pt:0

# 3. When you find sounds you like, record them and produce a dataset.
python -m synth_setter.tools.surge_xt_interactive \
    --param-spec-name surge_xt \
    --pred outputs/pred-0.pt:0 \
    --output-dataset-dir-path outputs/curated-patches/

# 4. (Optional) re-run with --checkpoint-path to also evaluate the
#    captured patches end-to-end (predict → render → metrics).
python -m synth_setter.tools.surge_xt_interactive \
    --param-spec-name surge_xt \
    --pred outputs/pred-0.pt:0 \
    --output-dataset-dir-path outputs/curated-patches/ \
    --checkpoint-path outputs/checkpoints/last.ckpt

# 5. Confirm the file:
python -c "import h5py; f = h5py.File('outputs/curated-patches/train.h5'); \
    print({k: f[k].shape for k in f})"
```

## Known limitations

These are accepted trade-offs, not bugs we plan to fix soon. Surface to
your teammates so they aren't blindsided.

- **0.5 s editor warm-up (non-Darwin only)** — `load_plugin` in
  `src/synth_setter/data/vst/core.py` briefly opens the editor (gated by
  [`_EDITOR_INIT_DELAY_SECONDS`](../../src/synth_setter/data/vst/core.py)) so the
  plugin populates its full parameter dict before we apply params. On
  slow machines parameter discovery may still be incomplete; the
  visible symptom is a `KeyError` from `set_params`, and the workaround
  is to bump the constant. On macOS the warmup is skipped entirely (see
  the `#714` SIGTRAP comment in `core.py`); the post-load `process(...)`
  flush in `render_params` is what commits Surge XT's preset state on
  that platform.
- **Plugin reloaded on every render in `make_hdf5_dataset`** — `render_params`
  calls `load_plugin(plugin_path)` per sample
  ([`src/synth_setter/data/vst/core.py`](../../src/synth_setter/data/vst/core.py)). This is an
  intentional workaround for a silent / repeated-render bug surfaced
  during this branch's development; without per-call reloads, the
  plugin retained stale state. The cost is ~7 s of plugin-load
  overhead per sample, so capturing 100 patches at 4 s each takes
  more than ten minutes. Acceptable for human-scale capture; do not
  use this path for large pipeline runs.
- **Silent captured patches fast-fail** — `generate_sample` raises
  `ValueError` when `fixed_synth_params` is set and the render falls
  below `MAKE_DATASET_MIN_LOUDNESS = -55.0`
  ([`src/synth_setter/tools/surge_xt_interactive.py`](../../src/synth_setter/tools/surge_xt_interactive.py)).
  The synth patch dominates loudness, so re-sampling note params alone
  can't lift a silent patch above threshold; rather than loop, the
  whole `make_hdf5_dataset` call aborts and points at the offending patch.
  Workaround: only press `p` while you can hear the patch — once a
  silent patch is captured, the session's dataset render will fail.
- **Blocking keyboard input** — `keyboard_loop` uses
  `click.getchar()`, which only checks `stop_event` between
  keystrokes. After the editor closes, you may need to press one key
  to let the script proceed to dataset rendering. Documented inline
  in [`src/synth_setter/tools/surge_xt_interactive.py`](../../src/synth_setter/tools/surge_xt_interactive.py).
- **No explicit lock on plugin parameters** — the audio thread reads
  the plugin's parameter state to render the next buffer at the same
  time the GUI thread may be writing it. `pedalboard` may handle this
  internally, but the contract isn't documented upstream. In practice
  this hasn't caused audible glitching, but a long session under load
  could.
- **No append support for `--output-dataset-dir-path`** — the script
  refuses an existing directory; see the warning in *Output dataset
  format* above.

## Troubleshooting

**`AudioStream` fails to open / no audio device.** You're running
headless or your default device is not configured. Use
`--session-recording-path outputs/session.wav` —
`play_audio_recorded` renders a deterministic 10-second middle-C clip
of the loaded patch directly to a WAV via `pedalboard.io.AudioFile`,
without ever opening an audio device. Enough to confirm the plugin
loaded and the params applied without any live audio.

**Sample-rate mismatch.** `play_audio` resamples on the fly via
`StreamResampler` when `PLAYBACK_SAMPLE_RATE != SAMPLE_RATE`. If your
device only supports 48 kHz, edit `PLAYBACK_SAMPLE_RATE = 48000` at
the top of the script.

**`KeyError` from `record_patch`.** A param name in the spec isn't in
`plugin.parameters`. Likely causes: wrong `--param-spec-name` for the
loaded plugin, or the preset put the plugin into a state where some
params are hidden. Try `--param-spec-name surge_xt`; that key resolves
to a base preset via `preset_paths` in `src/synth_setter/data/vst/__init__.py`.

**Prediction tensor shape mismatch.** `--pred` requires the second
dim of the loaded tensor to match `param_specs[--param-spec-name]` row
length. Print `pred_tensor.shape` and compare to
`len(param_specs[name])`.

**Dataset generation fails with `ValueError: fixed_synth_params render produced loudness …`.** One of the captured patches was below
`MAKE_DATASET_MIN_LOUDNESS`. The error message includes the measured
loudness; re-run the session and only press `p` while you can hear the
patch. See *Known limitations*.

**`ValueError: fixed_synth_params_list has length …`.** You re-ran
with an existing `--output-dataset-dir-path`. Use a fresh directory;
see the warning in *Output dataset format*.

## Related

- [`docs/design/eval-pipeline.md`](../design/eval-pipeline.md) —
  where `pred-*.pt` files come from (and the pipeline that
  `--checkpoint-path` invokes: a thin wrapper over `src/synth_setter/cli/eval.py mode=predict` + `predict_vst_audio.py` + `compute_audio_metrics.py`).
- [`docs/glossary.md`](../glossary.md) — `param_spec`, VST, mel
  spectrogram.
- [`docs/design/data-pipeline.md`](../design/data-pipeline.md) —
  downstream consumer of the produced HDF5.
- [`docs/getting-started.md`](../getting-started.md) — env setup and
  Surge XT install.
