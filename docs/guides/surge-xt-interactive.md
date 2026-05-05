# Guide: Interactive Surge XT prediction & patch capture

> **Status**: Stable
> **Last Updated**: 2026-05-02
> **Source**: [`scripts/surge_xt_interactive.py`](../../scripts/surge_xt_interactive.py)

> Last refresh: `--session-recording-path` now renders a deterministic
> 10-second middle-C clip (was: a tee of the live stream's first 20 s).

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

Bare audition — open the editor on the base preset for a given spec, no preloaded params:

```bash
python scripts/surge_xt_interactive.py --param-spec-name surge_xt
```

Audition a single prediction row (row index 0 inside `outputs/pred-0.pt`):

```bash
python scripts/surge_xt_interactive.py --param-spec-name surge_xt --pred outputs/pred-0.pt:0
```

Audition a row from an existing HDF5 dataset:

```bash
python scripts/surge_xt_interactive.py --param-spec-name surge_xt --dataset-ref outputs/test.h5:0
```

Record patches and render them into a fresh dataset:

```bash
python scripts/surge_xt_interactive.py \
    --param-spec-name surge_xt \
    --pred outputs/pred-0.pt:0 \
    --output-dataset-path outputs/curated-patches.h5
```

Render a deterministic test clip of the loaded patch to a WAV — useful
when no audio output device is available, and for reproducible audio
diffs of model predictions:

```bash
python scripts/surge_xt_interactive.py \
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
`--dataset-ref`, and `--output-dataset-path`.

`--pred` and `--dataset-ref` are mutually exclusive — passing both
raises `click.UsageError`.

## CLI reference

| Flag                       | Type               | Default                        | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--plugin-path` / `-p`     | path               | `plugins/Surge XT.vst3`        | Path to VST3 plugin.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `--preset-path` / `-r`     | path               | `presets/surge-base.vstpreset` | Base preset to load before applying any `--pred` / `--dataset-ref` params. Pick the preset that matches `--param-spec-name` (see `preset_paths` in `src/data/vst/__init__.py`).                                                                                                                                                                                                                                                             |
| `--pred`                   | `PATH:BATCH_IDX`   | unset                          | Prediction reference. When set, the predicted row is decoded and applied to the plugin before the editor opens. Example: `outputs/pred-0.pt:0`.                                                                                                                                                                                                                                                                                             |
| `--dataset-ref`            | `PATH:DATASET_IDX` | unset                          | Dataset reference. When set, the dataset row is decoded and applied to the plugin before the editor opens. Example: `outputs/test.h5:0`.                                                                                                                                                                                                                                                                                                    |
| `--param-spec-name`        | str                | `surge_xt`                     | Parameter spec name (key into `param_specs`) used to decode prediction/dataset rows applied to the plugin and to enumerate which synth params are captured when recording patches. Use the matching `preset_paths` entry for `--preset-path`.                                                                                                                                                                                               |
| `--output-dataset-path`    | path               | unset                          | HDF5 file to write recorded patches to. After the editor is closed, patches captured via the keyboard loop (press `p` to record, `q` to quit) are rendered through the plugin and written to this dataset via `src.data.vst.generate_vst_dataset.make_dataset`. Must not already exist — `make_dataset` writes fixed-size HDF5 datasets and cannot append.                                                                                  |
| `--session-recording-path` | path               | unset                          | Optional WAV file to render a deterministic test clip to. When set, the script renders a fixed `SESSION_RECORDING_DURATION_SECONDS` (10 s) WAV containing middle C from `NOTE_START` (2 s) to `NOTE_END` (4 s) through the loaded plugin and exits the audio thread. No live device output. Output depends only on plugin state (preset + `--pred` / `--dataset-ref` params) — same inputs always produce the same WAV. No-op when not set. |

Tip — the help strings above are quoted verbatim from the Click
decorators in `scripts/surge_xt_interactive.py`. Run
`python scripts/surge_xt_interactive.py --help` to confirm the current
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
                             │  surge_xt_interactive   │ ──┐
  *.h5 dataset ──row N─────▶ │  (audition + capture)   │   │
                             └─────────────────────────┘   │
                                       │                   │
                              ▼ user presses 'p'           │
                            synth_patches: list[dict]      │
                                       │                   │
                                       ▼                   │
                            make_dataset(                  │
                                fixed_synth_params_list=…) │
                                       │                   │
                                       ▼                   │
                            outputs/curated-patches.h5 ◀───┘
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
python scripts/surge_xt_interactive.py --param-spec-name surge_xt --pred outputs/pred-0.pt:0

# 3. When you find sounds you like, record them and produce a dataset.
python scripts/surge_xt_interactive.py \
    --param-spec-name surge_xt \
    --pred outputs/pred-0.pt:0 \
    --output-dataset-path outputs/curated-patches.h5

# 4. Confirm the file:
python -c "import h5py; f = h5py.File('outputs/curated-patches.h5'); \
    print({k: f[k].shape for k in f})"
```

## Known limitations

These are accepted trade-offs, not bugs we plan to fix soon. Surface to
your teammates so they aren't blindsided.

- **0.5 s editor warm-up (non-Darwin only)** — `load_plugin` in
  `src/data/vst/core.py` briefly opens the editor (gated by
  [`_EDITOR_INIT_DELAY_SECONDS`](../../src/data/vst/core.py)) so the
  plugin populates its full parameter dict before we apply params. On
  slow machines parameter discovery may still be incomplete; the
  visible symptom is a `KeyError` from `set_params`, and the workaround
  is to bump the constant. On macOS the warmup is skipped entirely (see
  the `#714` SIGTRAP comment in `core.py`); the post-load `process(...)`
  flush in `render_params` is what commits Surge XT's preset state on
  that platform.
- **Plugin reloaded on every render in `make_dataset`** — `render_params`
  calls `load_plugin(plugin_path)` per sample
  ([`src/data/vst/core.py`](../../src/data/vst/core.py)). This is an
  intentional workaround for a silent / repeated-render bug surfaced
  during this branch's development; without per-call reloads, the
  plugin retained stale state. The cost is ~7 s of plugin-load
  overhead per sample, so capturing 100 patches at 4 s each takes
  more than ten minutes. Acceptable for human-scale capture; do not
  use this path for large pipeline runs.
- **Silent captured patches fast-fail** — `generate_sample` raises
  `ValueError` when `fixed_synth_params` is set and the render falls
  below `MAKE_DATASET_MIN_LOUDNESS = -50.0`
  ([`scripts/surge_xt_interactive.py`](../../scripts/surge_xt_interactive.py)).
  The synth patch dominates loudness, so re-sampling note params alone
  can't lift a silent patch above threshold; rather than loop, the
  whole `make_dataset` call aborts and points at the offending patch.
  Workaround: only press `p` while you can hear the patch — once a
  silent patch is captured, the session's dataset render will fail.
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
params are hidden. Try `--param-spec-name surge_xt` against the
default Surge XT preset to rule out config issues.

**Prediction tensor shape mismatch.** `--pred` requires the second
dim of the loaded tensor to match `param_specs[--param-spec-name]` row
length. Print `pred_tensor.shape` and compare to
`len(param_specs[name])`.

**Dataset generation fails with `ValueError: fixed_synth_params render produced loudness …`.** One of the captured patches was below
`MAKE_DATASET_MIN_LOUDNESS`. The error message includes the measured
loudness; re-run the session and only press `p` while you can hear the
patch. See *Known limitations*.

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
