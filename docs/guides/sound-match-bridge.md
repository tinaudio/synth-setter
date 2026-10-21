# Sound-match bridge (predict-capture)

The Python half of the live sound-match bridge ([#1787](https://github.com/tinaudio/synth-setter/issues/1787)).
A CLAP host plugin (C++ side, separate repo) hosts Surge XT in a DAW, captures
4 s of external audio on a MIDI trigger, and spawns this repo's
`synth-setter-predict-capture`; the CLI predicts the Surge patch that best
matches the sound and writes it as a CSV the plugin applies live. The two
halves communicate **only** through the file contract below.

## The file contract

- Bridge root: `~/synth-setter-bridge/` with `capture-sample-dir/` and
  `param-prediction-dir/`; each side `mkdir -p`s what it needs.

- Capture: `capture-sample-dir/<uuid>.wav` — float32 stereo, host sample rate
  (44.1 kHz **not** guaranteed; resampling happens on this side), 4.0 s. The
  C++ side writes `<uuid>.wav.tmp` then renames; `*.tmp` files are ignored.

- Invocation (C++ `posix_spawn`, absolute paths):

  ```bash
  python -m synth_setter.cli.predict_capture <abs-wav-path> \
    --prediction-dir <abs-bridge>/param-prediction-dir
  ```

- Output: `param-prediction-dir/<uuid>/params.csv` (schema
  `pb_name,clap_name,clap_module_name,clap_param_id,clap_value`, one row per
  mapped synth parameter, `clap_value` already in the parameter's native CLAP
  domain) plus `pred-0.pt` (raw prediction tensor, debugging aid).

- **Failure semantics:** any error exits nonzero; `params.csv` is written as
  `.tmp` + atomic rename and a retried uuid unlinks the previous run's CSV up
  front, so the *absence* of `params.csv` is the failure signal. The C++ side
  times out after 180 s per uuid.

## CLI options

A Surge FXP can be rendered as the input target instead of supplying a capture
WAV. The fixed render is stereo 44.1 kHz with MIDI note 60 at velocity 100,
a 2 s note, and a 2 s release tail; it is saved beside the predictions as
`target.wav`:

```bash
synth-setter-predict-capture \
  --fxp presets/surge-base.fxp \
  --prediction-dir outputs/predictions \
  --checkpoint r2://models/inverse/surge.ckpt \
  --checkpoint-sha256 <sha256>
```

The WAV positional argument and `--fxp` are mutually exclusive. Output and log
names use the selected input's stem.

This complete `surge_simple` example uses the checkpoint and matching mel
statistics from the 2M-sample SurgePy training run. The checkpoint resolver
accepts R2 directly; materialize the statistics locally before invoking the
CLI:

```bash
CHECKPOINT_SHA=87a0fcb20f9efbcb56fffc6e3ee6e1c8743c47013c297e33516b472543468c07
STATS_SHA=c0c45d75a8b77004b3802c761bc77b5b34e7709a08343b2cf70fee04b7f52a19
STATS=/tmp/surge-simple-${STATS_SHA}-stats.npz
rclone copyto \
  "r2:experiments/browser-evaluation/${STATS_SHA}/stats.npz" "$STATS" --checksum

synth-setter-predict-capture \
  --fxp presets/surge-simple.fxp \
  --prediction-dir outputs/predictions \
  --checkpoint \
    "r2://experiments/browser-evaluation/${CHECKPOINT_SHA}/checkpoint.ckpt" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --stats-file "$STATS" \
  --param-spec-name surge_simple \
  --render-audio
```

`--render-audio` is opt-in. After inference, it decodes the predicted synth
parameters and renders MIDI note 60 at velocity 100 from 0–2 s into a 4 s,
stereo 44.1 kHz `pred.wav`. The native SurgePy render uses the selected spec's
registered FXP baseline and verified packaged parameter map, never the input
target FXP. `params.csv` is published only after this requested render succeeds.
Without the flag, inference retains its prior WAV/FXP behavior and does not
load the Surge engine.

`--checkpoint` accepts a local path or exact `r2://` object URI and defaults to
a `# SET ME` deployment constant in `cli/predict_capture.py` — until it is set,
every invocation must pass the flag. `--checkpoint-sha256` optionally pins the
local or downloaded bytes. The LightningModule class is detected from the
checkpoint's state dict (`--model-class {flow,ff}` overrides); `--stats-file`
applies the training
run's saved mel mean/std and **must** be passed when the served checkpoint was
trained with `use_saved_mean_and_variance`, or the model receives unnormalized
input (the CLI warns when it is omitted). `--map` overrides the packaged CLAP
param map, which otherwise follows `--param-spec-name`. Every run — crashes
included — appends to `<log-dir>/<input-stem>.log` (`--log-dir`, default set
per deployment next to the checkpoint constant).

## Regenerating the joint parameter maps

`src/synth_setter/data/vst/<spec>_param_map.json` stores Pedalboard, CLAP, and
DawDreamer identities plus version, parameter-count, preset-resource, and
preset-hash provenance. Capture each host with `build_param_map.py`'s
`dump-clap`, `dump-pedalboard`, and `dump-dawdreamer` commands, then join the
three files without loading a plugin runtime:

```bash
python -m synth_setter.tools.build_param_map build \
  --pedalboard-dump pedalboard.json --clap-dump clap.json \
  --dawdreamer-dump dawdreamer.json --param-spec-name surge_xt \
  --out src/synth_setter/data/vst/surge_xt_param_map.json
```

The join hard-fails on incomplete, ambiguous, duplicated, version-skewed, or
preset-skewed identities. Commit the regenerated map; completeness tests pin
all three supported Surge specs.
