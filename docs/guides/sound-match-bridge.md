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

`--checkpoint` defaults to a `# SET ME` deployment constant in
`cli/predict_capture.py` — until it is set, every invocation must pass the
flag. `--model-class {flow,ff}` selects the LightningModule; `--stats-file`
applies the training run's saved mel mean/std and **must** be passed when the
served checkpoint was trained with `use_saved_mean_and_variance`, or the model
receives unnormalized input (the CLI warns when it is omitted). `--map`
overrides the packaged CLAP param map.

## Regenerating the CLAP param map

`src/synth_setter/data/vst/surge_xt_clap_map.json` (packaged, the CLI's `--map`
default) maps every `SURGE_XT_PARAM_SPEC` pyname to its CLAP id/name/range. It
is built from `surge_xt_clap_info.json`, a raw dump of the installed Surge XT
CLAP taken by the first-party ctypes host in `data/vst/clap_introspect.py`.
After a Surge upgrade:

```bash
# 1. Dump the installed CLAP (no display needed)
python -m synth_setter.tools.build_clap_map dump

# 2. Rebuild the map — loads the VST3 via pedalboard, so on Linux run under
#    the headless wrapper
src/synth_setter/scripts/run-linux-vst-headless.sh \
  .venv/bin/python -m synth_setter.tools.build_clap_map build
```

`build` joins the dump with pedalboard's base-preset view through the
patch-invariant parameter index, re-validates that premise elementwise against
init-state names and `surge_params.csv`, and fails loudly listing every
unmapped parameter. Commit both JSONs; the completeness tests in
`tests/data/vst/test_clap_map_completeness.py` pin the result.
