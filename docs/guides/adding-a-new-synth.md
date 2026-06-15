# Guide: Adding a new VST3 synth

> **Source**: [`src/synth_setter/cli/introspect_plugin.py`](../../src/synth_setter/cli/introspect_plugin.py), [`src/synth_setter/data/vst/param_spec_registry.py`](../../src/synth_setter/data/vst/param_spec_registry.py)

______________________________________________________________________

## What you get

synth-setter's pipeline is synth-agnostic: rendering, storage, features,
distributed workers, and models all read parameter width and behavior from a
registered `ParamSpec` and `RenderConfig`, never from a synth literal (see
[architecture](../architecture.md)). Onboarding a new VST3 synth is therefore
**additive** — no edits to core pipeline, storage, or model code. A synth is
fully described by three registered artifacts:

| Artifact        | Where it lives                                   | Registry key                |
| --------------- | ------------------------------------------------ | --------------------------- |
| `ParamSpec`     | `src/synth_setter/data/vst/<name>_param_spec.py` | `param_specs["<name>"]`     |
| Baseline preset | `presets/<name>-base.vstpreset`                  | `preset_paths["<name>"]`    |
| `RenderConfig`  | `src/synth_setter/configs/render/<name>.yaml`    | selected by `render=<name>` |

All three are keyed by the synth name (`<name>`, a Python identifier) in
[`src/synth_setter/data/vst/param_spec_registry.py`](../../src/synth_setter/data/vst/param_spec_registry.py).
The preset filename convention is `<name>-base.vstpreset` for new registrations;
several existing `surge*` keys use shorter legacy names (e.g. `surge_xt` →
`presets/surge-base.vstpreset`) that the registry maps explicitly.

The one genuinely hard part is the `ParamSpec`: pedalboard can enumerate a
plugin's parameters, but raw names and 0–1 ranges carry **no semantics** — which
parameters matter, sensible sub-ranges, and categorical groupings all need
curation. The `synth-setter-introspect-plugin` tool scaffolds an editable draft
so you start from a working spec instead of a blank file.

## Prerequisites

- Project Python env (`make install`; see
  [getting-started](../getting-started.md)).
- The synth's `.vst3` bundle on disk. On Linux, run GUI-heavy plugins through
  the headless wrapper
  [`src/synth_setter/scripts/run-linux-vst-headless.sh`](../../src/synth_setter/scripts/run-linux-vst-headless.sh).
- Most Linux-precompiled VST3 synths are x86_64-only, so plan to render and
  validate on an amd64 host.

## Step 1 — Scaffold a draft spec

Run the introspection CLI against the bundle. It loads the plugin via
pedalboard, optionally applies a starting preset, classifies each parameter, and
emits a draft spec module, a captured baseline `.vstpreset`, and a per-parameter
CSV triage table.

```bash
synth-setter-introspect-plugin \
  --plugin-path /path/to/MySynth.vst3 \
  --spec-name mysynth
```

Useful flags (`synth-setter-introspect-plugin --help` for the full list):

- `--plugin-name` — factory class to open from a multi-class bundle (e.g.
  `'Six Sines'`); omit for single-class bundles.
- `--preset-path` — a starting `.vstpreset` to apply before capture, so the
  baseline reflects a sensible patch rather than the plugin's cold default.
- `--load-timeout` — seconds to wait for plugin init (default `600`);
  multi-minute loads are normal for some synths.
- `--out-spec` / `--out-preset` / `--out-csv` — override the default output
  paths (loose files in the cwd unless `--register` is given).
- `--force` — overwrite existing outputs; off by default so a re-run won't
  clobber a hand-tuned spec.

The CSV records the drafted outcome per parameter — read it to see what was
kept, pruned, or marked categorical.

## Step 2 — Hand-tune the spec

The draft is a starting point, not a finished spec. Open
`<name>_param_spec.py` and curate it using the parameter types in
[`src/synth_setter/data/vst/param_spec.py`](../../src/synth_setter/data/vst/param_spec.py):

- `ContinuousParameter(name, min, max, ...)` — a 0–1 host value sampled over a
  sub-range; narrow `min`/`max` to the musically useful band.
- `CategoricalParameter(name, values, raw_values, weights, encoding)` — discrete
  choices (waveform, filter type) with optional sample weights; `encoding`
  is `"scalar"` or `"onehot"`.
- `DiscreteLiteralParameter(name, min, max, encoding)` — an integer range.
- `NoteDurationParameter(name, max_note_duration_seconds)` — samples when the
  note starts and ends within the audio buffer (not an ADSR envelope); lives in
  the `note_params` list.

A `ParamSpec` takes two lists: `synth_params` (the synth's parameters) and
`note_params` (`pitch`, a `DiscreteLiteralParameter` whose MIDI window the
registered specs set to 48–72 — widen or narrow it for your synth — plus
`note_start_and_end`). Prune parameters that don't
affect the rendered tone (bypass, MIDI-routing, polyphony, glide) so the model
learns only meaningful dimensions. Curated widths vary widely across the
registered specs — from a 4-parameter toy spec to the full 162-parameter Surge
patch:

| Synth               | `synth_params` | encoded width |
| ------------------- | -------------- | ------------- |
| `surge_4` (fixture) | 4              | 7             |
| `surge_simple`      | 89             | 92            |
| `obxf`              | 94             | 187           |
| `surge_xt`          | 162            | 300           |

The encoded width (`len(param_specs[name])`, the `num_params` the shard writer
and models use) exceeds the curated count (`len(spec.synth_params)`) because
onehot-encoded categoricals expand one parameter into several dimensions, and
the note parameters add their own.
See [`surge_xt_param_spec.py`](../../src/synth_setter/data/vst/surge_xt_param_spec.py)
and [`obxf_param_spec.py`](../../src/synth_setter/data/vst/obxf_param_spec.py)
for hand-tuned examples.

## Step 3 — Register the synth

Wire the spec, preset, and render config into the checkout. The CLI does this
for you with `--register`:

```bash
synth-setter-introspect-plugin \
  --plugin-path /path/to/MySynth.vst3 \
  --spec-name mysynth \
  --register --verify
```

`--register` writes the spec module, preset, and CSV to their conventional
paths, generates `src/synth_setter/configs/render/mysynth.yaml`, and inserts the
import + `param_specs` + `preset_paths` entries into the registry. `--verify`
then runs the post-draft battery (pre-commit gates, registry import + sample,
Hydra compose, classifier audit), writes `verify-mysynth.md` at the checkout
root, and exits non-zero on any BLOCK. Read that report to see what to fix
before the synth is generation-ready.

If you prefer to register by hand (or are committing a hand-tuned spec on top of
an earlier draft), make these edits in
[`param_spec_registry.py`](../../src/synth_setter/data/vst/param_spec_registry.py):

```python
from synth_setter.data.vst.mysynth_param_spec import MYSYNTH_PARAM_SPEC

param_specs: dict[str, ParamSpec] = {
    # ...
    "mysynth": MYSYNTH_PARAM_SPEC,
}

preset_paths: dict[str, str] = {
    # ...
    "mysynth": "presets/mysynth-base.vstpreset",
}
```

The render config pins this synth's identity and inherits generic render knobs
(sample rate, cadence, batch size) from the base `surge_xt` render config
(`configs/render/surge_xt.yaml`):

```yaml
# src/synth_setter/configs/render/mysynth.yaml
defaults:
  - surge_xt

plugin_path: "plugins/MySynth.vst3"
preset_path: "presets/mysynth-base.vstpreset"
param_spec_name: "mysynth"
renderer_version: "1.2.3"
```

`renderer_version` is cross-checked against the loaded plugin before rendering,
so pin the exact version you onboarded against.

`--register` writes the output files and rewrites the registry module, so run
`make format` and commit before generating — the smoke run reads the committed
checkout.

## Step 4 — Generate a smoke dataset

With the synth registered, pass `render=<name>` to any generate-dataset
experiment. The `render=mysynth` override replaces the experiment's default
render group (e.g. `smoke-shard` defaults to `render=surge_simple`):

```bash
synth-setter-generate-dataset \
  experiment=generate_dataset/smoke-shard \
  render=mysynth \
  paths.output_dir=/path/to/output
```

This renders a small smoke dataset, proving the synth resolves through
`spec_from_cfg` and renders non-silent audio end-to-end. Scale up by pointing
`render=mysynth` at a larger experiment config.

## Optional — bake the synth into the Docker image

To run the synth in CI or on distributed workers, add a fetch step to the
`vst3-synths-fetch` stage in
[`docker/ubuntu22_04/Dockerfile`](../../docker/ubuntu22_04/Dockerfile): download
the release asset, pin its `sha256sum`, and unpack the `.vst3` into the staging
dir. The synths fetched there are x86_64-only, so each step early-exits on
non-amd64 builds. The build then runs a per-synth headless-X11 load check and
symlinks the bundle under `plugins/`. Dataset generation resolves the plugin
from the render config's `plugin_path`; `SYNTH_SETTER_PLUGIN_PATH` only sets the
default for tools that don't take a render config (tests, the interactive CLIs).

## See also

- [architecture](../architecture.md) — where the registry sits in the pipeline.
- [Surge XT interactive guide](surge-xt-interactive.md) — auditioning and
  capturing patches once a synth is registered.
- Epic [#1582](https://github.com/tinaudio/synth-setter/issues/1582) —
  multi-synth generalization, of which this guide is Phase 4.
