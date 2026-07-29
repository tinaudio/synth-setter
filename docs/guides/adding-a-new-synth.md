# Guide: Adding a new VST3 synth

> **Source**: [`src/synth_setter/cli/introspect_plugin.py`](../../src/synth_setter/cli/introspect_plugin.py), [`src/synth_setter/data/vst/param_spec_registry.py`](../../src/synth_setter/data/vst/param_spec_registry.py)

______________________________________________________________________

## What you get

synth-setter's pipeline is synth-agnostic: rendering, storage, features,
distributed workers, and models all read parameter width and behavior from a
registered `ParamSpec` and `RenderConfig`, never from a synth literal (see
[architecture](../architecture.md)). Onboarding a new VST3 synth is therefore
**additive** — no edits to core pipeline, storage, or model code. A synth is
fully described by four registered artifacts:

| Artifact        | Where it lives                                   | Registry key                                        |
| --------------- | ------------------------------------------------ | --------------------------------------------------- |
| Identity row    | `src/synth_setter/synth_spec.py`                 | `SYNTHS["<name>"]`                                  |
| `ParamSpec`     | `src/synth_setter/data/vst/<name>_param_spec.py` | `param_specs["<name>"]`                             |
| Baseline preset | `presets/<name>-base.vstpreset`                  | named by the identity row                           |
| `RenderConfig`  | `src/synth_setter/configs/render/<backend>.yaml` | selected by `render=vst` (or another backend group) |

The **identity row is the authoring point**: which param spec, which plugin, which
baseline preset. `plugin_state_paths` and the root identity group
`src/synth_setter/configs/synth/<name>.yaml` (selected via `synth=<name>`) are
projections of it, pinned against the table by
[`tests/test_synth_spec.py`](../../tests/test_synth_spec.py) and
[`tests/schemas/test_synth_config.py`](../../tests/schemas/test_synth_config.py). The `ParamSpec`
objects themselves live in
[`param_spec_registry.py`](../../src/synth_setter/data/vst/param_spec_registry.py).
The preset filename convention is `<name>-base.vstpreset` for new registrations;
several existing `surge*` keys use shorter legacy names (e.g. `surge_xt` →
`presets/surge-base.vstpreset`) that the registry maps explicitly.

This workflow is specifically for VST3 plugins. Checked-in Faust programs use a
registered source/spec pair instead: their `plugin_state_paths` entry is empty,
they render through the `dawdreamer_faust` backend (`render=faust`), and parameter names preserve the
exact addresses reported by Faust compilation.

The one genuinely hard part is the `ParamSpec`: pedalboard can enumerate a
plugin's parameters, but raw names and 0–1 ranges carry **no semantics** — which
parameters matter, sensible sub-ranges, and categorical groupings all need
curation. The `synth-setter-introspect-plugin` tool scaffolds an editable draft
so you start from a working spec instead of a blank file.

## Prerequisites

- Project Python env (`make install`; see
  [getting-started](../getting-started.md)).
- The synth installed through `make install-plugins` or another exact package
  entry in `studiorack.json`. Unmanaged/legacy `.vst3` paths remain supported.
  On Linux, run GUI-heavy plugins through the headless wrapper
  [`src/synth_setter/scripts/run-linux-vst-headless.sh`](../../src/synth_setter/scripts/run-linux-vst-headless.sh).
- Most Linux-precompiled VST3 synths are x86_64-only, so plan to render and
  validate on an amd64 host.

For direct programmatic rendering, synth-setter exposes the same abstract
dataclass contract for both hosts:

```python
from synth_setter.data.vst import DawDreamerRenderer

renderer = DawDreamerRenderer(
    plugin_path="plugins/MySynth.vst3",
    preset_path="presets/mysynth-base.vstpreset",
    sample_rate=44100,
    channels=2,
    signal_duration_seconds=4.0,
)
audio = renderer.render(
    params={"cutoff": 0.5},
    midi_note=60,
    velocity=100,
    note_start_and_end=(0.0, 2.0),
)
```

`PedalboardRenderer` has the same constructor and `render` contract.
`SurgePyRenderer` implements that contract specifically for Surge XT using an
`.fxp` patch and native Surge parameter identities; its validated configs use
the `surgepy` plugin sentinel, disable GUI toggling, and reload per render.
`TorchSynthRenderer` implements the contract for the in-process torchsynth backend
(`renderer_backend: torchsynth`, no `.vst3` bundle or preset — its param specs
are defined in `torchsynth_param_spec.py` rather than introspected, so the
steps below don't apply). DawDreamer
0.8.3 requires a CPython 3.12 render worker. Its published wheels cover
Linux x86_64, macOS x86_64/arm64, and Windows x86_64; Linux arm64 is not
supported. This requirement applies to the worker that renders audio.

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
- `--plugin-state-path` — a starting `.vstpreset` to apply before capture, so the
  baseline reflects a sensible patch rather than the plugin's cold default.
- `--load-timeout` — seconds to wait for plugin init (default `600`);
  multi-minute loads are normal for some synths.
- `--out-spec` / `--out-preset` / `--out-csv` — override the default output
  paths in loose-file mode; they cannot be combined with `--register`, which
  writes to fixed checkout paths.
- `--force` — overwrite existing outputs; off by default so a re-run won't
  clobber a hand-tuned spec.

The CSV records the drafted outcome per parameter — read it to see what was
kept, pruned, or marked categorical.

## Step 2 — Hand-tune the spec

The draft is a starting point, not a finished spec. Open
`<name>_param_spec.py` and curate it using the parameter types in
[`src/synth_setter/data/vst/param_spec.py`](../../src/synth_setter/data/vst/param_spec.py):

- `ContinuousParameter(name, min, max, ...)` — a finite renderer-native range
  encoded onto `[0, 1]` for the model; narrow `min`/`max` to the musically useful
  band.
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

| Synth                        | `synth_params` | encoded width |
| ---------------------------- | -------------- | ------------- |
| `surge_4` (4-param toy spec) | 4              | 7             |
| `surge_simple`               | 89             | 92            |
| `obxf`                       | 94             | 187           |
| `surge_xt`                   | 162            | 300           |

The encoded width (`param_specs[name].encoded_width`) exceeds the curated count
(`len(spec.synth_params)`) because onehot-encoded categoricals expand one
parameter into several dimensions, and the note parameters add their own. VST
model configs resolve this width from the root `synth` group's
`param_spec_name`; experiments must not repeat it as a `num_params`, `d_out`,
or `latent_dim` literal.
See [`surge_xt_param_spec.py`](../../src/synth_setter/data/vst/surge_xt_param_spec.py)
and [`obxf_param_spec.py`](../../src/synth_setter/data/vst/obxf_param_spec.py)
for hand-tuned examples.

## Step 3 — Register the synth

Wire the spec, preset, and identity config into the checkout. For a package
pinned in `studiorack.json`, use its package slug:

```bash
synth-setter-introspect-plugin \
  --studiorack-plugin organization/package \
  --spec-name mysynth \
  --register --verify
```

The package must already be installed. The command resolves it from the
checkout manifest and `STUDIORACK_PLUGINS_DIR`, creates a stable
`plugins/MySynth.vst3` alias, and records that portable path. The managed option
requires `--register` and is mutually exclusive with `--plugin-path`. Use
`--plugin-path /path/to/MySynth.vst3` instead for private, unmanaged, or legacy
bundles.

`--register` writes the spec module, preset, and CSV to their conventional
paths, generates `src/synth_setter/configs/synth/mysynth.yaml` (the root identity
group every run selects via `synth=mysynth`), adds the `SYNTHS` row, and
inserts the import + `param_specs` entry into the registry. `--verify`
then runs the post-draft battery (pre-commit gates, registry import + sample,
Hydra compose, classifier audit), writes `verify-mysynth.md` at the checkout
root, and exits non-zero on any BLOCK. Read that report to see what to fix
before the synth is generation-ready.

If you prefer to register by hand (or are committing a hand-tuned spec on top of
an earlier draft), make **two** Python edits. First the identity row in
[`synth_spec.py`](../../src/synth_setter/synth_spec.py), which `--register`
extends structurally:

```python
_synth_rows: dict[str, tuple[str, str, str, str]] = {
    # ...
    "mysynth": (
        "mysynth",
        "plugins/MySynth.vst3",
        "presets/mysynth-base.vstpreset",
        "1.2.3",
    ),
}
```

Then the spec itself in
[`param_spec_registry.py`](../../src/synth_setter/data/vst/param_spec_registry.py):

```python
from synth_setter.data.vst.mysynth_param_spec import MYSYNTH_PARAM_SPEC

param_specs: dict[str, ParamSpec] = {
    # ...
    "mysynth": MYSYNTH_PARAM_SPEC,
}
```

Skipping the identity row is the common mistake: `--verify` will pass, then
`tests/test_synth_spec.py` fails in CI because `SYNTHS` has no entry.

The root synth group is a generated projection of that row and owns all
identity fields, including the required artifact version; VST datamodules,
models, and callbacks resolve it through `${synth.param_spec_name}`:

```yaml
# src/synth_setter/configs/synth/mysynth.yaml
name: "mysynth"
param_spec_name: "mysynth"
plugin_path: "plugins/MySynth.vst3"
plugin_state_path: "presets/mysynth-base.vstpreset"
synth_version: "1.2.3"
```

Render groups are backend-named, not per-synth: a registered VST3 synth uses
the generic `vst` render base (`src/synth_setter/configs/render/vst.yaml`)
directly via `render=vst`, so no render config is generated or needed.

`synth.synth_version` is cross-checked against the loaded plugin before
rendering, so pin the exact version you onboarded against.

`--register` writes the output files and rewrites the registry module, so run
`make format` and commit before generating — the smoke run reads the committed
checkout. Faust source identities are registered manually with an empty state
entry and rendered via `render=faust` (the `dawdreamer_faust` backend group);
the VST3 introspection command does not generate them.

## Step 4 — Generate a smoke dataset

With the synth registered, pass `synth=<name> render=vst` to any
generate-dataset experiment. The override replaces the experiment's default
identity (e.g. `smoke-shard` defaults to `synth=surge_simple`):

```bash
synth-setter-generate-dataset \
  experiment=generate_dataset/smoke-shard \
  synth=mysynth render=vst \
  paths.output_dir=/path/to/output
```

This renders a small smoke dataset, proving the synth resolves through
`spec_from_cfg` and renders non-silent audio end-to-end. Scale up by pointing
`synth=mysynth render=vst` at a larger experiment config.

## Optional — bake the synth into the Docker image

To run the synth in CI or on distributed workers, add its exact package version
and VST3 bundle basename to `studiorack.json`. Include the package in the
`builder-install-studiorack-plugins` selection and the Docker presence/load
checks. Keep source-build fallbacks only when the registry has no compatible
artifact for a supported image platform. Dataset generation resolves the stable
alias from the synth identity; `SYNTH_SETTER_PLUGIN_PATH` only sets the default
for tests and interactive tools that do not compose a synth config.

## See also

- [architecture](../architecture.md) — where the registry sits in the pipeline.
- [VST interactive guide](vst-interactive.md) — auditioning and
  capturing patches once a synth is registered.
- Epic [#1582](https://github.com/tinaudio/synth-setter/issues/1582) —
  multi-synth generalization, of which this guide is Phase 4.
