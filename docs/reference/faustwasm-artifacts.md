# FaustWasm offline artifacts

Install Node.js 18 or newer, then export a registry-backed artifact to a new directory:

```bash
uv run python -m synth_setter.tools.export_faustwasm \
  --synth faust_bright_organ \
  --output build/faust-bright-organ
```

The destination must not exist. Compilation uses a sibling staging directory and publishes the
completed artifact atomically, so a failed export never leaves the requested destination looking
successful. Python callers can pass `extract_backend_version("faustwasm")` as the
`backend_version` argument to `export_faustwasm_artifact`. The compiler derives the native output
count from Faust metadata; renderer-owned compilation also
checks that count against its configured channels.

To bundle the artifact with the browser player and serve it locally:

```bash
cd scripts/faustwasm/browser
npm ci
node export-browser.mjs \
  --artifact ../../../build/faust-bright-organ \
  --output ../../../build/faust-bright-organ-site
cd ../../..
uv run python -m http.server --directory build/faust-bright-organ-site 8000
```

Open `http://localhost:8000`. The browser exporter and player are supplied by the stacked browser
entrypoint; they consume the persisted manifest and Wasm files without compiling Faust source.

Select `render=faustwasm` for bright organ, bubble, or church organ,
`render=faustwasm_filter_osc` for the mono filter oscillator, and `render=faustwasm_fdn` for the
mono feedback delay network. Pair each render group with the existing `synth=faust_*` identity.

`faust_fdn_n8_mono_householder` renders the `pyfdn_n8_mono_householder` parameter spec verbatim: the
Faust identity resolves the same spec object, so a pyFDN row decodes to identical native values on
both backends. Array-valued fields expand to one compiled slider per element, named by the spec's
native coordinate labels (`delays.0` … `delays.7`, `input_matrix.3.0`, `output_matrix.0.7`,
`direct_matrix.0.0`); the spec-derived `feedback_matrix` is compiled into the source as the fixed
Householder reflection of the all-ones vector, so a patch must carry it and it must match. The decay shelves are
designed inside the DSP from `post_delay.rt_dc_seconds` and `post_delay.rt_nyquist_seconds` with
pyFDN's 6 kHz crossover. Only FaustWasm flattens these fields; selecting `render=faust` or the
Faust C++ backend with this identity fails validation. The impulse is generated in-DSP at the first
sample of each fresh instance, so a mono `renderNote` capture of 176 400 frames at 44.1 kHz is the
complete four-second response; parity against `pyFDN.build_to_impz` holds to `atol=1e-5`
(`tests/data/vst/test_faustwasm_pyfdn_parity.py`). FaustWasm requires render contract version 2; legacy version 1
omits backend provenance and is rejected.

## Manifest schema

`export-artifacts.mjs` writes `manifest.json` with `schemaVersion: 1` and these required fields:

- `identity`, `faustwasmVersion`, `libfaustVersion`, `compileOptions`, and `sourceSha256` pin source
  and compiler provenance.
- `mode` is `mono` or `poly`; `voices` is zero for mono and the voice pool size for poly;
  `outputs` is the native channel count.
- `parameters` contains `canonicalAddress`, `wasmAddress`, `min`, `max`, `kind`, and exact discrete
  `values` for every sampled control. Mapping completeness and native domains are checked against
  compiled metadata.
- `files.dsp`, and `files.mixer` / `files.effect` when present, each contain a relative `path` and
  SHA-256 digest. `dspMeta` and optional `effectMeta` carry the Faust factory JSON needed to load
  those modules without recompilation.

The public `runtime.mjs` exports `loadFaustArtifact`, `createOfflineSynth`,
`applyCanonicalPatch`, and `renderNote`. Browser and Node consumers share this contract. Canonical
addresses remain the dataset identity even where compiler hosts differ, including apostrophe
normalization and the church-organ wet/dry label.

The Python wheel includes the pinned FaustWasm compiler, runtime, and Node entrypoints. Rendering
and export therefore work outside a repository checkout without running root-level `npm ci`.
