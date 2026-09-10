# FaustWasm offline artifacts

Install the exact Node runtime, then export a registry-backed artifact to a new directory:

```bash
npm ci
uv run python -m synth_setter.tools.export_faustwasm \
  --synth faust_bright_organ \
  --output build/faust-bright-organ
```

The destination must not exist. Compilation uses a sibling staging directory and publishes the
completed artifact atomically, so a failed export never leaves the requested destination looking
successful. Python callers can use
`export_faustwasm_artifact(SynthName("faust_bright_organ"), Path("build/faust-bright-organ"))`.
The compiler derives the native output count from Faust metadata; renderer-owned compilation also
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

Select `render=faustwasm` for bright organ, bubble, or church organ, and
`render=faustwasm_filter_osc` for the mono filter oscillator. Pair either render group with the
existing `synth=faust_*` identity. FaustWasm requires render contract version 2; legacy version 1
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

The Python renderer and exporter resolve these Node assets from a synth-setter checkout. An
installed wheel fails with an actionable message rather than assuming `scripts/` is present.
