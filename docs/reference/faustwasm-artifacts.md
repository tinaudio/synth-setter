# FaustWasm offline artifacts

Install the exact Node runtime before using the backend:

```bash
npm ci
```

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
- `parameters` contains `canonicalAddress`, `wasmAddress`, `min`, `max`, and `kind` for every
  sampled control. Mapping completeness and native domains are checked against compiled metadata.
- `files.dsp`, and `files.mixer` / `files.effect` when present, each contain a relative `path` and
  SHA-256 digest. `dspMeta` and optional `effectMeta` carry the Faust factory JSON needed to load
  those modules without recompilation.

The public `runtime.mjs` exports `loadFaustArtifact`, `createOfflineSynth`,
`applyCanonicalPatch`, and `renderNote`. Browser and Node consumers share this contract. Canonical
addresses remain the dataset identity even where compiler hosts differ, including apostrophe
normalization and the church-organ wet/dry label.

The Python renderer currently resolves these Node assets from a synth-setter checkout. An installed
wheel fails with an actionable message rather than assuming `scripts/` is present.
