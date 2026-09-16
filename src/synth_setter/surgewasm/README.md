# Surge XT browser host

This directory builds and hosts the unmodified Surge XT WASM engine from commit
`dd68c74346c828ef25bd6504867936648161b7a7` (1.4 development version). The
bundle contains `surge-host.mjs`, `surge-host.wasm`, the unchanged
`surge-xt.clap.wasm`, a digest manifest, the GPL license, and source/build
instructions.

## Build

Activate emsdk 6.0.9 and make CMake available, then use either an existing clean
checkout or a dedicated clone cache:

```sh
node src/synth_setter/surgewasm/build.mjs \
  --source /path/to/surge \
  --output /path/to/absent/output

node src/synth_setter/surgewasm/build.mjs \
  --cache /path/to/surge-cache \
  --output /path/to/absent/output
```

The build configures the upstream tree without patches, builds
`surge-wasm-demo` with at most four parallel jobs, compiles `host.c` as C, and
links it with `em++`. The output path must not exist and appears only after the
complete bundle is ready.

## Runtime API

Copy `runtime.mjs` beside the generated bundle and call:

```js
const result = await renderSurge({
    root: new URL('../engine/', import.meta.url).href,
    preset,
    parameters: [{ id: 308, name: 'A Filter 1 Cutoff', value: 0.5 }],
    note: 60,
    velocity: 100,
    noteStart: 0,
    noteEnd: 0.5,
    sampleRate: 44100,
    frames: 44100,
});
```

`preset` is a caller-digest-verified `Uint8Array` containing an FXP chunk.
Parameter IDs are Surge synth-side IDs, values are normalized to `[0, 1]`, and
note start/end values are seconds. Only supplied parameters are checked against
the live host. Names may match the wrapper's `getParameterNameExtendedByFXGroup`
names or SurgePy's unextended FX names: the adapter removes only the CLAP-added
FX group decoration before comparison. IDs, FX slots, and parameter names still
must match. These are synth-side IDs, not the hashed IDs of the regular CLAP wrapper.
The result is `{left, right, version, engineCommit, parameterCount}` with two
trimmed `Float32Array` channels.

The runtime floors note-on and rounds note-off upward to 32-frame process
boundaries because the upstream demo host queues events at the start of a
process call. Equal endpoints remain an on/off pair in one block. Native SurgePy
additionally shifts the rendered attack to the requested sample; this adapter
does not, so sample-level timing parity is not claimed. Node integration
tests may pass the optional `loader` Emscripten factory because Node cannot
dynamically import an HTTP module URL. Browser callers omit it, and the runtime
imports `${root}/surge-host.mjs` directly.

## Batch rendering for host comparisons

```sh
node src/synth_setter/surgewasm/render.mjs \
  /path/to/bundle presets/surge-base.fxp requests.json /path/to/absent/output
```

`requests.json` is a nonempty array of runtime request objects, without `root` or
`preset`. Every row uses the supplied FXP. The CLI verifies a private bundle
snapshot before loading its host, renders real stereo float WAVs, and publishes the output
directory atomically. `report.json` records engine/preset provenance and requested
parameter descriptors—not parameter readback—beside `sample_00.wav`, etc.

Run the real-WASM test against a completed bundle:

```sh
SURGE_WASM_BUNDLE=/path/to/bundle \
  node --test src/synth_setter/surgewasm/test-runtime.mjs
```
