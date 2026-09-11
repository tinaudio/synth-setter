# Native Faust C++ backend

Select `render=faustcpp` for the stereo Faust identities or
`render=faustcpp_filter_osc` for the mono filter oscillator. Pair the render group with a
matching `synth=faust_*` group.

The backend requires the `faust` CLI and `g++` on `PATH`. `RenderConfig.backend_version` must
match `faust --version`; the Ubuntu 22.04 production and parity environments use Faust 2.37.3
with g++ 12. The renderer resolves canonical parameter addresses across the space-versus-underscore
MapUI formatting used by supported Faust versions. `block_size` controls native offline processing
and participates in the v2 render-contract digest.

`FaustCppRenderer` verifies the registered source digest, generates C++ with the packaged
architecture, and compiles one executable per shard renderer. Each row invokes that executable,
which creates a fresh DSP instance, applies the complete canonical patch through Faust `MapUI`,
renders exact note boundaries, and writes channel-major float32 audio. Compilation is therefore
amortized while oscillator, delay, filter, and voice state remain isolated between rows.

The implementation follows the compile pipeline demonstrated by
[`sletz/faust-mcp`](https://github.com/sletz/faust-mcp): Faust source → custom C++ architecture →
optimized native executable. No source was copied because that repository does not declare a
license. The synth-setter architecture emits audio rather than analysis JSON and preserves the
existing `AudioRenderer` contract.

## Production-path verification

```bash
uv run pytest \
  tests/data/vst/test_faustcpp_renderer.py \
  tests/test_generate_dataset.py::test_generate_dataset_faustcpp_writes_real_lance_row -v
```

The E2E test drives `build_generate_args()` through the real generate-dataset subprocess, Faust
compiler, C++ compiler, native renderer, mel/MP3 transforms, and Lance writer. It then reads the
row with Lance and checks its shapes, dtypes, finiteness, and non-silence.

## Three-backend dataset benchmark

Run the reproducible benchmark with:

```bash
synth-setter-benchmark-faust-backends \
  --output /tmp/faust-backends-16x5 \
  --rows 16 \
  --trials 5 \
  --seed 1808 \
  --block-size 2048
```

The benchmark builds one deterministic bounded bright-organ corpus, then passes those exact synth
patches and MIDI events to every host. Five rotated forward/reverse trials reduce execution-order
bias. Each timing covers renderer construction and compilation, 16 four-second 44.1 kHz stereo
renders, mel/MP3 transforms, and the Lance commit. The runner consumes every Lance dataset,
requires byte-identical normalized parameter rows across all 15 runs, and writes raw timings,
backend versions, the Git revision and dirty state, host details, corpus digest, medians, and
interquartile ranges to `results.json`.

A local run on an Intel Core i9-14900KF running Linux 6.17 produced:

| Backend          | Version | Median wall time | Wall-time IQR | Median throughput | Throughput IQR |
| ---------------- | ------- | ---------------: | ------------: | ----------------: | -------------: |
| DawDreamer Faust | 0.8.3   |          4.995 s |       0.042 s |      3.203 rows/s |   0.027 rows/s |
| Native Faust C++ | 2.70.3  |          3.036 s |       0.117 s |      5.269 rows/s |   0.200 rows/s |
| FaustWasm        | 0.18.3  |          2.557 s |       0.174 s |      6.257 rows/s |   0.443 rows/s |

Native C++ was 1.65× faster than DawDreamer with 39.2% lower median wall time. FaustWasm was 1.19×
faster than native C++ on this workload. These figures characterize one machine, the bounded
benchmark corpus, and a small shard. The committed production config pins Faust 2.37.3 rather than
the locally available 2.70.3, so benchmark target worker images before capacity planning.
