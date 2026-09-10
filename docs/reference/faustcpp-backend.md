# Native Faust C++ backend

Select `render=faustcpp` for the stereo Faust identities or
`render=faustcpp_filter_osc` for the mono filter oscillator. Pair the render group with a
matching `synth=faust_*` group.

The backend requires the `faust` CLI and `g++` on `PATH`. `RenderConfig.backend_version` must
match `faust --version`; the Ubuntu 22.04 container installs the distro Faust package and the
committed render group pins that package's upstream version. `block_size` controls native offline
processing and participates in the v2 render-contract digest.

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
  tests/data/vst/test_faust_dataset_e2e.py::test_faustcpp_generate_cli_writes_real_lance_row -v
```

The E2E test drives `build_generate_args()` through the real generate-dataset subprocess, Faust
compiler, C++ compiler, native renderer, mel/MP3 transforms, and Lance writer. It then reads the
row with Lance and checks its shapes, dtypes, finiteness, and non-silence.

## DawDreamer comparison

A local AB/BA benchmark ran five trials per backend with 16 rows per trial. Both backends used the
bright-organ source, seed 1808, 44.1 kHz stereo four-second audio, velocity 100, one-row render
batches, per-render DSP isolation, no loudness floor, and block size 2048. Every accepted
`param_array` was byte-identical across each backend pair. Timings include renderer construction
and compilation, all renders, mel/MP3 transforms, and the Lance commit.

| Backend          | Version | Median wall time | Wall-time IQR | Median throughput | Throughput IQR |
| ---------------- | ------- | ---------------: | ------------: | ----------------: | -------------: |
| DawDreamer Faust | 0.8.3   |          8.166 s |       0.098 s |      1.959 rows/s |   0.023 rows/s |
| Native Faust C++ | 2.70.3  |          5.584 s |       0.101 s |      2.865 rows/s |   0.052 rows/s |

On an Intel Core i9-14900KF running Linux 6.17, native C++ completed the full generation path
1.46× faster, reducing median wall time by 31.6%. These figures characterize this machine and
small-shard workload; run the same production-path comparison on target worker hardware before
using the ratio for capacity planning.
