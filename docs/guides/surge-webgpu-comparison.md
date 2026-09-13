# Compare Surge neural inference backends

Start with the [browser-native Surge setup](surge-wasm-sketch-evaluation.md).
The exported site includes the pinned ONNX Runtime WebGPU runtime and its
JSEP assets; no separate server-side inference service is needed.

## A/B comparison

1. Upload content and sketch audio. Leave **WebGPU neural inference** unchecked
   for the WASM baseline.
2. Run evaluation and download `evaluation-wasm.json` and the prediction WAV.
3. Check **WebGPU neural inference**, retaining the same inputs, seed, CFG and
   step count, then run again. Download `evaluation-webgpu.json` and its WAV.
4. Read the previous-run parameter delta and neural timings. The comparison
   appears only when checkpoint/statistics, sampling settings, noise, mel and
   sketch features match exactly. Changing the seed or inputs suppresses it.

Only the conditioning and velocity networks switch providers. Mel extraction,
PESTO, JavaScript RK4 integration and Surge synthesis stay on their existing
paths. WebGPU sessions explicitly disable CPU execution-provider fallback;
unsupported hardware or graph operations produce an error. Uncheck the box to
run on WASM again. Sessions are cached separately so switching back does not
accidentally reuse the GPU provider.

The download records the backend, GPU adapter information, neural elapsed time
and prior matching run ID. Neural timing excludes model loading, feature
extraction and synthesis, but can include first-run shader compilation. Treat
single-run timings as diagnostics, not a benchmark or speed guarantee.

PESTO's near-silence PyTorch/ONNX discrepancy is tracked in
[#3550](https://github.com/tinaudio/synth-setter/issues/3550). Keeping feature
extraction on WASM isolates it from this neural-backend comparison. Neither a
small parameter delta nor similar audio establishes native1.3/WASM1.4 sonic
parity; oscillator state can also change rendered audio between runs.

The UI accepts content/sketch CFG strengths up to **200** and RK4 runs up to
**20,000 steps**, without changing checkpoint defaults. High guidance can
produce unstable results; long runs can take substantially more time. Boundary
coverage verifies that a real 20,000-step model run starts, then closes the
browser; it does not claim completion of that full neural run. A separate
analytic-field test completes all 20,000 RK4 steps.

## Audio and spectrograms

Results include players, WAV downloads and spectrograms for **Target**, **Sketch**
and **Prediction**. Authored contours have no source sketch recording, so that
mode explicitly omits the sketch player/spectrogram rather than synthesizing a
surrogate.

All three plots use a shared −100 to 0 dBFS scale, a periodic 2048-sample Hann
window, a 512-sample hop and a logarithmic 20 Hz–Nyquist display. Channel powers
are averaged instead of downmixing waveforms, preserving anti-phase stereo
energy. These are display STFTs, not the model's normalized mel features or the
separately labelled music-control heatmap. Predicted synth parameters are
collapsible to keep playback and plots accessible.

## Real-browser verification

Supply an exported site, real input WAVs and the worktree's Python/SoundFile
environment. Run from the worktree root:

```bash
SURGE_BROWSER_URL=http://127.0.0.1:8768/surge/index.html \
SURGE_BROWSER_CONTENT=/absolute/content.wav \
SURGE_BROWSER_SKETCH=/absolute/sketch.wav \
SURGE_BROWSER_OUTPUT=logs/surge-browser-options \
node --test src/synth_setter/web/surge/browser-options.e2e.mjs
```

The test toggles WASM → WebGPU → WASM, compares actual downloaded records,
consumes prediction WAVs through SoundFile, plays sketch audio, checks the three
plots, suppresses mismatched-seed comparisons, and verifies explicit failure
and subsequent WASM recovery with WebGPU disabled.

Headless Chromium may require local driver flags, supplied as a JSON array in
`SURGE_BROWSER_GPU_ARGS`. For this NVIDIA/Vulkan setup:

```bash
export SURGE_BROWSER_GPU_ARGS='["--enable-unsafe-webgpu","--enable-features=Vulkan","--use-angle=vulkan","--use-vulkan=native"]'
```

Always inspect recorded adapter information. SwiftShader proves software
WebGPU compatibility only, not hardware acceleration. The default browser
checkbox remains unchecked regardless of the test flags.
