# Browser-native Surge sketch evaluation

The static Surge page runs music feature extraction, ONNX flow inference, and
Surge audio rendering in the browser. Exporting the model and building the
engine require native tools; serving the exported page does not run Python
inference or native SurgePy.

## Engine boundary

The engine is the unchanged Surge proof-of-concept commit
`dd68c74346c828ef25bd6504867936648161b7a7` (1.4.0 development), not the
SurgePy 1.3.4 release used for training. The separate synth-setter host supplies
the C++ runtime, patch loading, and lifecycle management. It does not patch
the engine or its dependency pins. The bundled upstream demo's missing C++
runtime is tracked in #3542.

Cross-version audio differences are measurements, not a parity failure or
proof of model quality. A 1.3.4 backport follows successful 1.4 integration;
it is not part of this build.

## Build and export

Use an isolated synth-setter worktree with `uv sync`. Install and activate
Emscripten **6.0.9**, and make CMake available on `PATH`.

```bash
npm ci --prefix src/synth_setter/web
node src/synth_setter/surgewasm/build.mjs \
  --cache /tmp/surge-wasm-source --output build/surge-engine
uv run python -m synth_setter.tools.export_browser_surge_bundle \
  --output build/surge-model
node src/synth_setter/web/surge/export-site.mjs \
  --model build/surge-model --engine build/surge-engine \
  --output build/surge-site
python3 -m http.server 8765 --bind 127.0.0.1 --directory build/surge-site
```

Open <http://127.0.0.1:8765/surge/index.html>. Export destinations must not
already exist. The exporter uses the content-addressed real checkpoint and
statistics in `configs/browser_surge.yaml` by default, avoiding mutable
training `last.ckpt` objects. Native render settings still come from
`configs/sketch_render.yaml`; explicit
`--checkpoint`, `--checkpoint-sha256`, `--stats`, and `--stats-sha256`
overrides are also supported. Only the `surge_simple` music model contract
is accepted. Export trusts the selected checkpoint: do not load untrusted
PyTorch checkpoint files.

The engine bundle includes its license and source/build instructions. Keep
those with redistributed artifacts and provide corresponding source as
required by Surge's GPL license.

## Evaluate

1. Upload content/target audio.
2. Upload separate sketch audio, or author a fixed MIDI-pitch contour with
   loudness, centroid, and onset/end controls.
3. Choose both, content-only, sketch-only, or unconditional guidance; set
   CFG strengths, RK4 steps, and seed.
4. Run evaluation, inspect the predicted parameters and controls, and play
   or download the stereo prediction and target.

Inputs are converted to stereo 44.1 kHz, padded/truncated to four seconds,
and clipped to `[-1, 1]`, matching the native sketch loader's amplitude
contract (including resampling overshoot). Sampling controls initially use
the exported checkpoint's defaults.
The model consumes stereo mel features and 386 music controls on 32 frames.
Authored pitch controls are explicit activation contours, not PESTO estimates
from a recorded sound.

The page reports training-statistics-normalized mel MAE, waveform RMSE, RMS
levels, and predicted peak. Waveform error is phase-sensitive. These are not
FDN reverb metrics, the complete native sketch metric suite, or perceptual
quality scores. The downloaded JSON retains model provenance and source
revision, a unique run ID and UTC completion time, noise, features,
predictions, decoded patch, renderer version, audio, and metrics.

## Tailnet preview

Serve only the exported site directory, not the repository or checkpoint cache.
Run the HTTP server in a persistent terminal, then add an unused HTTPS port with
Tailscale Serve:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory build/surge-site
```

```bash
tailscale serve status
tailscale serve --bg --https=10000 http://127.0.0.1:8766
```

Open the printed Tailnet URL with `/surge/index.html` appended. HTTPS supplies
the secure context used for artifact checksums. Preserve existing FDN routes;
do not use Funnel. The preview remains available while the host and HTTP server
are running.

## Real verification

```bash
cd src/synth_setter/web && npx playwright install chromium && cd ../../..
SYNTH_SETTER_SURGE_WASM_BUNDLE="$PWD/build/surge-engine" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run pytest tests/integration/test_browser_surge_wasm_e2e.py -v \
  --basetemp=logs/browser-surge-wasm --junitxml=logs/browser-surge-wasm.xml
```

The test uses real SurgePy input audio, the configured trained checkpoint,
real Chromium/ONNX Runtime Web, and the compiled Surge WASM engine. It checks
browser features and predictions against native computations, consumes the
predicted patch, and retains browser/native audio and a cross-version
comparison. A second run clears the sketch upload, authors a note contour,
and checks sketch-only predictions and both downloaded artifacts.
R2 access, native SurgePy, the engine bundle, and Chromium are
required; the production-path CI lane rejects a skipped E2E.

The workflow is `.github/workflows/browser-surge-wasm-e2e.yml`. Tracking:
#3538, stacked above the FDN browser effort #3483.
