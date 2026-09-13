# pyFDN sketch flow evaluation in the browser

This playbook runs the whole pyFDN sketch-flow evaluation on a static web page: upload an
impulse response, choose the conditioning mode and guidance strengths, and get predicted
parameters, a FaustWasm render of them, audio playback, and impulse-response metrics. After the
bundle is exported nothing runs in Python.

```text
WAV upload → OfflineAudioContext decode (44.1 kHz mono, 4 s)
→ frontend.onnx (librosa-parity mel + training statistics)
→ sketch.mjs (pyfdn_reverb controls, 10 × 32)
→ conditioning.onnx + RK4 over velocity.onnx with branch weights
→ decode.mjs → FaustWasm fdnHouseholder render (176 400 frames)
→ metrics.mjs (MSS, octave EDC RMSE, octave RT60 log-RMSE, T30 percentage error, C50 MAE)
```

The page lives in `src/synth_setter/web/fdn/`; the JavaScript ports it relies on are described
in [browser pyFDN ports](../reference/browser-fdn-ports.md).

## 1. Prerequisites

```bash
uv sync --frozen --python 3.12.13 --extra cpu --no-default-groups --group dev
npm ci --prefix src/synth_setter/web
(cd src/synth_setter/web && npx playwright install chromium)
```

Node 18 or newer is required by the FaustWasm exporter; the e2e driver needs the Playwright
Chromium build.

## 2. Export the model bundle

The bundle is produced from a `pyfdn/flow_sketch` checkpoint and its training mel statistics by
`synth-setter-export-browser-fdn-bundle` (see the [CLI reference](../reference/cli.md)). The
validated export of the 2026-09-08 `flow_sketch` run is content-addressed by the SHA-256 of its
`manifest.json` and stored under `r2:experiments/browser-fdn/<sha256>`; the digest every
browser FDN workflow uses is pinned in `src/synth_setter/web/fdn/BUNDLE_SHA256`:

```bash
BUNDLE_SHA256=$(cat src/synth_setter/web/fdn/BUNDLE_SHA256)
rclone copy --checksum "r2:experiments/browser-fdn/${BUNDLE_SHA256}" build/fdn-bundle
echo "${BUNDLE_SHA256}  build/fdn-bundle/manifest.json" | sha256sum --check
```

The bundle holds `frontend.onnx`, `conditioning.onnx`, and `velocity.onnx` (about 162 MB in
float32) plus the manifest with the parameter spec, sketch contract, sampling defaults, and
per-file digests.

## 3. Export the Faust FDN artifact

The renderer is the registered `faust_fdn_n8_mono_householder` synth, which shares the
`pyfdn_n8_mono_householder` parameter spec and designs the decay shelves inside the DSP
([FaustWasm artifacts](../reference/faustwasm-artifacts.md)):

```bash
uv run python -m synth_setter.tools.export_faustwasm \
  --synth faust_fdn_n8_mono_householder --output build/fdn-faust
```

## 4. Assemble and serve the site

```bash
node src/synth_setter/web/fdn/export-site.mjs \
  --model build/fdn-bundle --faust build/fdn-faust --output build/fdn-site
uv run python -m http.server --directory build/fdn-site 8000
```

Open `http://localhost:8000`, which redirects to `fdn/index.html`. The page reports `Ready`
once both manifests and every digest have been verified. Choose a WAV impulse response (any
rate or channel count; it is resampled and downmixed to the checkpoint grid), a conditioning
mode, the content and sketch CFG strengths, the number of integration steps, and a noise seed,
then run the evaluation.

| Mode           | Branch weights over `[unconditional, sketch_only, content_only, full]`  |
| -------------- | ----------------------------------------------------------------------- |
| Mel and sketch | `[1 − s, s − c, 0, c]`, identical to the training-time three-branch CFG |
| Mel only       | `[1 − c, 0, c, 0]`                                                      |
| Sketch only    | `[1 − s, s, 0, 0]`                                                      |
| Unconditional  | `[1, 0, 0, 0]`                                                          |

The checkpoint's own defaults are 200 steps with both strengths at 2; the page starts at 50
steps so the first run finishes in seconds on the WASM backend. Every run record (noise, weights,
parameters, sketch, target, prediction, metrics) is exposed on `window.fdnEval` for scripting.

## 5. Run the parity test locally

`tests/integration/test_browser_fdn_e2e.py` renders a spec-sampled pyFDN response, exports the
site, drives it in headless Chromium, and checks the page's decoded target, sketch, metrics, and
FaustWasm render against the Python references. With the checkpoint and statistics available it
also replays the browser's noise through `VSTFlowMatchingModule.sample_batch`:

```bash
export BROWSER_FDN_MODEL_BUNDLE=$PWD/build/fdn-bundle
export BROWSER_FDN_FAUST_ARTIFACT=$PWD/build/fdn-faust
export SYNTH_SETTER_FDN_SKETCH_CHECKPOINT=/path/to/model.ckpt
export SYNTH_SETTER_FDN_SKETCH_STATS=/path/to/stats.npz
uv run pytest tests/integration/test_browser_fdn_e2e.py -v
```

The checkpoint and statistics variables are optional; without them the sampling-parity step is
skipped and the render, sketch, and metric checks still run.

## 6. CI

`Browser pyFDN sketch E2E` (`.github/workflows/browser-fdn-e2e.yml`) runs on trusted pull
requests touching the page, the FaustWasm stack, or the flow export, and on `workflow_dispatch`.
It fetches the pinned bundle from R2, exports the Faust artifact from source, runs the node
suite, executes the e2e test, and fails if that test was skipped. Screenshots, run records, and
the target WAV are uploaded as the `browser-fdn-e2e-<run id>` artifact.

`Browser pyFDN site` (`.github/workflows/browser-fdn-site.yml`) builds the static site on
demand and uploads it as the `browser-fdn-site` artifact without the `model/` directory: the
graphs are a full export of a private checkpoint and must not enter an artifact of this public
repository. To serve the artifact, unzip it and copy the fetched bundle in as `model/`.
Publishing to a public host is not automated: the repository's GitHub Pages deployment belongs
to the documentation site.

## Troubleshooting

- **`Error: Could not load model/manifest.json`** — the site was served from a directory other
  than the exporter's output; serve `build/fdn-site`, not `build/fdn-site/fdn`.
- **`model bundle digest mismatch`** — the bundle directory was edited after export; re-fetch it.
- **`FaustWasm version mismatch`** — the artifact was compiled by a different FaustWasm than the
  vendored runtime; re-export it with the current checkout.
- **Playwright cannot launch Chromium** — run `npx playwright install --with-deps chromium`
  inside `src/synth_setter/web`.
