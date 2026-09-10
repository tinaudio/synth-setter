# SurgePy flow evaluation in the browser

This playbook runs the complete **evaluation** path from a real trained
checkpoint. It does not train a new model or establish sound-matching quality.

```text
real SurgePy/audio file → normalized mel + sketch controls
→ trained checkpoint → conditioning.onnx + velocity.onnx
→ Chromium / ONNX Runtime Web WASM → RK4/CFG prediction
→ native SurgePy → predicted WAV + parameter CSV + audio metrics
```

The browser runs the learned encoder, sketch tokenizer, guided velocity field,
and RK4 integration. Audio decoding/features, checkpoint loading, SurgePy
rendering, and metrics remain in the local Python CLI. SurgePy itself is **not**
compiled to WebAssembly. No inference request goes to a remote model service.

## 1. Install the real prerequisites

Use Linux x86-64, Python 3.12.13, Node 24, and a worktree-local environment. The
pinned SurgePy extension currently supports Linux x86-64. No GPU, VST3 bundle,
Docker image, Xvfb, RunPod job, or W&B service is needed for this path.

From the repository worktree:

```bash
uv sync --frozen --python 3.12.13 --extra cpu --no-default-groups --group dev
npm ci --prefix src/synth_setter/web
(cd src/synth_setter/web && npx playwright install chromium)
uv run --extra cpu python -c "import surgepy; import onnxruntime"
rclone lsd --checksum r2:experiments/browser-evaluation
```

The headless Chromium installation is needed for the automated E2E test. For
interactive use, open the printed URL in a local Chromium-based browser. ONNX Runtime
Web uses its CPU/WASM execution provider; WebGPU is not required or enabled.

R2 must be configured with read access to the fixture prefix. See
[storage credentials](../design/storage-provenance-spec.md#9-secrets).
CI uses the existing `setup-r2` action and repository secrets
`RCLONE_CONFIG_R2_ACCESS_KEY_ID`, `RCLONE_CONFIG_R2_SECRET_ACCESS_KEY`, and
`RCLONE_CONFIG_R2_ENDPOINT`. Credentials never enter the browser payload.

## 2. Select the verified checkpoint and training statistics

These content-addressed snapshots are also used by the production-path test:

```bash
export CHECKPOINT_SHA=87a0fcb20f9efbcb56fffc6e3ee6e1c8743c47013c297e33516b472543468c07
export STATS_SHA=c0c45d75a8b77004b3802c761bc77b5b34e7709a08343b2cf70fee04b7f52a19
export CHECKPOINT="r2://experiments/browser-evaluation/${CHECKPOINT_SHA}/checkpoint.ckpt"
export STATS="r2://experiments/browser-evaluation/${STATS_SHA}/stats.npz"
```

The checkpoint was produced by the real `flow_sketch_prelim` training run at
`global_step=99999`, with the `surge_simple` parameter spec (92 encoded
coordinates). Its source was:

```text
r2://intermediate-data/checkpoints/flow_sketch_prelim/flow_sketch_prelim-20260902T044048985Z-eed5063da1164b1e92ac62a55ffc17b3/last.ckpt
```

The statistics source was:

```text
r2://experiments/data/surge-simple-surgepy-lance-2m-40k-10k/surge-simple-surgepy-lance-2m-40k-10k-20260824T195308545Z/stats.npz
```

Both snapshots preserve the downloaded bytes and SHA-256 digests. Use the
content-addressed copies above, not the mutable `last.ckpt` object. The older
built-in sketch/NSynth checkpoint pin differs from that object; this is tracked
in [#3333](https://github.com/tinaudio/synth-setter/issues/3333). The explicit
arguments below avoid relying on that stale default.

For your own trained model, provide its trusted digest and its **training** mel
statistics. Checkpoints are loaded through Python serialization only after
digest verification; never accept an untrusted checkpoint and matching digest.

The current export supports the standard velocity-parameterized
`VSTFlowMatchingModule` with mel and sketch conditioning, one input pair per
browser session, float32 CPU weights/inputs, and sketch frames already on the
checkpoint's control-token grid. Endpoint models and sampling subclasses are
rejected rather than silently changing their inference semantics. Exported
shapes are fixed to the supplied pair.

## 3. Supply audio, or produce a real SurgePy input

To exercise both input roles without external recordings, generate two real
four-second stereo clips with different pitch and timing:

```bash
mkdir -p logs/browser-demo
uv run --extra cpu python - <<'PY'
from synth_setter.cli.sketch_render import load_render_config
from synth_setter.data.vst.core import write_wav
from synth_setter.renderer_factory import make_audio_renderer

render = load_render_config()
for name, note, velocity, window in (
    ("sketch", 60, 100, (0.0, 2.0)),
    ("content", 72, 80, (0.5, 1.5)),
):
    audio = make_audio_renderer(render).render(
        {}, midi_note=note, velocity=velocity, note_start_and_end=window
    )
    write_wav(audio, f"logs/browser-demo/{name}.wav", render.sample_rate, render.channels)
PY
```

Alternatively, substitute your own `sketch.wav` and `content.wav` files. The sketch
supplies timing/pitch; the content supplies timbre. The CLI decodes, resamples,
and pads/crops both to the checkpoint's 44.1 kHz, stereo, four-second grid, then
applies the pinned statistics and production sketch extraction.

## 4. Run browser inference and native evaluation

```bash
uv run --extra cpu synth-setter-sketch \
  logs/browser-demo/sketch.wav logs/browser-demo/content.wav \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --stats "$STATS" --stats-sha256 "$STATS_SHA" \
  --inference-runtime browser --browser-port 8765 --device cpu \
  --content-cfg 2 --sketch-cfg 3 --sample-steps 8 --seed 17 \
  --output-dir logs/browser-demo/evaluation --no-upload
```

1. Wait for `Browser evaluation: http://127.0.0.1:8765` in the terminal.
2. Open that exact URL and click **Run flow evaluation**.
3. The page reports sampling progress, then displays the predicted parameter
   vector. The server validates its session token, width, and finite float32
   values before accepting it.
4. The one-shot loopback server closes after accepting the prediction. The CLI
   resumes native rendering/metrics and prints the completed arm directory;
   inspect the durable artifacts below.

For a remote devcontainer, forward port 8765 to the same local port and open
`http://127.0.0.1:8765`. This is a loopback developer tool, not an authenticated
public web deployment. It rejects non-bundle paths and cross-origin prediction
requests. A session expires after ten minutes without a valid prediction.

The CLI reports CPU preprocessing/export separately from browser WASM inference.
Browser mode accepts `--device auto` or `--device cpu`; explicit CUDA/MPS settings
are rejected rather than silently overridden. Runtime assets are checked before
checkpoint/audio loading.

Eight steps keep the smoke test small. The normal model setting is 200 steps;
choose the step count explicitly for an experiment. Browser sampling supports
1–1000 steps and never silently falls back to Python inference. Use a fresh
output directory for each invocation: existing arms and graph bundles are not
overwritten. Multiple CFG arms run as sequential browser sessions.

## 5. Inspect the output

The command above produces:

```text
logs/browser-demo/evaluation/arms/cfg-c2-s3/
├── browser/
│   ├── conditioning.onnx
│   ├── velocity.onnx
│   ├── input.json          # normalized inputs, noise, guidance, steps
│   ├── prediction.json     # accepted browser parameter vector
│   └── provenance.json     # artifact digests, source revision, renderer, seed
├── sketch.wav
├── target.wav
├── pred.wav
├── params.csv
└── metrics.csv
```

`provenance.json` records the producing source checkout's `git_revision`, captured
before preprocessing. A `-dirty` suffix covers staged, unstaged, and untracked
changes; installations without their own source checkout report `git-unavailable`
rather than borrowing the operator's current repository.

`metrics.csv` contains MSS, weighted MFCC, spectral optimal transport, RMS
similarity, MLDR, and mid/side MLDR. Finite metrics and audible output establish
integration, not model quality. Listen to `target.wav` and `pred.wav` with your
audio player. `--no-upload` keeps generated artifacts local; downloading the
pinned checkpoint/statistics still reads R2.

To compare against native inference, repeat the command with
`--inference-runtime torch` and a **different** output directory, retaining the
same input files, checkpoint, statistics, seed, guidance, and step count.

## 6. Run the same production-path test locally and in CI

```bash
mkdir -p logs
npm test --prefix src/synth_setter/web
uv run --extra cpu pytest tests/models/test_flow_onnx.py \
  tests/evaluation/test_browser_flow.py tests/evaluation/test_browser_http.py \
  tests/test_sketch_render.py \
  tests/integration/test_browser_surgepy_e2e.py -v \
  --basetemp=logs/browser-e2e --junitxml=logs/browser-e2e.xml \
  --cov=synth_setter.evaluation.browser_flow --cov=synth_setter.models.flow_onnx \
  --cov=synth_setter.cli.sketch_render --cov-report=xml:coverage.xml
```

The E2E test generates distinct real SurgePy inputs, invokes the installed CLI, drives
Chromium through the shipped UI, and consumes its prediction with the real
renderer and metric implementations. It also rejects malformed browser output
and non-bundle file access. There are no mocks, fakes, synthetic checkpoints,
intercepted requests, or substitute models on this path.

The native reference independently reconstructs preprocessing from the original
files and noise from the CLI seed, rather than trusting the browser payload.
Assertions cover transported input/noise fidelity, asymmetric CFG (content 2,
sketch 3), browser/PyTorch parameter parity (`rtol=atol=2e-4`), source revision,
input-role audio fidelity within one PCM16 quantization step, stereo dimensions,
finite non-silent prediction audio, and all six finite metrics. The checkpoint is
a real training output, not a randomly initialized smoke model.

The adjacent fast tests cover export/input contracts, direct HTTP rejection and
authentication behavior, source provenance, and multi-step time-dependent RK4.

[Browser SurgePy flow E2E](../../.github/workflows/browser-flow-e2e.yml) runs this
same command on trusted PRs and `main` with CPU PyTorch. It installs the real
pinned npm runtime and Chromium, checks R2/SurgePy prerequisites, and fails if
the E2E test is skipped. Fork PRs cannot access the private R2 artifacts and are
excluded at the job boundary.

The workflow uploads `browser-surgepy-e2e-<run-id>` with the screenshot, CLI log,
WAVs, parameter/metric CSVs, input/prediction/provenance JSON, JUnit report, and
coverage. Large ONNX graphs are reproducible from the checkpoint and are not
uploaded as CI artifacts.

## Troubleshooting

- **Missing runtime assets:** run `npm ci --prefix src/synth_setter/web`.
- **Missing headless browser:** install Chromium from that directory using
  `npx playwright install chromium`. Playwright prints any missing OS libraries.
- **Digest mismatch:** stop and verify artifact provenance; do not disable the
  check or substitute a random-weight model.
- **Port already in use:** omit `--browser-port` for dynamic allocation, or
  choose another port and update your forwarding rule. Binding failures name the
  port; retain any exported bundle for diagnosis and use a fresh output directory.
- **Browser error/timeout:** inspect the page status and terminal; keep the
  generated bundle for diagnosis. No failed prediction is sent to rendering.
- **Existing arm:** choose a new `--output-dir`; do not mix graph generations.
- **Stacked PR auto-approval is red:** the known main-only policy is tracked in
  [#2359](https://github.com/tinaudio/synth-setter/issues/2359). It is separate
  from the production E2E result and does not mean the stack is merge-ready.
