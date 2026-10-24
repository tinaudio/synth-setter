# Four-model Surge comparison

Choose **Load interactive demo** on the poster, or open the standalone Surge
page with `?compare=1`. The poster mounts the same comparison controls directly
without an iframe; neither path downloads model or engine assets until **Run
selected models** is chosen. All selected models load before their independent
browser workers start inference. Each result has its
own audio, spectrogram, JSON record and **Download FXP patch** link.

Each card reports the file being downloaded, received/total MiB, and graph
initialization. Velocity graphs exceed 100 MB each: leave progressing downloads
open, or select fewer models on a slow connection. A network transfer with no
bytes for 60 seconds fails with its filename; the comparison's four-minute
loading deadline resets on progress rather than cutting off healthy transfers.
Retry with **Run selected models** after a loading error.

| Arm             | Training step | Encoded output width |
| --------------- | ------------: | -------------------: |
| Simple baseline |       101,999 |                   92 |
| Simple sketch   |        99,999 |                   92 |
| XT baseline     |       199,863 |                  300 |
| XT sketch       |       200,000 |                  300 |

The corresponding synth parameter counts are 89 and 162; output widths also
include encoding and note coordinates. The previous 60k sketch snapshot is not
one of these four models. Exact R2 locations, SHA-256, MD5, sizes and statistics
pins are in `src/synth_setter/configs/browser_surge_comparison.yaml`.

## Comparison boundaries

Inputs and sampling settings are shared. The seed produces identical noise
within each parameter-space pair, not a shared 92/300-coordinate interpretation.
Baseline models ignore sketch input and use content-only guidance. Sketch-only
mode requires selecting sketch models alone.

Each model uses its own statistics-bound mel graph. XT statistics come from the
dataset named by both producing W&B runs. The simple pair uses the existing
repository evaluation statistics pin; historical W&B entries for those two runs
were unavailable during this publication. The parameter spaces and training
schedules differ, so cross-space results are not a controlled sketch ablation.

WebGPU is optional and off by default. It changes only the neural conditioning
and velocity graphs, with CPU fallback disabled. Concurrent timings include
resource contention and are not isolated benchmarks. The existing frontend
numerical discrepancy tracked in
[#3550](https://github.com/tinaudio/synth-setter/issues/3550) is not fixed by this
comparison feature.

## FXP downloads

The separate host saves the actual Surge state after rendered parameter events
have been applied. It wraps the native component state in an FXP container;
there is no fabricated XML preset or server-side patch conversion.

**These are Surge 1.4 patches.** SurgePy 1.3 loaded, saved and rendered the tested
patches, but reported patch version 28 versus 24 and warned that newer features
may be unavailable. Full 1.3 compatibility is not claimed. MIDI pitch, velocity
and note timing remain in the evaluation record/WAV rather than the FXP.

## Public Hugging Face snapshot

Repository: [`khaledtin/surge-browser-comparison`](https://huggingface.co/khaledtin/surge-browser-comparison) (public).

Verified revision: `16be86f391829413923322ff9f2185a5bad4d69c`.

- `checkpoints/<id>/model.ckpt`: original, byte-verified checkpoint.
- `checkpoints/<id>/stats.npz`: selected normalization archive.
- `browser/<id>/`: exported ONNX bundle and baseline preset.
- `browser/catalog.json`: four-model selection metadata.
- `checksums.json`: SHA-256 values for checkpoints, statistics and bundle files.

The deployed site serves checksum-verified copies downloaded from this immutable
HF revision. The public snapshot requires no HF token, and no credentials are
shipped to the browser. Export manifests retain the
honest `0853871487…-dirty` producing-revision stamp from the development export;
this is not a claim of a clean release build.

## Reproduce

Export directly from the pinned R2 snapshots:

```bash
uv run python -m synth_setter.tools.export_browser_surge_comparison --output build/models
node src/synth_setter/web/surge/export-site.mjs \
  --model build/models/simple-sketch --comparison build/models \
  --engine /absolute/fxp-capable-engine-bundle --output build/comparison-site
```

The engine bundle must contain the new state-saving host exports; the pinned
Surge engine itself remains unchanged.

For Cloudflare Pages, upload the checksum-verified ONNX files to the public R2
`models/<id>/` paths named in `_redirects` using `rclone --checksum`. Only after
those objects are readable without credentials, copy the exported tree for
Pages and remove the redirected `models/*/*.onnx` files. Keep the catalog,
manifests, presets, `_headers`, `_redirects`, and both licenses in the Pages
input. The R2 upload and Pages deployment are separate release operations; the
site export itself does not mutate either service.

Run the real-browser comparison against the served site:

```bash
SURGE_WASM_BUNDLE=/absolute/fxp-capable-engine-bundle \
SURGE_BROWSER_URL=http://127.0.0.1:8772/surge/index.html \
SURGE_BROWSER_CONTENT=/absolute/content.wav \
SURGE_BROWSER_SKETCH=/absolute/sketch.wav \
SURGE_BROWSER_OUTPUT=logs/comparison \
node --test src/synth_setter/web/surge/comparison.e2e.mjs
```

The test always exercises WASM and exercises WebGPU only when Chromium obtains a
WebGPU adapter; an unavailable adapter is reported as a skip, not a pass. It
checks four overlapping inference runs, within-pair noise identity, distinct
checkpoint/run IDs, non-silent predictions, browser-decoded WAVs, and FXPs
reloaded by the real Surge host. For headless hardware WebGPU flags, see [the
backend comparison guide](surge-webgpu-comparison.md).
