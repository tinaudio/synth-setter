# KR106 native dataset renderer

`kr106-native` binds the pinned KR106 2.5.13 header-only DSP with pybind11. Building requires C++17 and network access to download the SHA-256-pinned upstream archive, not JUCE or a VST host. Rendering is offline. The audio dependency group installs the local package:

```bash
uv sync --extra cpu --frozen
```

## Dataset integration

Use `experiment=generate_dataset/ultramaster-kr106-native-lance-smoke` for 20 rows, or `experiment=generate_dataset/ultramaster-kr106-native-lance-2m-40k-10k` for the production split sizes. These are native variants of the existing operators; configure R2 credentials, destination, and queue settings as usual before invoking `synth-setter-generate-dataset`.

The native identities `ultramaster_kr106_native`, `ultramaster_kr106_onehot_native`, and `ultramaster_kr106_single_note_native` retain their existing KR106 ParamSpecs and encodings. Their distinct synth/backend identities prevent native and VST shards from sharing a dataset digest.

`SynthParamMap` resolves normalized repository controls to native IDs. `KR106NativeRenderer` converts ranges once; the extension implements engine-specific dispatch. Master volume retains the squared taper, voice counts snap to 6/8/10, and oversampling snaps to 1/2/4. MIDI SysEx output has no audio effect. Unknown, non-finite, and out-of-range controls fail rather than disappearing silently.

## Rendering contract

- A fresh DSP instance for every clip; the Python adapter can persist across a shard.
- Defaults → extracted baseline → selected program → explicit controls. The baseline model selects the program bank; subsequent explicit model selection overrides it.
- Baselines are extracted from committed VST preset DSP arrays, with both source-preset and native-baseline SHA-256 checks. Relative baseline paths resolve against the operator workspace, not the worker's output directory.
- Fixed maximum blocks (512 by default), split at exact note boundaries. Transport starts at beat zero, plays at 120 BPM, and follows the sample clock.
- Exact configured duration, including the configured release interval. No automatic tail, normalization, clipping, GUI, desktop settings, or file I/O.
- NumPy-owned, contiguous, channel-first float32 output, rendered directly into its storage. The GIL remains held. Mono is an explicit downmix in the Python adapter.
- Backend version `0.1.0`, render contract v2, source revision and archive digest pinned in the registered identity. Changing DSP, timing, or conversion policy requires a provenance/version update.

Regenerate baseline/map snapshots from the committed presets with:

```bash
uv run python scripts/generate_kr106_native_maps.py
```

## Verification

```bash
uv run python -m pytest packages/kr106-native/tests tests/data/vst/test_kr106_native_renderer.py tests/data/vst/test_kr106_native_map.py
uv run python -m pytest tests/integration/test_kr106_native_pipeline_e2e.py
```

The pipeline test uses real native synthesis, Lance writes, the local R2 transport fixture, finalization, a one-step model training run, and prediction/evaluation with native audio rendering. It does not replace the renderer with a mock or skip when the extension is unavailable.

`tests/data/vst/test_kr106_native_vst_parity_e2e.py` additionally requires the real KR106 VST and both hosts. It checks identical persisted labels and perceptual compatibility for a seeded J60 patch and an unmodulated J106 saw. MSS, RMS-envelope similarity, and wMFCC use the full clip. SOT uses the played-note interval: its per-bin time normalization makes quiet analog-noise tails dominate full-clip comparisons, including comparisons between the VST hosts themselves. Thresholds remain MSS < 3, RMS > 0.95, SOT < 0.01, and wMFCC < 4; full-clip SOT is retained as a diagnostic.

Native A→B→A audio must be sample-identical on the same platform. VST-host output is not promised to be sample-identical, and cross-platform bitwise determinism is not established. These fixtures do not prove equivalence for every patch or host transport configuration.

## License

The upstream KR106 DSP is GPL-3.0. See `kr106_native/PROVENANCE.md` and `kr106_native/KR106-GPL-3.0.txt`. Wheels and source distributions carry the provenance and license; the build retrieves the unmodified DSP source from the recorded archive.
