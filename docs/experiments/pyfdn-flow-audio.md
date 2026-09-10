# pyFDN flow matching with direct audio feedback

`experiment=pyfdn/flow_audio_flamo` attaches the FLAMO renderer when Hydra
constructs the flow model. No pretrained checkpoint or separate simulator
finetuning stage is required. The experiment loads target audio from Lance and
uses the existing multi-scale spectral loss alongside flow matching; its inherited
`lambda_audio=0.03` already contributes to optimization.

```text
flow prediction / endpoint estimate
  → PyFDNParameterDecoder
  → native tensor matrices, delays, and filter controls
  → FLAMO graph with externally bound parameters
  → waveform → spectral distance to target audio → flow-network gradients
```

## Decoder and graph construction

`PyFDNParameterDecoder` reads bounds, native shapes, and encoded spans from the
registered ParamSpec. It reproduces offline `decode_model_output`: clipping for
ordinary controls, whole-vector projection for directions, periodic angle-pair
decoding, and the declared integer rounding rules. Feedback matrices are derived
in PyTorch, not replaced with a fixed Householder matrix.

`dss_to_flamo` constructs the graph once from a native template. Its NumPy boundary
is used only during construction. Each forward pass binds the decoded tensors
through FLAMO's external-parameter API; no predicted value crosses NumPy or becomes
an independent trainable parameter. FLAMO updates its cached parameter values
while evaluating those supplied tensors, so rows are evaluated serially before
stacking. The renderer is not a concurrent/shared-service interface.

Filter design reuses pyFDN's tensor-capable first-order shelf and GEQ assembly.
The Götz adapter preserves input tone correction and the delayed direct branch.

## Supported renderer identities

- `pyfdn_n8_mono_householder`
- `pyfdn_n8_mono_householder_vector`
- `pyfdn_n8_mono_kronecker`
- `pyfdn_gotz_n8_mono_fixed_delays`
- `pyfdn_gotz_n8_mono_learned_delays`
- `pyfdn_gotz_n8_mono_fixed_delays_givens`
- `pyfdn_gotz_n8_mono_learned_delays_givens`

Select the corresponding `synth=` configuration and a dataset generated with that
same identity. The renderer accepts that spec's width rather than assuming the
fixed-Householder encoding.

The decoder also covers the registered pitch-shift and DiffVox native controls,
but their renderer graphs are explicitly unsupported: pitch shifting is
time-varying, and DiffVox is a composite stereo effects chain. Integer delays and
Kronecker reflection choices have zero gradients. Continuous gains, feedback
coordinates, and filter controls retain gradients.

## Numerical and runtime limits

FLAMO evaluates a frequency-domain response. Finite FFT periods introduce circular
tail aliasing; increasing `model.audio_loss.renderer.fft_size` reduces it. Output
amplitude follows pyFDN without clipping. The current adapter renders mono impulse
responses at the canonical pyFDN sample rate, not arbitrary input audio.

Device-only `.to(device)`, `.float()`, and `.double()` are exercised. Avoid
same-dtype `.to(dtype=torch.float32)` with the pinned FLAMO version: it can corrupt
complex recursion buffers (tracked in #3380). Compiled and distributed feedback
retain the existing runtime guards.

## Reproducible integration experiment

This creates a tiny real local pyFDN Lance dataset, trains the configured flow
model, checks audio-only gradients to its vector field, then reloads the real
checkpoint through evaluation and reads rendered WAVs and audio metrics:

```bash
uv run pytest tests/test_train.py::test_train_flamo_real_pyfdn_dataset_checkpoint_evaluates -q
```

It runs for fixed and learned Householder feedback. Kronecker/Givens prediction
CSV export has an independent native-angle/encoded-width bug tracked in #3378;
that export path is not claimed to work here.

The same-prediction parity and CPU/CUDA gradient checks are:

```bash
uv run pytest tests/models/components/test_pyfdn_decoder.py tests/models/components/test_differentiable_renderer.py tests/models/components/test_pyfdn_audio_gradients.py -q
```

These checks compare offline decode → pyFDN render against online decode → FLAMO
render for every supported graph in float32 and float64, and verify continuous
control gradients and isolation between batch rows. They establish integration,
not improved learned sound-matching quality.
