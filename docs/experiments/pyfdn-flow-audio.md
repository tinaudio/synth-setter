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
in PyTorch, not replaced with a fixed Householder matrix. Ordinary clipping uses
an affine straight-through gradient so out-of-range flow predictions are not
stranded at physical bounds; forward values still match offline decoding exactly.

`BasicFDNParamSpec.to_basic_fdn()` converts a native template into a
`BasicFDN(build: pyFDN.FDNBuild)` describing the **entire** effect. Offline
rendering uses `BasicFDN.impulse_response()`; `BasicFDN.to_flamo()` delegates graph
construction directly to upstream `dss_to_flamo`, without topology reconstruction.
Its NumPy boundary is used only during construction. Each forward pass binds the decoded tensors
through FLAMO's external-parameter API; no predicted value crosses NumPy or becomes
an independent trainable parameter. FLAMO updates its cached parameter values
while evaluating those supplied tensors, so rows are evaluated serially before
stacking. The renderer is not a concurrent/shared-service interface.

`PyFDNParameterDecoder.decode_build_fields()` converts the model-specific controls
to tensor counterparts of the varying build fields, including pyFDN's
first-order decay shelf. This encoding/binding layer is separate from `BasicFDN`.

## Build geometry and registered model encodings

`FlamoFDNDifferentiableRenderer(fdn=..., decoder=..., parameter_width=...)`
accepts a complete `BasicFDN` and a separate tensor-field decoder. Input/output
counts and sample rate come from the build, not a mono assumption. Each input
impulse is rendered separately; output shape is
`(batch, outputs * inputs, samples)`, ordered by
`channel = output * input_count + input`. No transfer path is summed or selected
away. All three optional SOS hooks are bound in the upstream graph.

Audio feedback compares corresponding channels before averaging their scalar
losses. It rejects channel-count mismatches rather than broadcasting or downmixing.
A singleton channel axis may be removed for legacy consumers without changing
samples; multiple channels remain intact.

The `from_param_spec()` factory adapts these currently registered basic encodings:

- `pyfdn_n8_mono_householder`
- `pyfdn_n8_mono_householder_vector`
- `pyfdn_n8_mono_kronecker`

Select the corresponding `synth=` configuration and a dataset generated with that
same identity. The renderer accepts that spec's width rather than assuming the
fixed-Householder encoding.

All other registered pyFDN effects are rejected at renderer construction, even
though the tensor decoder can decode their native controls. Götz adds input tone
correction and a delayed direct path outside its current build; pitch shifting is
time-varying; DiffVox is a composite stereo effects chain. Possessing an FDN core
does not make these complete `BasicFDN` effects, and no fallback graph is supplied.
Integer delays and Kronecker reflection choices have zero gradients. Continuous
gains, feedback coordinates, and filter controls retain gradients.

## Numerical and runtime limits

FLAMO evaluates a frequency-domain response. Finite FFT periods introduce circular
tail aliasing; increasing `model.audio_loss.renderer.fft_size` reduces it. Output
amplitude follows pyFDN without clipping. The renderer produces complete impulse
responses, not arbitrary input-audio convolutions. The registered factory uses the
canonical pyFDN dataset rate; directly supplied builds may use another rate.

Device-only `.to(device)`, `.float()`, `.double()`, and same-dtype
`.to(dtype=torch.float32)` are exercised. The adapter rebuilds FLAMO's recursion
buffers after recursive tensor conversions to contain the pinned dependency defect
tracked in #3380. Compiled and distributed feedback retain the existing runtime guards.

Importing pinned `torchsynth.util` mutates global `torch.pi` to a float32
approximation (#3402). This makes subsequent float64 FLAMO construction lose
precision, including strict MIMO parity tests when run after TorchSynth training
tests. Isolated parity runs pass; mixed-backend float64 parity remains blocked.
No tolerance relaxation or local DSP workaround masks this defect.

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
