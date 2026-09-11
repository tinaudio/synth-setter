# Faust shimmer FDN dataset

`faust_shimmer_fdn` is the supplied eight-line shimmer feedback delay network,
excited by one unit impulse at sample zero of every fresh render. It produces
two identical output channels. The source includes the excitation, so the
registered source digest identifies both the FDN and its input signal.

The dataset varies FDN, pitch-shift, output, and safety controls—not MIDI pitch
or note duration. Compatibility note values are fixed and do not occupy encoded
parameter columns. Each render starts with fresh delay and filter state.

## 50,000-sample experiment

Preview the resolved configuration without generating or uploading data:

```bash
uv run synth-setter-generate-dataset \
  experiment=generate_dataset/faust-shimmer-fdn-lance-50k --cfg job --resolve
```

To generate:

```bash
uv run synth-setter-generate-dataset \
  experiment=generate_dataset/faust-shimmer-fdn-lance-50k
```

This command **generates data and uploads to R2**; it is not a preview. The
configuration requests 50,000 training samples, no validation/test samples, and
five 10,000-row Lance shards in the `experiments` bucket. Audio is four seconds
at 44.1 kHz. Override the split sizes if held-out data is needed.

The −70 LUFS loudness floor accounts for a single impulse rather than a
sustained note. Presets below the loudness meter's absolute gate are rejected
and resampled, so accepted parameters are not uniformly distributed over the
entire UI range. The full 50,000-sample run is separate from the small
integration test used to verify the addition.

## DSP provenance and safety

The supplied source declares **CC BY 4.0** and attributes the Faust port to the
pyFDN `example_shimmer_fdn` implementation of `td.PitchShift`, and to Dal Santo,
Pi, Prawda, Schlecht, and Välimäki, *Shimmer Reverberation with Nonlinear Feedback
Delay Networks*, DAFx26 (2026). The
[pyFDN notebook](https://artificial-audio.github.io/pyFDN/_static/marimo/notebooks/example_shimmer_fdn.html)
is the implementation reference. See the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

The repository adaptation adds fixed impulse excitation and condenses source
comments; it retains the supplied delay lengths, feedback matrix, absorption
filters, pitch shifters, DC compensation, energy guard, loop limiter, and output
sanitization/clipping. The energy-guard bypass remains a sampled control; the
loop limiter and output guard cannot be disabled. Long T60 settings can outlast
the four-second capture, so these samples are truncated responses, not complete
decay measurements.
