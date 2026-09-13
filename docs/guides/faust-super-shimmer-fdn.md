# Faust super-shimmer FDN

`faust_super_shimmer_fdn` is a 16-line feedback delay network excited by one
unit impulse at the start of each fresh render. Lines 0–7 retain the original
shimmer FDN's dual-head cubic pitch shifter and enable toggles; lines 8–15 use
four-voice Hann-windowed granulation at fixed playback ratio 1. The fixed ratio
changes texture without adding another transpose control.

The feedback matrix is the normalized fourth tensor power of H2. Its entries
are ±1/4, so the matrix is orthogonal. Nominal line lengths are 1,103–8,623
samples. Each line uses cubic fractional-delay reads modulated by a phase-offset
sine LFO. Rate spans 0.01–0.5 Hz and depth spans 0–256 samples; the 16,384-sample
line safely contains the longest modulated read and cubic interpolation taps.

Granular duration spans 0.04–0.12 seconds, position 0.08–0.25 seconds, and jitter
0–0.02 seconds. At ratio 1 the greatest requested offset is 0.27 seconds, below
the granulator's 65,536-sample line through 192 kHz. The loop
limiter, non-amplifying energy guards, non-finite rejection, output clipping,
damping filters, and fixed four-second compatibility note mapping match the
original shimmer source's safety contract.

## Attribution

The damping, dual-head pitch shifter, and safety processing derive from the
[original shimmer FDN](faust-shimmer-fdn.md#dsp-provenance-and-safety), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): the Faust port of
pyFDN `example_shimmer_fdn` / `td.PitchShift`, based on Dal Santo, Pi, Prawda,
Schlecht, and Välimäki, *Shimmer Reverberation with Nonlinear Feedback Delay
Networks*, DAFx26 (2026). This variant changes the network order, feedback
matrix, and delay modulation, and adds unity-ratio granulation.

## Faust-library compatibility

The pinned DawDreamer 0.8.3 Faust bundle predates
[`ef.granular`](https://faustlibraries.grame.fr/libs/misceffects/#efgranular)
and its `an.window_hann` dependency. The source therefore includes the exact
MIT-licensed `granular` algorithm and minimal Hann helper closure from
`faustlibraries` commit
[`271228a`](https://github.com/grame-cncm/faustlibraries/blob/271228a08981fa10b07732f0861421d1e20d4022/misceffects.lib#L1389-L1444).
Only helper names are localized; equations, constants, clamping, and random
sample-and-hold behavior are unchanged. This compatibility copy avoids changing
the renderer or its global library search path.

## Dataset use

Select the generated Hydra group with `synth=faust_super_shimmer_fdn`. The
source has no preset artifact and renders through the DawDreamer backend. Its
23 native controls encode to 32 model columns because the eight shimmer-line
toggles and energy-guard bypass use two-column one-hot encodings.

Generate and upload ten samples through the existing shimmer experiment:

```bash
uv run synth-setter-generate-dataset \
  experiment=generate_dataset/faust-shimmer-fdn-lance-50k \
  synth=faust_super_shimmer_fdn \
  'train_val_test_sizes=[10,0,0]' render.samples_per_shard=10 \
  task_name=faust-super-shimmer-fdn-smoke
```

This uses the original experiment's −70 LUFS acceptance threshold and
four-second capture. Rejected quiet renders are resampled; this is not a
uniformly sampled listening suite. The command stages fragments in R2;
run `synth-setter-finalize-dataset` on the emitted dataset root to finalize.
