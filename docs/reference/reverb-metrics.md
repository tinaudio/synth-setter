# Reverb response metrics

The audio-metrics CLI adds these metrics with `--renderer-backend pyfdn`.
Predict-mode evaluation forwards this option when `render.renderer_backend=pyfdn`,
including evaluation of a checkpoint produced by training. Enable
`evaluation.render_vst=true` and `evaluation.compute_metrics=true` in predict mode.
The metrics are evaluation diagnostics, not additions to the training objective.

Results appear in `metrics/metrics.csv`, `metrics/aggregated_metrics.csv`, and
`audio/<metric>_{mean,std}` in the evaluation metric dictionary and
`metrics/metrics.json`. One-sample aggregate standard deviations are undefined.
Generic non-pyFDN evaluation retains its existing metric columns.

## pyFDN ResponseLoss coverage

The wrappers use all ten public concrete `ResponseLoss` implementations in
[pyFDN v0.4.2](https://github.com/artificial-audio/pyFDN/tree/v0.4.2/src/pyFDN/train/losses),
with their upstream default settings. Inputs are finite, equal-length mono impulse
responses at a shared sample rate. Floating-point filter responses may have gain
above unity: these APIs do not impose PCM sample bounds or clip responses before
scoring amplitude-sensitive losses. Upstream window and decay-fit requirements
still apply: silence, insufficient response length, or an unfittable decay can
make a loss undefined. Such failures and non-finite results raise an error naming
the metric rather than being silently omitted.

### Matching losses

These compare prediction against target; lower is better.

| Column                          | Upstream class          | Quantity                                                             |
| ------------------------------- | ----------------------- | -------------------------------------------------------------------- |
| `pyfdn_match_impulse_response`  | `MatchImpulseResponse`  | Samplewise impulse-response MSE                                      |
| `pyfdn_match_magnitude`         | `MatchMagnitude`        | FFT-magnitude MSE                                                    |
| `pyfdn_match_spectrogram`       | `MatchSpectrogram`      | FLAMO multi-resolution STFT loss                                     |
| `pyfdn_match_mel_spectrogram`   | `MatchMelSpectrogram`   | FLAMO multi-resolution mel-spectrogram loss                          |
| `pyfdn_match_energy_decay`      | `MatchEnergyDecay`      | Target-masked octave EDC RMS error in dB                             |
| `pyfdn_match_cumulative_energy` | `MatchCumulativeEnergy` | RMS error of normalized, compressed cumulative time–frequency energy |

`octave_edc_rmse_db` remains a compatibility alias of the same
`MatchEnergyDecay` result. Do not count the two columns as independent evidence.

### Standalone diagnostics

Each base name below produces `_target` and `_pred` columns. These are individual
response objectives, **not target-versus-prediction distances**. Identical
responses produce equal diagnostic values, not necessarily zero.

| Column base                       | Upstream class            | Quantity                                                          |
| --------------------------------- | ------------------------- | ----------------------------------------------------------------- |
| `pyfdn_flat_magnitude`            | `FlatMagnitude`           | Magnitude MSE relative to unit magnitude                          |
| `pyfdn_asymmetric_flat_magnitude` | `AsymmetricFlatMagnitude` | Gain-invariant spectral flatness objective emphasizing peaks      |
| `pyfdn_flat_spectrogram`          | `FlatSpectrogram`         | Gain-invariant multi-resolution Welch-spectrum flatness objective |
| `pyfdn_energy`                    | `Energy`                  | Squared deviation of total impulse-response energy from one       |

In particular, `pyfdn_energy_*` is not raw energy. A smaller flatness objective
is not necessarily a better match to a deliberately colored target.

## Joint time–frequency transport

`joint_time_frequency_ot` compares globally normalized linear STFT-energy maps.
It is balanced Wasserstein-1 on a time/log-frequency grid, with ground cost

```math
c((t,f),(t',f')) = \frac{|t-t'|}{0.1\,\mathrm{s}} + |\log_2(f/f')|.
```

Thus 100 ms of temporal displacement costs as much as one octave of frequency
displacement. The score is dimensionless and lower is better. This weighting is
an explicit experimental choice, not a validated perceptual equivalence.

Unlike framewise SOT, joint transport can move mass across both axes and detects
changed time–frequency associations even when the separate time and frequency
marginals agree. Global normalization removes overall gain, but preserves
relative energy between frequencies and times. It does not align away onset
shifts. Both-silent inputs score zero; a single silent input has no normalized
energy distribution and raises an error.

The representation uses a 2048-sample Hann STFT with 512-sample hop, pooled by
energy sum into 32 log-frequency bands and at most 64 time cells. At 44.1 kHz,
positive-frequency bins span approximately 21.53 Hz to 22.05 kHz; DC energy is
assigned to the lowest band because zero frequency has no logarithm. Grid
coordinates retain physical time and log-frequency spacing after pooling.

The implementation solves a sparse adjacent-edge minimum-cost flow, rather
than allocating a dense all-pairs transport matrix.
Its resolution limits narrow-mode and closely spaced reflection discrimination.
Quiet tails contribute little because mass is linear energy; keep EDC dB-RMSE,
T30, and C50 alongside this score. This metric is neither JTFS nor unbalanced OT.
