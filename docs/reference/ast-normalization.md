# Online AST normalization

The stored-mel and online AST encoders share the same transformer. Stored mels
are normally standardized by the datamodule; online mels can instead be
standardized inside `LogMelFrontend`, after peak-relative dB conversion and
before AST. Do not apply both normalization stages to the same input.

## Finalize

`finalize_dataset.yaml` exposes two top-level settings:

```yaml
estimate_normalization_stats: false
seed: 1234
```

With estimation disabled, finalize retains its exact reduction of the winning
training shards' stored-mel statistics. The seed is unused in this mode.

With estimation enabled:

- An existing `stats.npz` is reused, not overwritten. Only the statistics step
  is skipped; finalize continues its remaining work.
- If statistics are missing, finalize samples up to 10,000 training waveforms
  without replacement using the seed, computes raw online log-mels, and saves
  per-position mean and population standard deviation.
- The calibration frontend uses the canonical log-mel settings and the render
  sample rate and length. Estimation requires mono audio and at least two
  training waveforms.
- Validation and test audio never contribute to calibration.

A completed dataset remains immutable: its completion marker makes finalize a
no-op. Prefer existing exact training statistics when comparing stored and
online AST; independently estimated statistics do not establish exact parity.

## Online training

Enable calibration for an online AST experiment:

```bash
uv run python -m synth_setter.cli.train \
  experiment=pyfdn/flow_ast_online estimate_normalization_stats=true seed=1234
```

The training path reuses existing training-dataset statistics when available;
otherwise it calibrates on up to 10,000 training waveforms and saves
`stats.npz` under the run output directory. Normalization is fixed before the
first training step, and distributed workers receive the same statistics.

Unlike finalize, training already uses its top-level `seed` for model and
training randomness. Disabling estimation does **not** disable that existing
seeding behavior. When training's seed is `null`, calibration uses 1234 without
changing the normal training seed policy.

## Known statistics

The estimation flag controls calibration, not explicitly configured normalization.
Supply a local training-statistics file without requesting estimation:

```yaml
model:
  encoder:
    frontend:
      normalization_stats_path: /path/to/training/stats.npz
```

Alternatively, supply both values explicitly:

```yaml
model:
  encoder:
    frontend:
      normalization_mean: -40.0
      normalization_std: 20.0
```

These numbers are illustrative, not recommended defaults. Scalars apply global
normalization; existing stored-mel statistics generally have one value per
channel, frequency bin, and time frame. Use the original arrays for parity.
Statistics must be finite, standard deviations positive, and shapes compatible
with the frontend output. File and explicit-value settings are mutually exclusive.

## Checkpoints and third-party evaluation

Resolved normalization statistics are non-trainable model buffers saved in
checkpoints. Resume and evaluation must use those training statistics, not
estimate replacements from validation, test, or third-party corpora. If the
Hydra config explicitly names a statistics file that is no longer available,
set `model.encoder.frontend.normalization_stats_path=null` when reconstructing
the model; checkpoint loading restores the saved buffers without that file.

The stored-mel third-party path continues to use its training `mel_stats_uri`.
The online path applies normalization in the frontend. A model trained without
normalization should also be evaluated without normalization.

Torchaudio and librosa raw frontend outputs agree within numerical tolerance;
sharing statistics does not make their FFT implementations bit-identical.
