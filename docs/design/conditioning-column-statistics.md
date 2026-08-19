# Conditioning-column statistics

## Method

The sample is the first 1,000 training rows from
`r2://experiments/data/surge-simple-lance-440k-20k-20k/surge-simple-lance-440k-20k-20k-20260706T005448315Z/train.lance`.
The six columns already materialized there were streamed directly from R2. T5Gemma and PupuJEPA
Tiny/Large were materialized from the same 1,000 source `audio` and `param_array` rows with the
production `synth-setter-add-embeddings` command, then streamed from the resulting local Lance
dataset. PupuJEPA Large is included because it is now a cached conditioning profile, although it
was not in the original requested list.

All standard deviations are population standard deviations. For sequence columns, each channel's
mean and standard deviation aggregate its values across rows and frames. The frame-L2 coefficient
of variation (CV) is calculated within each row and then averaged across rows, so it excludes
between-row magnitude variation. A dead channel has standard deviation below `1e-6`.

```bash
uv run python scripts/dev/characterise_conditioning_columns.py DATASET.lance \
  --rows 1000 --output conditioning-statistics.md
```

## Results

| Column               | Rows | Channel mean min / median / max    | Channel std min / median / max | Global mean | Global std | Global min | Global max | Dead channels | Row L2 mean | Row L2 std | Mean within-row frame-L2 CV |
| -------------------- | ---: | ---------------------------------- | ------------------------------ | ----------: | ---------: | ---------: | ---------: | ------------: | ----------: | ---------: | --------------------------: |
| music2latent (`m2l`) | 1000 | -3.241250 / -0.182910 / 2.123648   | 0.442614 / 0.525712 / 0.725511 |   -0.197419 |   1.263385 |  -7.113281 |   6.531250 |             0 |   93.647503 |   4.530034 |                    0.056223 |
| same_s               | 1000 | -1.116463 / -0.093098 / 1.234508   | 0.163584 / 0.230661 / 0.370314 |   -0.042528 |   0.592932 | -28.152214 |  39.349365 |             0 |   62.773991 |   6.313474 |                    0.052864 |
| same_l               | 1000 | -1.210466 / -0.067223 / 1.341422   | 0.158052 / 0.235080 / 0.366926 |   -0.044321 |   0.629139 |  -8.613997 |   8.687063 |             0 |   66.739660 |   5.139751 |                    0.050616 |
| matpac_plus          | 1000 | -1.710274 / -0.023306 / 0.528809   | 0.000468 / 0.094513 / 0.440439 |   -0.027790 |   0.155596 |  -2.217311 |   1.330280 |             0 |   48.950045 |   1.483457 |                    0.022554 |
| matpac_plus band 1   | 1000 | -1.462614 / -0.023470 / 0.510552   | 0.000502 / 0.102721 / 0.440439 |   -0.027731 |   0.159357 |  -2.170089 |   1.330280 |             0 |   22.402181 |   0.697817 |                    0.024551 |
| matpac_plus band 2   | 1000 | -1.601936 / -0.023964 / 0.524440   | 0.000481 / 0.096422 / 0.423376 |   -0.027516 |   0.155817 |  -2.200127 |   1.279681 |             0 |   21.913408 |   0.702047 |                    0.027294 |
| matpac_plus band 3   | 1000 | -1.650063 / -0.022806 / 0.528809   | 0.000479 / 0.094464 / 0.407988 |   -0.027696 |   0.154746 |  -2.217311 |   1.299698 |             0 |   21.770462 |   0.736061 |                    0.026783 |
| matpac_plus band 4   | 1000 | -1.699768 / -0.021572 / 0.524747   | 0.000468 / 0.089904 / 0.375960 |   -0.027697 |   0.153772 |  -2.213012 |   1.257797 |             0 |   21.638172 |   0.718218 |                    0.025411 |
| matpac_plus band 5   | 1000 | -1.710274 / -0.024226 / 0.461242   | 0.000487 / 0.087787 / 0.351116 |   -0.028309 |   0.154223 |  -2.205247 |   1.108494 |             0 |   21.715780 |   0.690185 |                    0.021812 |
| pupujepa_tiny        | 1000 | -2.752047 / -0.001471 / 10.365058  | 0.275437 / 0.444383 / 3.893067 |   -0.025001 |   1.075861 |  -7.664564 |  15.612594 |             0 |  420.626707 |  30.947511 |                    0.081109 |
| pupujepa_large       | 1000 | -22.357522 / -0.000309 / 11.975433 | 0.784604 / 1.857159 / 7.626107 |    0.020806 |   2.454772 | -40.010300 |  31.701796 |             0 | 2221.844657 |  13.519245 |                    0.009130 |
| t5gemma              | 1000 | -13.082806 / -0.082158 / 9.919632  | 2.400819 / 4.142338 / 8.031981 |   -0.096137 |   5.196103 | -52.176792 |  31.708429 |             0 | 2304.372245 |   0.000000 |                    0.076731 |
| clap                 | 1000 | -0.093430 / -0.002290 / 0.119841   | 0.019192 / 0.028088 / 0.043186 |   -0.000975 |   0.044183 |  -0.174224 |   0.194696 |             0 |    1.000000 |   0.000000 |                           — |
| ssondo               | 1000 | -0.300554 / -0.033079 / 0.596355   | 0.004504 / 0.014059 / 0.101316 |   -0.029548 |   0.083472 |  -0.329537 |   0.771413 |             0 |    2.738516 |   0.166143 |                           — |

## Recommendations

- **music2latent (`m2l`): standardise** — channel offsets span -3.24 to 2.12 and are not removed by the encoder.
- **same_s: standardise** — channel scales and offsets vary substantially despite stable row norms.
- **same_l: standardise** — its scale profile closely matches SAME-S and is not unit-normalized.
- **matpac_plus: standardise** — use one shared scale across all five bands, not per-band scaling, to preserve relative spectral energy.
- **pupujepa_tiny: already normalised — skip** — its transformer output is globally centered with standard deviation 1.08.
- **pupujepa_large: standardise** — unlike Tiny, its global standard deviation is 2.45 and channel scales are highly uneven.
- **t5gemma: standardise** — its global standard deviation is 5.20 and per-channel medians are 4.14.
- **clap: already normalised — skip** — every sampled row has L2 norm 1.000000.
- **ssondo: standardise** — row norms average 2.74 and channel scales vary by more than 20x.

## Training integration

Each measured profile now declares its normalization policy. Sequence columns use one dataset-level
mean and standard deviation per channel, broadcast over frames; vector columns use one pair per
feature. MATPAC++ uses one global pair across all channels and bands so relative spectral energy is
preserved. CLAP and PupuJEPA Tiny skip normalization. MeanAudio remains unchanged because it was not
measured here.

The streaming Welford command writes an immutable, column-specific artifact beside the finalized
training split. It accepts local or canonical R2 Lance URIs and publishes
`conditioning_stats.<column>.npz`; finalized `stats.npz` remains owned exclusively by Lance
finalization.

```bash
uv run python -m synth_setter.pipeline.data.stats TRAIN.lance \
  --conditioning-column same_l \
  --conditioning-shape 256 44 \
  --conditioning-normalization per_channel
```

The datamodule reuses the training affine for validation and test splits. Missing column statistics
are a backward-compatible no-op. Standard deviations below `1e-6` are replaced with one, so a dead
channel becomes zero after mean subtraction instead of producing a non-finite value.

## Specific checks

### CLAP L2 normalization

`pretrained_encoder.py` returns the `pooler_output` from
`ClapModel.get_audio_features`. Transformers applies `F.normalize(audio_features, dim=-1)` before
returning that output. The stored data confirms the contract exactly: row-L2 mean 1.000000 and
standard deviation 0.000000. CLAP should not be standardized after this normalization.

### T5Gemma scale

The comment in `pipeline/data/t5gemma.py` is not a claim that the stored representation has
standard deviation 1.75. It describes bfloat16 output drift between Torch releases: differences
had standard deviation 1.75 and maxima up to 11.5. The real float32 column has global standard
deviation 5.196, median channel standard deviation 4.142, and values from -52.177 to 31.708, so the
1.75 figure is refuted as a description of data scale. All sampled row embeddings are identical
(row-L2 standard deviation 0) because the currently registered `param_names` text normalizer
intentionally emits the same parameter-name caption for every row; token and channel values still
vary within that caption.

### MATPAC++ bands

The five band-level global standard deviations are 0.1594, 0.1558, 0.1547, 0.1538, and 0.1542;
mean row norms are 22.40, 21.91, 21.77, 21.64, and 21.72. The bands are therefore close in scale but
not individually unit-normalized. Their small, systematic energy differences are spectral shape,
so any future standardization should use a shared affine scale across all 3,840 channels rather
than independently forcing each band to the same energy.
