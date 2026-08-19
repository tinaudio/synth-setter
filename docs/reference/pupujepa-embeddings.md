# PupuJEPA audio embeddings

`synth-setter-add-embeddings` exposes the frozen PupuJEPA Tiny and Large teachers as
`embeddings=[pupujepa_tiny]` and `embeddings=[pupujepa_large]`. Offline augmentation and online
waveform conditioning both use `PupuJepaAudioEncoder`; the predictor, student, masking, and
training runtime are not included.

## Provenance and identity

The inference subset is adapted from MIT-licensed
[`sizigi/PupuJEPA`](https://github.com/sizigi/PupuJEPA) commit
`54a621e9f879be7659d81b6a3c493bba855cc85f`. The retained license is in
`LICENSES/PupuJEPA-MIT.txt`. The default artifacts come directly from
[`spellbrush/PupuJEPA`](https://huggingface.co/spellbrush/PupuJEPA) revision
`2ba230e41440c5b450a8dc8ad5d4a3cc9930f01d`:

| Variant | Args                              | Weights                                                                         | Selected-file digest                                               |
| ------- | --------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Tiny    | `pupujepaV2_25hz_tiny/args.json`  | `pupujepaV2_25hz_tiny/checkpoint/step-0500000_loss-0.125064/model.safetensors`  | `7bfd3e04fce4131496362a69eed5b478980181668e918adfaaef4e602bbceb2a` |
| Large   | `pupujepaV2_25hz_large/args.json` | `pupujepaV2_25hz_large/checkpoint/step-0500000_loss-0.176985/model.safetensors` | `9e16f31ee25371dcb0e7e97264dfeab2d9318f55f6939504d22fc66e29c3fc84` |

Only the selected variant's two files are materialized through `huggingface_hub`; no remote code
is loaded. Digests frame snapshot-relative paths and contents while ignoring files from another
variant in the shared snapshot. Configuration is validated before safetensors loading, and the
patch embed plus teacher key set is loaded strictly. The implementation pins `timm==1.0.28`,
tested against PupuJEPA's EVA/RoPE teachers and the existing TinyMU MATPAC path.

## Representation contract

Audio is downmixed to mono, resampled to 24 kHz, and transformed with the upstream frontend:
1,024-sample Hann STFT, 240-sample hop, `center=False`, 392-sample reflection padding on each side,
and 128 default librosa mel filters over 0–12 kHz. Magnitude is clamped to `1e-5`, natural-log
scaled, then normalized by mean `-4.089994845986366` and standard deviation
`2.0242277159094813`.

Both teachers patch four mel frames by 16 frequency bins. Four-second audio produces 400 mel
frames and 100 time patches. Other nonempty lengths are accepted when they contain at least one
complete four-frame patch.

| Variant | Teacher                        | Sequence shape                | Mean-pooled vector   | Offline batch cap |
| ------- | ------------------------------ | ----------------------------- | -------------------- | ----------------- |
| Tiny    | width 192, depth 12, 3 heads   | `(batch, 1536, time_patches)` | `pupujepa_tiny_vec`  | 16                |
| Large   | width 1024, depth 24, 16 heads | `(batch, 8192, time_patches)` | `pupujepa_large_vec` | 1                 |

Values, rank, orientation, and frame geometry are validated before Lance persistence. Both vectors
use the registry's cosine IVF_PQ policy, and both encoders run alone (`co_resident=False`). Large's
one-row cap limits attention memory. Its monolithic checkpoint is about 3.13 GiB, the loaded
teacher subset is about 1.50 GiB in float32, and one four-second sequence occupies about 3.125 MiB
before Lance overhead.

## Usage

```bash
synth-setter-add-embeddings \
  lance_uri=/path/to/dataset/train.lance \
  embeddings=[pupujepa_tiny,pupujepa_large]
```

Use `conditioning=pupujepa_tiny` or `conditioning=pupujepa_large` for cached four-second
sequences. Use the corresponding `_online` profile to resample waveforms and pool the frozen
teacher sequence at training or evaluation time. All profiles use the existing `EmbeddingPool`
head sized to the 100 time patches a four-second render emits, matching the cached profiles'
sequence length. Frozen teachers run in float32 even when the surrounding trainer uses mixed
precision.
