# PupuJEPA Tiny audio embeddings

`synth-setter-add-embeddings` exposes the frozen PupuJEPA Tiny teacher as
`embeddings=[pupujepa_tiny]`. Offline augmentation and online waveform conditioning both use
`PupuJepaAudioEncoder`; the predictor, student, masking, and training runtime are not included.

## Provenance and identity

The inference subset is adapted from MIT-licensed
[`sizigi/PupuJEPA`](https://github.com/sizigi/PupuJEPA) commit
`54a621e9f879be7659d81b6a3c493bba855cc85f`. The retained license is in
`LICENSES/PupuJEPA-MIT.txt`. The default artifacts come directly from
[`spellbrush/PupuJEPA`](https://huggingface.co/spellbrush/PupuJEPA) revision
`2ba230e41440c5b450a8dc8ad5d4a3cc9930f01d`:

- `pupujepaV2_25hz_tiny/args.json`
- `pupujepaV2_25hz_tiny/checkpoint/step-0500000_loss-0.125064/model.safetensors`

Only those files are materialized through `huggingface_hub`; no remote code is loaded. Their tree
digest is pinned to `7bfd3e04fce4131496362a69eed5b478980181668e918adfaaef4e602bbceb2a`.
The configuration is validated before safetensors loading, and the patch embed plus teacher key set
is loaded strictly. The implementation pins `timm==1.0.28`, tested against both PupuJEPA's
EVA/RoPE teacher and the existing TinyMU MATPAC path.

## Representation contract

Audio is downmixed to mono, resampled to 24 kHz, and transformed with the upstream frontend:
1,024-sample Hann STFT, 240-sample hop, `center=False`, 392-sample reflection padding on each side,
and 128 default librosa mel filters over 0–12 kHz. Magnitude is clamped to `1e-5`, natural-log
scaled, then normalized by mean `-4.089994845986366` and standard deviation
`2.0242277159094813`.

The teacher patches four mel frames by 16 frequency bins. Its 192-dimensional states are grouped
across eight frequency patches, yielding `(batch, 1536, time_patches)`. Four-second audio produces
400 mel frames and 100 time patches. Other nonempty lengths are accepted when they contain at
least one complete four-frame patch. Values, rank, orientation, and frame geometry are validated
before Lance persistence. `pupujepa_tiny_vec` stores the temporal mean for the registry's cosine
IVF_PQ policy; the encoder runs alone (`co_resident=False`) and uses bounded 16-row chunks.

## Usage

```bash
synth-setter-add-embeddings \
  lance_uri=/path/to/dataset/train.lance \
  embeddings=[pupujepa_tiny]
```

Use `conditioning=pupujepa_tiny` for cached four-second sequences. Use
`conditioning=pupujepa_tiny_online` to resample waveforms and pool the frozen teacher sequence at
training or evaluation time. Both profiles use the existing `EmbeddingPool` head; the online head
accepts up to 256 time patches, matching the checkpoint's 1,024-frame training window. The frozen
teacher runs in float32 even when the surrounding trainer uses mixed precision.
