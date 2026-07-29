# MATPAC++ audio embeddings

`synth-setter-add-embeddings` exposes the frozen MATPAC++ representation as
`embeddings=[matpac_plus]`. TinyMU supplies the package API; this integration does not load its
language decoder or implement TinyMU training.

## Package and checkpoint identity

TinyMU exposes MATPAC through the public `tinymu.matpac` package API. The normal synth-setter
runtime installs that package from the immutable `ktinubu/TinyMU` commit
`fef8564593fceb5625c10f56a46b256216e7173d`; operators do not need a separate source checkout.
TinyMU source is MIT-licensed, while its MATPAC implementation and pretrained encoder weights are
Apache-2.0 licensed. The VCS requirement and resolved commit are recorded in `pyproject.toml` and
`uv.lock`.

The default checkpoint is the immutable R2 object:

```text
r2://intermediate-data/tinymu/source/pretrained/AndreasXi/TinyMU/0735fc50bc8b881d687dedccdd48b742927611b3/matpac_plus_as_48_1_map_enconly.pt
```

Its required SHA-256 is
`e8cec6847b2d918c8f77f82d79d90adf7dd82f99e80fa12eb3444f87f24bb998`. The integration downloads it
through the canonical rclone path into
`${XDG_CACHE_HOME:-$HOME/.cache}/synth-setter/models/embeddings/matpac-plus-0735fc50bc8b881d687dedccdd48b742927611b3/`,
then verifies the digest before moving it into place. A local `checkpoints.matpac_plus=/path/to/file`
override is accepted only when its digest is identical. Other R2 URIs are rejected.

## Measured encoder contract

The contract was measured with the checkpoint above and TinyMU's real `resource/example1.wav` at
the pinned source commit:

- input is finite `(batch, channels, samples)` audio with one or two channels;
- channels are averaged to float32 mono and resampled to 16 kHz;
- clips are not truncated to a fixed duration;
- clips shorter than 2,800 samples after resampling (175 ms) are rejected;
- upstream `precise` inference uses an 80-bin, 25 ms/10 ms-hop log-mel frontend and pads mel frames
  to 16-frame patch boundaries;
- output is transposed from upstream `(batch, tokens, 3840)` to the conditioning convention
  `(batch, 3840, tokens)`, persisted as a float32 fixed-shape tensor;
- the frozen model runs in eval/inference mode and repeated CPU inference is bitwise deterministic;
- `matpac_plus_vec` is the temporal mean, a `FixedSizeList<float32, 3840>` used by the registry's
  cosine IVF_PQ policy.

Measured token counts are 1 for 175 ms, 7 for one second, 25 for four seconds, and 63 for ten
seconds. Output width 3,840 is five frequency patches times the checkpoint's 768-dimensional ViT
width. Runtime validation pins these shape-defining settings and rejects incompatible
architectures, state dictionaries, output shapes, dtypes, NaN, or infinity.

## Usage and conditioning

Augment each four-second Lance split through the same registry path as every other embedding:

```bash
uv sync
synth-setter-add-embeddings \
  lance_uri=/path/to/dataset/train.lance \
  embeddings=[matpac_plus]
```

Run the command separately for each required split. The endpoint modifies only the selected Lance
dataset; dataset cards and completion markers remain finalize-owned.

The generic training and evaluation path selects the resulting sequence with
`conditioning=matpac_plus`. That profile declares `input_shape: [3840, 25]` and uses the existing
`embedpool` encoder. It is therefore specific to four-second source clips; a dataset with another
clip duration must provide a matching generic conditioning shape rather than silently padding or
truncating stored embeddings.
