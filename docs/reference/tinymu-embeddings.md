# TinyMU audio embeddings

`synth-setter-add-embeddings` exposes TinyMU's frozen MATPAC representation as
`embeddings=[tinymu]`. This integration runs only the audio encoder; it does not load TinyMU's
language decoder or implement TinyMU training.

## Source and checkpoint identity

TinyMU currently has no detected license file. synth-setter therefore does not copy, package, or
redistribute TinyMU source. The adapter loads only `src/models/matpac/model.py` from an external
checkout and requires both its Git HEAD and model-file blob to match:

- repository: `ktinubu/TinyMU`
- commit: `eadbe2fc96cbbb5cdb9f91604c7a4e63782e6e7b`

Set `tinymu_source_dir=/path/to/TinyMU` in the Hydra command or export
`TINYMU_SOURCE_DIR=/path/to/TinyMU`. A missing checkout, a different commit, or a modified model
file is an error. `uv sync` installs the measured `timm==0.4.12` dependency as part of the
standard embedding runtime; it does not install TinyMU source.

The default checkpoint is the immutable R2 object:

```text
r2://intermediate-data/tinymu/source/pretrained/AndreasXi/TinyMU/0735fc50bc8b881d687dedccdd48b742927611b3/matpac_plus_as_48_1_map_enconly.pt
```

Its required SHA-256 is
`e8cec6847b2d918c8f77f82d79d90adf7dd82f99e80fa12eb3444f87f24bb998`. The adapter downloads it
through the canonical rclone path into
`${XDG_CACHE_HOME:-$HOME/.cache}/synth-setter/models/embeddings/tinymu-0735fc50bc8b881d687dedccdd48b742927611b3/`,
then verifies the digest before moving it into place. A local `checkpoints.tinymu=/path/to/file`
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
- `tinymu_vec` is the temporal mean, a `FixedSizeList<float32, 3840>` used by the registry's
  cosine IVF_PQ policy.

Measured token counts are 1 for 175 ms, 7 for one second, 25 for four seconds, and 63 for ten
seconds. Output width 3,840 is five frequency patches times the checkpoint's 768-dimensional ViT
width. Runtime validation pins these shape-defining settings and rejects incompatible source,
state dictionaries, output shapes, dtypes, NaN, or infinity.

## Usage and conditioning

For a finalized four-second dataset root:

```bash
uv sync
TINYMU_SOURCE_DIR=/path/to/TinyMU \
  synth-setter-add-embeddings \
  dataset_root_uri=/path/to/dataset \
  embeddings=[tinymu]
```

Root mode requires a finalized `dataset.complete`, then augments every present `train.lance`,
`val.lance`, and `test.lance` split. It records pending work in `dataset.json`, removes the readiness
marker before mutation, checkpoints each embedding commit, and restores the marker only when all
work is complete. Schema version 2 records the MATPAC source commit, checkpoint revision and
SHA-256, producer Git SHA and transform digest, transform and index settings, emitted columns, row
count, Lance version, and index status.
Re-running resumes incomplete work; any changed output identity fails closed. Root augmentation is
single-operator: do not run concurrent augmentation commands against the same root because R2 has
no compare-and-set for card updates.
Use `lance_uri=/path/to/train.lance` only for an intentional single-split operation, which does not
own a dataset-root provenance card.

The generic training and evaluation path selects the resulting sequence with
`conditioning=tinymu`. That profile declares `input_shape: [3840, 25]` and uses the existing
`embedpool` encoder. It is therefore specific to four-second source clips; a dataset with another
clip duration must provide a matching generic conditioning shape rather than silently padding or
truncating stored embeddings.
