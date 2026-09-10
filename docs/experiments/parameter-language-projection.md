# Parameter-language projection

Tracking: #3369.

## Offline field metadata

Dataset generation/finalization accepts `param_language_dimension=128` (also 256,
512, or 768). The default `null` leaves existing datasets unchanged. Finalize runs
frozen `google/embeddinggemma-300m` on CPU once per logical field, caches the full
768-dimensional table in its work directory, and publishes `param_language.npz`
before Lance finalization and `dataset.complete`. Nothing is repeated per dataset
row. A replacement finalizer validates and reuses an already-published language
artifact without loading the encoder. Published identity/dimension mismatches
fail closed; corrupt or non-native-width local full-vector caches regenerate.

The model revision is pinned to
`57c266a740f537b4dc058e1b0cda161fd15afa75`. Its Sentence Transformers document
pipeline supplies the prompt and pooling; retained prefixes are L2-normalized
following the [model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card).
EmbeddingGemma is distinct from the existing SA3 T5Gemma conditioner.

The artifact contains float32 `embeddings` shaped `(logical_fields, dimension)`
and strict JSON `metadata`. Descriptions record spec/synth identity, field class,
encoded span, and available native bounds/category labels/shape. Native bounds
are not labelled as physical units. Fingerprints cover ordered descriptions and
tensor bytes. Consumers reject incompatible specs, changed metadata, malformed
vectors, and checksum mismatches. Category vectors remain one logical field.

Model access requires accepting the model license and configuring a Hugging Face
token. Finalization needs network/model-cache access; subsequent artifact loading
does not load a language model. Reuse the same finalizer work directory to derive
another supported dimension without re-encoding. Do not modify a completed run's
spec to retrofit embeddings: create a new opted-in dataset run.

## Training consumers

Use a finalized, locally hydrated dataset containing `param_language.npz`:

```bash
uv run synth-setter-train experiment=surge/flow_simple model/projection=language \
  datamodule.dataset_root=/path/to/dataset datamodule.download_dataset_root_uri=null
uv run synth-setter-train experiment=surge/slap_language \
  datamodule.dataset_root=/path/to/dataset datamodule.download_dataset_root_uri=null
```

The synth selector must match the artifact's identity. These examples retain each
experiment's default synth; supply `synth=...` when the finalized dataset differs.
Flow's dimension knob is `model.vector_field.projection.embedding_dim`; SLAP's is
`model.param_encoder.encoder.projection.embedding_dim`. Both default to 128 and
must match the selected artifact. Each projection also exposes `embedding_path`.

The opt-in configurations install a Lightning callback that initializes the
vectors after checkpoint restoration and before fit, validation, test, or predict.
Numeric forward calls never open files. A standalone projection caller must call
`initialize_embeddings()` before the first fresh forward. Checkpoints carry vectors,
initialization state, field identity, and encoder revision; restoring one does not
require its original metadata file or text model. Use the same projection selector
when restoring. Assignment-matrix visualization remains unsupported.

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest \
  tests/pipeline/data/test_param_language.py \
  tests/integration/test_finalize_param_language.py -q --no-cov
```

The integration test uses the real pretrained encoder, real staged Lance data,
real R2 publication, and the artifact loader. It proves artifact usability, not
improved model quality. Real training coverage also lives in
`tests/integration/test_language_projection_training.py`.

## Research ablations

The paired-seed research runner uses real Surge audio, finalized-format field
metadata, production training modules, checkpoint restoration, and held-out
inference:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ROOT="$PWD/.cache/param-language-research"
uv run python -m scripts.dev.param_language_ablation --root "$ROOT"
for seed in 101 102; do
  for variant in grouped learned random language128 language768; do
    uv run python -m scripts.dev.param_language_ablation --root "$ROOT" \
      --consumer flow --variant "$variant" --seed "$seed" --steps 200
    uv run python -m scripts.dev.param_language_ablation --root "$ROOT" \
      --consumer slap --variant "$variant" --seed "$seed" --steps 1000
  done
done
```

Use a fresh root for an independent matrix; completed run directories are not
silently overwritten. The CLI accepts 100–1,000 optimizer steps. The recorded
matrix reverses variant order for seed 102; no treatments run concurrently on
the GPU. Results, CSV loss histories, resolved configs, and checkpoints remain
under `$ROOT/runs/`. A dataset fingerprint rejects accidental changes between
preparation and training.

Controls share initial numeric heads/backbone weights. All residual variants use
`value + fusion(concat(value, semantic))`, a shared adapter/fusion, and zero final
fusion weights at initialization. `learned` trains independently seeded field
identity vectors; `random` freezes the same initial vectors. Neither consumes
language artifacts. Language variants freeze actual EmbeddingGemma vectors.
Flow output heads stay unrestricted; SLAP keeps unused decoder heads frozen.

The published [smoke-study results](parameter-language-projection-results.md)
are a mechanics check, not evidence of a semantic quality advantage.
