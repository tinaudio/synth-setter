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

## Verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest \
  tests/pipeline/data/test_param_language.py \
  tests/integration/test_finalize_param_language.py -q --no-cov
```

The integration test uses the real pretrained encoder, real staged Lance data,
real R2 publication, and the artifact loader. It proves artifact usability, not
improved model quality. Projection and training ablations are subsequent PRs.
