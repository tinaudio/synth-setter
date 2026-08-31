# Conditioning-rank parity

`AudioSpectrogramTransformer` has always emitted one conditioning slot per vector-field layer via
`n_conditioning_outputs`, while the cached-embedding encoders emitted a single shared vector.
`EmbeddingPool` and `VectorProjection` now take the same `n_conditioning_outputs` argument, so a
cached encoder can supply layer-specific conditioning too.

## Contract

`n_conditioning_outputs` defaults to `1`, which returns rank-2 `[batch, d_model]` and is bitwise
identical to the previous pooled behaviour — same parameter shapes, same state-dict keys, so
existing checkpoints load unadapted. Any value above one returns `[batch, slots, d_model]`.

Both flow backbones already consumed rank-3 conditioning before this change: they repeat the time
encoding across the slot axis and index `z[:, i]` per layer. Neither field is modified here.

## CFG nulls

The fields keep one shared `cfg_dropout_token`. Dropout broadcasts it across the slot axis, so
every layer of a dropped row reads the same null — which is what every trained-from-scratch AST run
has used, at eight slots against one null. Per-layer nulls are a separate hypothesis; the
slot-collapse cosine logged during training is the evidence that would justify them.

## Enabling it

Layerwise conditioning is opt-in per experiment, not a default. Cached-embedding profiles such as
`clap` and `same_l` configure the projection or pooling module directly:

```yaml
model:
  encoder:
    n_conditioning_outputs: ${model.vector_field.num_layers}
```

Online profiles such as `clap_online` and `same_s_online` wrap that module as the frozen encoder's
head:

```yaml
model:
  encoder:
    head:
      n_conditioning_outputs: ${model.vector_field.num_layers}
```

Configure `n_conditioning_outputs` to the consuming field's `num_layers`. Both supported fields
index one slot per layer: fewer slots fail during forward indexing, while extra slots are accepted
but unused. This PR adds no encoder/field slot-count validation.
