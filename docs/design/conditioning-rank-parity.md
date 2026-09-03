# Conditioning-rank parity

`AudioSpectrogramTransformer` has always emitted one conditioning slot per vector-field layer via
`n_conditioning_outputs`, while the cached-embedding encoders emitted a single shared vector.
`EmbeddingPool` and `VectorProjection` now take the same `n_conditioning_outputs` argument, so a
cached encoder can supply layer-specific conditioning too.

## Contract

The Python APIs default `n_conditioning_outputs` to `1`, which returns rank-2 `[batch, d_model]`
and preserves the pooled parameter shapes and state-dict keys. Embedding-conditioning Hydra configs
default to `${oc.select:model.vector_field.num_layers,1}`: flow models emit one slot per field layer, while
models without a `vector_field` retain the single pooled output. Any value above one returns
`[batch, slots, d_model]`.

Both flow backbones already consumed rank-3 conditioning before this change: they repeat the time
encoding across the slot axis and index `z[:, i]` per layer. Neither field is modified here.

## CFG nulls

The fields keep one shared `cfg_dropout_token`. Dropout broadcasts it across the slot axis, so
every layer of a dropped row reads the same null — which is what every trained-from-scratch AST run
has used, at eight slots against one null. Per-layer nulls are a separate hypothesis; the
slot-collapse cosine logged during training is the evidence that would justify them.

## Configuration

Cached-embedding profiles such as `clap` and `same_l` use layerwise outputs automatically when the
model defines `vector_field.num_layers`. Online profiles such as `clap_online` and `same_s_online`
derive the same output count for their frozen encoder head.

Set `model.encoder.n_conditioning_outputs=1` for cached profiles, or
`model.encoder.head.n_conditioning_outputs=1` for online profiles, when loading a checkpoint trained
with the pooled encoder shape or when explicitly selecting shared conditioning.

Both supported fields index one slot per layer: fewer slots fail during forward indexing, while
extra slots are accepted but unused. There is no encoder/field slot-count validation.
