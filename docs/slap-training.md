# SLAP training and checkpoint compatibility

Use `experiment=surge/slap_ast_audio_vst_ff_param` for new SLAP runs.
The matching model composes `model/encoder/slap_ast_audio` into
`model.audio_encoder.encoder` and `model/encoder/vst_ff_param` into
`model.param_encoder.encoder`. The audio backbone retains its single AST
conditioning token and flattening step. Both Siamese projectors and predictors
operate in a 512-dimensional shared embedding space.

The parameter backbone uses feed-forward parameter-to-token projection followed
by the shared AST projection-head implementation. The shared flow, MLP, and
parameter-token components remain available to other models.

## Existing configurations and checkpoints

`SLAPModule` accepts the deprecated `text_encoder` constructor/config key as an
alias for `param_encoder`, preserving the supplied arm unchanged. Supply exactly
one name; passing both raises `ValueError`, even when they reference the same
object. Positional arm arguments retain their meaning. New configurations use
only `param_encoder`.

Checkpoint loading migrates `text_encoder.*` and `text_ema.*` state prefixes to
`param_encoder.*` and `param_ema.*`, including compiled arm prefixes. Target
predictor state is discarded because target arms expose projections only.
Legacy `text_encoder` checkpoint hyperparameters are accepted by the constructor
and renamed by the load hook. Coexisting canonical and legacy hyperparameter or
state namespaces are rejected rather than choosing one silently.

Arm modules are not saved in new checkpoint hyperparameters. When an existing
checkpoint does not contain its modules, instantiate the **original saved model
configuration** and supply those original dependencies to
`SLAPModule.load_from_checkpoint`, or use that configuration for Lightning
checkpoint resume. The canonical experiment is not a substitute for an old MLP
or parameter-token-transformer architecture. Strict loading must succeed; do not
disable it to force old weights into the canonical model. Complete saved Hydra
configurations using `text_encoder` remain valid, but old experiment/model group
names are not shipped as selectable alternatives.

`text_encoder` and `text_ema` remain read-only compatibility accessors for export
consumers. They reference the canonical arms without registering duplicate
checkpoint state. EMA exports continue to consume the projector output, not the
online predictor. Alias removal requires coordinating the export/checkpoint
consumers tracked in #3204, #3207, and #3208.

## Held-out retrieval

The canonical experiment enables `model.retrieval_eval` and
`datamodule.eval_sample_ids`. Validation and checkpoint-reloaded testing log
`retrieval/{val,test}/` metrics over each loader's complete observed gallery:
bidirectional Recall@1/5/10 and MRR, gallery size, normalized embedding variance,
matched cosine, and cyclic mismatched cosine. `loss/{val,test}/` remains the
optimization diagnostic. Retrieval uses **online predictor outputs**, not EMA
projections or ANN exports.

Predictions and IDs are detached to CPU during the existing loss forward, then
gathered across ranks at epoch end. Score matrices are query-chunked. Equal
cosine scores receive expected Recall/MRR under uniform tie-breaking; collapsed
vectors therefore cannot obtain perfect retrieval from row-order ties. Sanity
validation produces no retrieval metrics, and each evaluation loop clears its
observations. Trainer limits restrict the observed gallery; use unrestricted
validation/test loaders for full-split metrics.

IDs are transient int64 row offsets within a pinned Lance split version, not
new stored columns. Each loader has an independent ID namespace. Distributed
sampler padding and repeat-first rows retain the first rank/batch observation.
Repeated normalized predictions must be within L2 distance `0.01` (unit-vector
cosine distance `5e-5`), allowing bf16 batch-shape roundoff; larger conflicts
raise an error. Fake random datasets cannot provide these IDs, and OT-reordered
batches reject them. Unrelated loader defaults are unchanged.

Empty galleries have no retrieval scores; singleton galleries omit mismatched
cosine. Learning quality and non-collapse still require representative held-out
evaluation, not just successful integration tests.

## Verification scope

The fixed-batch manual-optimizer test checks loss reduction with frozen target
arms. It does not establish full memorization, non-collapse, or a mathematical
nonzero cosine-loss floor. Production learning verification follows integration
of the parameter backbone, canonical configuration, and retrieval changes.
