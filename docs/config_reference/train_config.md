# Training Config

`TrainConfig` documents the scalar fields the `synth-setter-train` entrypoint
reads off `src/synth_setter/configs/train.yaml` directly. Hydra-composed subtrees (`data`,
`model`, `trainer`, ...) are accepted via `extra="allow"` and validated by
their own schemas — see the sibling pages.

For the opt-in `estimate_normalization_stats` preprocessing step and explicit
frontend mean/std settings, see [online AST normalization](../reference/ast-normalization.md).

::: synth_setter.schemas.train_config.TrainConfig
