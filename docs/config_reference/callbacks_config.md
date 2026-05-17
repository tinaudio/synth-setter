# Callbacks Config

`CallbacksConfig` is the pydantic schema for the Lightning callback configs
under `configs/callbacks/`. Each YAML there either defines a single named
callback (`model_checkpoint.yaml` → `{"model_checkpoint": {"_target_": ...}}`)
or composes several via `defaults:` plus per-callback overrides
(`default.yaml`, `default_surge.yaml`). Once Hydra has finished composing,
`cfg.callbacks` is a flat mapping from callback-name → callback-instance
dict, which `synth_setter.utils.instantiate_callbacks` then iterates over —
`hydra.utils.instantiate` is called only on values that contain `_target_`,
so unrelated keys are silently skipped.

The schema mirrors that shape: `CallbacksConfig` is a
[`RootModel`](https://docs.pydantic.dev/latest/concepts/models/#rootmodel-and-custom-root-types)
wrapping `dict[str, CallbackInstance]`, and each value validates against
`CallbackInstance`. Variant-specific callback kwargs (`monitor`, `dirpath`,
`patience`, `every_n_train_steps`, ...) vary per callback and pass through
via `extra="allow"` on `CallbackInstance` — the constraints on those kwargs
live on the upstream Lightning / project-local callback class signatures.

## CallbacksConfig

::: synth_setter.schemas.callbacks_config.CallbacksConfig

## CallbackInstance

::: synth_setter.schemas.callbacks_config.CallbackInstance
