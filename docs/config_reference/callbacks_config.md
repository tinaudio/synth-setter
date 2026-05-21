# Callbacks Config

`CallbacksConfig` is a [`RootModel`](https://docs.pydantic.dev/latest/concepts/models/#rootmodel-and-custom-root-types)
wrapping `dict[str, CallbackInstance]` — the shape of `cfg.callbacks` after
Hydra composes the YAMLs under `src/synth_setter/configs/callbacks/`. Only `_target_` is
typed on each instance; per-callback kwargs pass through via `extra="allow"`.

## CallbacksConfig

::: synth_setter.schemas.callbacks_config.CallbacksConfig

## CallbackInstance

::: synth_setter.schemas.callbacks_config.CallbackInstance
