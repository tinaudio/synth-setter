# Logger Config

`LoggerConfig` is a [`RootModel`](https://docs.pydantic.dev/latest/concepts/models/#rootmodel-and-custom-root-types)
wrapping `dict[str, LoggerInstance]` — the shape of `cfg.logger` after Hydra
composes the YAMLs under `configs/logger/`. Only `_target_` is typed on each
instance; per-logger kwargs pass through via `extra="allow"` (refer to the
upstream logger class for its constructor).

## LoggerConfig

::: synth_setter.schemas.logger_config.LoggerConfig

## LoggerInstance

::: synth_setter.schemas.logger_config.LoggerInstance
