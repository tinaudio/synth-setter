# Logger Config

`LoggerConfig` is the pydantic schema for the Lightning logger configs under
`configs/logger/`. The layout mirrors callbacks: each YAML either defines a
single named logger (`wandb.yaml`, `tensorboard.yaml`, `mlflow.yaml`, ...)
or composes several via `defaults:` (`many_loggers.yaml` pulls in csv +
tensorboard + wandb). Composed `cfg.logger` is a flat mapping
name → logger-instance dict, iterated by
`synth_setter.utils.instantiate_loggers`.

The schema mirrors that shape: `LoggerConfig` is a
[`RootModel`](https://docs.pydantic.dev/latest/concepts/models/#rootmodel-and-custom-root-types)
wrapping `dict[str, LoggerInstance]`, and each value validates against
`LoggerInstance`. Logger-class kwargs (`api_key`, `project`, `save_dir`,
`tags`, ...) vary per logger and pass through via `extra="allow"` on
`LoggerInstance` — refer to each upstream logger class for its constructor
surface.

## LoggerConfig

::: synth_setter.schemas.logger_config.LoggerConfig

## LoggerInstance

::: synth_setter.schemas.logger_config.LoggerInstance
