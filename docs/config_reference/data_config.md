# Data Config

`DataConfig` is the pydantic schema for the YAMLs under `configs/data/`.
Only `_target_` is typed; datamodule constructor kwargs vary per module
and pass through via `extra="allow"`.

::: synth_setter.schemas.data_config.DataConfig
