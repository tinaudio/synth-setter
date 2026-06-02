# Datamodule Config

`DataModuleConfig` is the pydantic schema for the YAMLs under `src/synth_setter/configs/datamodule/`.
Only `_target_` is typed; datamodule constructor kwargs vary per module
and pass through via `extra="allow"`.

::: synth_setter.schemas.datamodule_config.DataModuleConfig
