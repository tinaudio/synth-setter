# Model Config

`ModelConfig` is the pydantic schema for the YAMLs under `src/synth_setter/configs/model/`.
The typed surface is `_target_` + optimizer + optional scheduler + `compile`;
variant-specific kwargs pass through via `extra="allow"`.

## ModelConfig

::: synth_setter.schemas.model_config.ModelConfig

## OptimizerConfig

::: synth_setter.schemas.model_config.OptimizerConfig

## SchedulerConfig

::: synth_setter.schemas.model_config.SchedulerConfig
