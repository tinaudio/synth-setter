# Model Config

`ModelConfig` is the pydantic schema for the per-model Hydra configs under
`configs/model/`. Each YAML file there picks one
`LightningModule` (`_target_`), a partial torch optimizer, an optional partial
LR scheduler, and the `compile` flag — everything else is variant-specific
and varies across model modules. The schema documents the common surface area
and accepts variant fields via `extra="allow"`, so adding a new model YAML
does not require a schema edit.

Every shipped YAML under `configs/model/` validates against `ModelConfig` —
the field tables below are the source of truth for the keys Hydra resolves
into `cfg.model` before `hydra.utils.instantiate(cfg.model)` builds the
LightningModule in `cli/train.py`.

## ModelConfig

::: synth_setter.schemas.model_config.ModelConfig

## OptimizerConfig

::: synth_setter.schemas.model_config.OptimizerConfig

## SchedulerConfig

::: synth_setter.schemas.model_config.SchedulerConfig
