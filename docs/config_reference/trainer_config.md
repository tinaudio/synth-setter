# Trainer Config

`TrainerConfig` is the pydantic schema for the YAMLs under
`configs/trainer/`. The typed surface covers the keys all shipped variants
set; any other `Trainer` kwarg passes through via `extra="allow"`. See
[Lightning's `Trainer` API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html)
for the full constructor.

::: synth_setter.schemas.trainer_config.TrainerConfig
