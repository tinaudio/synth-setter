# Trainer Config

`TrainerConfig` is the pydantic schema for the per-trainer Hydra configs
under `configs/trainer/`. Each YAML there composes onto `default.yaml` and
overrides the subset of `lightning.pytorch.trainer.Trainer` constructor
kwargs that matters for its target accelerator — CPU smoke runs, single-GPU
jobs, DDP, MPS — by selecting `accelerator`, `devices`, and (variant-specific)
`strategy`, `precision`, `min_steps` / `max_steps`. The typed surface here
covers the keys all shipped variants actually set; any other `Trainer` kwarg
passes through via `extra="allow"`, and Lightning's docs are the authority
on the full constructor.

Every shipped YAML under `configs/trainer/` validates against `TrainerConfig`.
The field table below is the source of truth for the typed surface; refer to
[Lightning's `Trainer` API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html)
for the full kwarg list.

::: synth_setter.schemas.trainer_config.TrainerConfig
