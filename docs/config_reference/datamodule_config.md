# Datamodule Config

The YAMLs under `src/synth_setter/configs/datamodule/` configure the Lightning
`LightningDataModule` each run trains on. Only `_target_` is load-bearing at the
config boundary: it is the fully-qualified datamodule class path that
`hydra.utils.instantiate(cfg.datamodule)` resolves in `cli/train.py`. Hydra
itself enforces `_target_`'s presence — a missing or malformed target fails
instantiation — so this group has no separate Pydantic schema.

Datamodule constructor kwargs vary per module (`SurgeDataModule` is path-driven,
`KSinDataModule` is synthetic) and pass straight through to the constructor; see
each datamodule class for its accepted arguments.
