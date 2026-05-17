# Data Config

`DataConfig` is the pydantic schema for the per-datamodule Hydra configs under
`configs/data/`. Each YAML there picks one `LightningDataModule` via
`_target_` (e.g. `synth_setter.data.ksin_datamodule.KSinDataModule`,
`synth_setter.data.surge_datamodule.SurgeDataModule`) and supplies the
constructor kwargs that module needs. The kwarg surface varies widely across
modules — synthetic-signal datamodules take seed tuples and `signal_length`,
file-driven ones take `dataset_root` / `stats_file` paths — so the schema
types only `_target_` and accepts the rest via `extra="allow"`. Adding a new
datamodule YAML does not require a schema edit.

Every shipped YAML under `configs/data/` validates against `DataConfig` —
the field table below is the source of truth for the keys Hydra resolves
into `cfg.data` before `hydra.utils.instantiate(cfg.data)` builds the
`LightningDataModule` in `cli/train.py`.

::: synth_setter.schemas.data_config.DataConfig
