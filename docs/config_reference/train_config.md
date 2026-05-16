# Training Config

`TrainConfig` is the pydantic schema for the top-level training configuration
that `synth-setter-train` (the `@hydra.main` entrypoint in
`src/synth_setter/cli/train.py`) consumes. It documents the scalar fields the
entrypoint reads directly — `task_name`, `tags`, `train`, `test`,
`ckpt_path`, `seed`, `optimized_metric`, `watch_gradients` — while letting
Hydra-composed subtrees (`data`, `model`, `trainer`, `callbacks`, `logger`,
`paths`, `extras`, `hydra`, ...) pass through unchanged. Those subtrees have
their own composition groups under `configs/` and are validated by
`hydra.utils.instantiate` at call time.

Every shipped composition of `configs/train.yaml` validates against this
schema — the field table below is the source of truth for what the
entrypoint expects.

::: synth_setter.schemas.train_config.TrainConfig
