# Extras Config

`ExtrasConfig` is the pydantic schema for the run-time toggles at
`configs/extras/default.yaml`. The toggles control four small startup
behaviours invoked from the top of `cli/train.py` via
`synth_setter.utils.extras(cfg)` — pretty-printing the composed config,
prompting for tags, silencing warnings, and setting
`torch.set_float32_matmul_precision`. They're read once and don't affect any
downstream model / pipeline state, so the schema is intentionally tiny and
strict (`float32_matmul_precision` is a `Literal` of the three values
PyTorch accepts).

The shipped `extras: default` composition validates against `ExtrasConfig` —
the field table below names every toggle the entrypoint reads.

::: synth_setter.schemas.extras_config.ExtrasConfig
