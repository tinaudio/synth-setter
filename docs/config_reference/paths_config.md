# Paths Config

`PathsConfig` is the pydantic schema for the path-layout YAML at
`configs/paths/default.yaml`. The five string fields are interpolated all
over the rest of the config tree (`${paths.output_dir}/checkpoints`,
`${paths.log_dir}/mlflow`, the trainer's `default_root_dir`, every logger's
`save_dir`), so a blank override here propagates as a broken path into half a
dozen places. The schema types every field as a non-blank string and the
validator rejects whitespace-only overrides at compose time.

The shipped `paths: default` composition validates against `PathsConfig` —
the field table below is the source of truth for the keys downstream YAMLs
may safely interpolate via `${paths.<name>}`.

::: synth_setter.schemas.paths_config.PathsConfig
