# Paths Config

`PathsConfig` is the pydantic schema for `configs/paths/default.yaml`. Every
field is typed `NonBlankStr` because the values are interpolated as
`${paths.<name>}` across many downstream YAMLs (logger save-dirs, callback
dirpaths, the trainer's `default_root_dir`), so a whitespace-only override
would propagate as a broken path into several places.

::: synth_setter.schemas.paths_config.PathsConfig
