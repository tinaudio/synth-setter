"""Pydantic schema for the path layout under ``configs/paths/``.

Every shipped composition selects ``paths: default``, so the schema models
``configs/paths/default.yaml``. The five string fields are all interpolated
elsewhere (``${paths.output_dir}/checkpoints``, ``${paths.log_dir}/mlflow``)
and a blank value would silently produce a broken run directory; they're
typed as :data:`~synth_setter.schemas._types.NonBlankStr` rather than plain
``str`` so the validator rejects whitespace-only overrides at compose time.
"""

from __future__ import annotations

from pydantic import Field

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["PathsConfig"]


class PathsConfig(StrictAllowExtraModel):  # noqa: DOC601,DOC603
    """Path layout consumed via ``${paths.<name>}`` interpolation across configs.

    These resolve via OmegaConf interpolation in lots of downstream YAMLs
    (logger save-dirs, callback dirpaths, the trainer's ``default_root_dir``,
    etc.), so any blank value here propagates as a broken path into half a
    dozen places. Per-field descriptions live on the ``Field`` definitions
    below.
    """

    root_dir: NonBlankStr = Field(
        description=(
            "Project root. Default ``${oc.env:PROJECT_ROOT}`` so the active "
            "checkout is the root; ``rootutils`` sets ``PROJECT_ROOT`` in the "
            "training entrypoint."
        ),
    )
    data_dir: NonBlankStr = Field(
        description=(
            "Default datamodule data root, ``${paths.root_dir}/data/``. "
            "Individual datamodule configs can override their own root."
        ),
    )
    log_dir: NonBlankStr = Field(
        description=(
            "Logger root, ``${paths.root_dir}/logs/``. Consumed by the "
            "mlflow / tensorboard / wandb logger configs and by Hydra's "
            "per-run ``output_dir`` template."
        ),
    )
    output_dir: NonBlankStr = Field(
        description=(
            "Per-run output dir resolved from ``${hydra:runtime.output_dir}`` — "
            "Hydra generates a unique path per launch using the template in "
            "``configs/hydra/default.yaml``."
        ),
    )
    work_dir: NonBlankStr = Field(
        description=(
            "Hydra's launch-time working directory, ``${hydra:runtime.cwd}``. "
            "Useful for resolving user-supplied relative paths."
        ),
    )
