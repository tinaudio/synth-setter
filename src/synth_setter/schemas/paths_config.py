"""Pydantic schema for ``configs/paths/default.yaml``.

Fields are typed ``NonBlankStr`` so a whitespace-only override is caught
here rather than propagating as a broken path through every
``${paths.<name>}`` interpolation downstream.
"""

from __future__ import annotations

from pydantic import Field

from synth_setter.schemas._types import NonBlankStr, StrictAllowExtraModel

__all__ = ["PathsConfig"]


class PathsConfig(StrictAllowExtraModel):
    """Path layout consumed via ``${paths.<name>}`` interpolation across configs.

    .. attribute :: root_dir

        Project root.

    .. attribute :: data_dir

        Default datamodule data root, ``${paths.root_dir}/data/``.

    .. attribute :: log_dir

        Logger root, ``${paths.root_dir}/logs/``.

    .. attribute :: output_dir

        Per-run output dir resolved from ``${hydra:runtime.output_dir}``.

    .. attribute :: work_dir

        Hydra's launch-time working directory, ``${hydra:runtime.cwd}``.
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
