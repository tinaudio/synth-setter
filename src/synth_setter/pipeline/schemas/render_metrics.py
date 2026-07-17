"""Validated metrics emitted by a VST shard renderer subprocess.

Use ``RenderRejectionMetrics.model_validate_json`` when the launcher reads a
renderer sidecar, and ``render_metrics_path`` to address that sidecar.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, NonNegativeInt

RENDER_METRICS_SUFFIX = ".render-metrics.json"


class RenderRejectionMetrics(BaseModel):
    """Counts of sampled renders rejected before shard rows were accepted.

    .. attribute :: model_config

        Pydantic model config sentinel — see ``ConfigDict(...)`` below for active settings.

    .. attribute :: clipped

        Renders rejected for exceeding the amplitude bounds.

    .. attribute :: silent

        Renders rejected for falling below the loudness threshold.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    clipped: NonNegativeInt = 0
    silent: NonNegativeInt = 0


def render_metrics_path(lance_path: Path | str) -> Path:
    """Return the renderer report path adjacent to a Lance shard.

    :param lance_path: Shard dataset whose sibling report is addressed.
    :returns: Sibling report path; the shard need not exist yet.
    """
    return Path(f"{lance_path}{RENDER_METRICS_SUFFIX}")
