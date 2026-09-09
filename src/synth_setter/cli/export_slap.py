"""Hydra entrypoint for SLAP retrieval export."""

from __future__ import annotations

import sys

import hydra
import structlog
from omegaconf import DictConfig

from synth_setter.pipeline.data.export_slap import export_slap
from synth_setter.pipeline.schemas.export_slap_config import ExportSlapConfig
from synth_setter.utils import register_resolvers

register_resolvers()

logger = structlog.get_logger(__name__)


@hydra.main(
    version_base="1.3", config_path="pkg://synth_setter.configs", config_name="export_slap"
)
def _hydra_main(cfg: DictConfig) -> None:
    """Validate the composed request and export all selected splits.

    :param cfg: Hydra-composed exporter configuration.
    """
    try:
        export_slap(ExportSlapConfig.from_hydra_cfg(cfg))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("slap_export_failed", error=str(exc))
        sys.exit(1)


def main() -> None:
    """Entrypoint used by the synth-setter-export-slap console script."""
    _hydra_main()


if __name__ == "__main__":
    main()
