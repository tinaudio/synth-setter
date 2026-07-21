#!/usr/bin/env python
"""Hydrate the CLAP + music2latent encoder weights from the project R2 mirror.

Fresh venvs and CI pods otherwise pull ~850 MB from Hugging Face on first use:
``laion/clap-htsat-unfused`` lands in the per-machine HF cache, while
``music2latent`` downloads its checkpoint into its own ``site-packages`` tree —
re-fetched for every new worktree venv. This CLI copies both from
``r2://intermediate-data/models/encoders`` instead; ``--mirror`` seeds that
prefix from local sources. Both directions ride rclone ``--checksum``, so
re-runs are idempotent.

CLI: ``synth-setter-hydrate-encoders [--mirror]``.
"""

from importlib.util import find_spec
from pathlib import Path

import click
import structlog

from synth_setter.pipeline.r2_io import (
    download_dir_no_overwrite,
    download_to_path,
    upload_dir,
    upload_to_uri,
)

logger = structlog.get_logger(__name__)

# Duplicated from add_embeddings to keep this module import-light (no pyarrow chain).
DEFAULT_CLAP_CHECKPOINT = "laion/clap-htsat-unfused"
DEFAULT_ENCODERS_URI = "r2://intermediate-data/models/encoders"
CLAP_PREFIX = "clap-htsat-unfused"
M2L_PREFIX = "music2latent"
M2L_FILENAME = "music2latent.pt"
# Minimum flat-snapshot contents for ``transformers`` to load the checkpoint offline;
# the upstream main revision ships pytorch_model.bin, converted mirrors safetensors.
CLAP_REQUIRED_FILES = ("config.json", "preprocessor_config.json")
CLAP_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _clap_snapshot_complete(directory: Path) -> bool:
    """Return whether ``directory`` holds a loadable flat CLAP snapshot.

    :param directory: Candidate snapshot directory.
    :returns: ``True`` when every config file and at least one weights file exist.
    """
    return all((directory / name).is_file() for name in CLAP_REQUIRED_FILES) and any(
        (directory / name).is_file() for name in CLAP_WEIGHT_FILES
    )


def default_clap_dir() -> Path:
    """Return the default local directory for the hydrated CLAP snapshot.

    :returns: Per-user cache directory outside any venv, so worktrees share it.
    """
    return Path("~/.cache/synth-setter/encoders").expanduser() / CLAP_PREFIX


def music2latent_weights_path() -> Path:
    """Return the checkpoint path the installed ``music2latent`` package loads from.

    :returns: ``<package>/models/music2latent.pt`` for the active environment.
    :raises ModuleNotFoundError: If ``music2latent`` is not installed.
    """
    spec = find_spec("music2latent")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("music2latent is not installed in this environment")
    return Path(spec.origin).parent / "models" / M2L_FILENAME


def hydrated_clap_checkpoint(clap_dir: Path | None = None) -> str:
    """Return the hydrated CLAP snapshot dir, or the HF checkpoint name as fallback.

    :param clap_dir: Snapshot directory to probe; ``None`` uses :func:`default_clap_dir`.
    :returns: A ``transformers``-loadable checkpoint reference.
    """
    directory = default_clap_dir() if clap_dir is None else clap_dir
    if _clap_snapshot_complete(directory):
        return str(directory)
    return DEFAULT_CLAP_CHECKPOINT


def hydrate(encoders_uri: str, *, clap_dir: Path, m2l_path: Path) -> None:
    """Copy both encoders from the R2 mirror to their local targets.

    :param encoders_uri: ``r2://`` prefix holding the mirrored encoder layout.
    :param clap_dir: Local directory receiving the flat CLAP snapshot.
    :param m2l_path: Local ``music2latent.pt`` destination; skipped if present.
    """
    logger.info("hydrating_clap", source=f"{encoders_uri}/{CLAP_PREFIX}", target=str(clap_dir))
    download_dir_no_overwrite(f"{encoders_uri}/{CLAP_PREFIX}", clap_dir)
    if m2l_path.is_file():
        logger.info("m2l_weights_already_present", target=str(m2l_path))
    else:
        m2l_uri = f"{encoders_uri}/{M2L_PREFIX}/{M2L_FILENAME}"
        logger.info("hydrating_m2l", source=m2l_uri, target=str(m2l_path))
        download_to_path(m2l_uri, m2l_path)


def mirror(encoders_uri: str, *, clap_source: Path, m2l_source: Path) -> None:
    """Seed the R2 mirror from local encoder sources.

    :param encoders_uri: ``r2://`` prefix receiving the mirrored encoder layout.
    :param clap_source: Flat CLAP snapshot directory (a materialized HF snapshot).
    :param m2l_source: Local ``music2latent.pt`` weights file.
    :raises FileNotFoundError: If ``m2l_source`` or a required CLAP file is absent.
    """
    if not _clap_snapshot_complete(clap_source):
        raise FileNotFoundError(
            f"CLAP source {clap_source} is not a complete snapshot: needs "
            f"{', '.join(CLAP_REQUIRED_FILES)} and one of {', '.join(CLAP_WEIGHT_FILES)}"
        )
    if not m2l_source.is_file():
        raise FileNotFoundError(f"m2l weights not found at {m2l_source}")
    logger.info("mirroring_clap", source=str(clap_source), target=f"{encoders_uri}/{CLAP_PREFIX}")
    upload_dir(clap_source, f"{encoders_uri}/{CLAP_PREFIX}")
    logger.info("mirroring_m2l", source=str(m2l_source), target=f"{encoders_uri}/{M2L_PREFIX}")
    upload_to_uri(m2l_source, f"{encoders_uri}/{M2L_PREFIX}/{M2L_FILENAME}")


@click.command()
@click.option(
    "--encoders-uri",
    default=DEFAULT_ENCODERS_URI,
    show_default=True,
    help="r2:// prefix holding the mirrored encoder layout.",
)
@click.option(
    "--mirror",
    "mirror_mode",
    is_flag=True,
    help="Upload local sources to the R2 mirror instead of hydrating from it.",
)
@click.option(
    "--clap-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Hydration target for the CLAP snapshot [default: ~/.cache/synth-setter/encoders].",
)
@click.option(
    "--m2l-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Hydration target for music2latent.pt [default: the installed package's models/].",
)
@click.option(
    "--clap-source",
    type=click.Path(path_type=Path),
    default=None,
    help="Mirror-mode CLAP snapshot source directory [default: --clap-dir resolution].",
)
@click.option(
    "--m2l-source",
    type=click.Path(path_type=Path),
    default=None,
    help="Mirror-mode music2latent.pt source [default: --m2l-path resolution].",
)
def main(
    encoders_uri: str,
    mirror_mode: bool,
    clap_dir: Path | None,
    m2l_path: Path | None,
    clap_source: Path | None,
    m2l_source: Path | None,
) -> None:
    """Hydrate encoder weights from the R2 mirror (default) or seed it (``--mirror``).

    :param encoders_uri: ``r2://`` prefix holding the mirrored encoder layout.
    :param mirror_mode: Upload local sources instead of hydrating.
    :param clap_dir: Local CLAP snapshot directory override.
    :param m2l_path: Local music2latent weights path override.
    :param clap_source: Mirror-mode CLAP source directory override.
    :param m2l_source: Mirror-mode music2latent weights source override.
    :raises SystemExit: With code 1 when a source or target cannot be resolved.
    """
    try:
        resolved_clap = default_clap_dir() if clap_dir is None else clap_dir
        resolved_m2l = music2latent_weights_path() if m2l_path is None else m2l_path
        if mirror_mode:
            mirror(
                encoders_uri,
                clap_source=resolved_clap if clap_source is None else clap_source,
                m2l_source=resolved_m2l if m2l_source is None else m2l_source,
            )
        else:
            hydrate(encoders_uri, clap_dir=resolved_clap, m2l_path=resolved_m2l)
    except (FileNotFoundError, ModuleNotFoundError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
