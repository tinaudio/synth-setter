"""Shared ``--keep-local`` plumbing for the generate / finalize CLIs.

``--keep-local`` is an operator-side dev-mode switch that redirects rclone's
``r2:`` remote to a local directory via ``RCLONE_CONFIG_R2_TYPE=alias`` and
``RCLONE_CONFIG_R2_REMOTE=<path>``. All R2 access in this codebase resolves
through rclone, so the redirect is invisible to the rest of the pipeline —
generate's shard uploads, finalize's shard downloads, the spec upload, and
the ``object_size`` skip probes all transparently land on (and read from)
the local filesystem. The flag lives outside Hydra so it does not appear
in the spec snapshot uploaded to "R2".
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

KEEP_LOCAL_FLAG = "--keep-local"


def split_keep_local(argv: list[str]) -> tuple[bool, list[str]]:
    """Pop ``--keep-local`` out of ``argv``; return ``(keep_local, remaining)``.

    Boolean-only flag — ``--keep-local=true`` / ``--keep-local true`` and any
    other suffix-bearing form is rejected so operators can't accidentally
    pass a value rclone would never see.

    :param argv: CLI overrides (``sys.argv[1:]``-shaped).
    :returns: ``(True, [...])`` if ``--keep-local`` was present exactly once;
        ``(False, argv)`` otherwise.
    :raises ValueError: ``--keep-local`` appeared with a ``=value`` suffix,
        or appeared more than once.
    """
    remaining: list[str] = []
    seen = False
    for token in argv:
        if token == KEEP_LOCAL_FLAG:
            if seen:
                raise ValueError(f"{KEEP_LOCAL_FLAG} passed more than once")
            seen = True
            continue
        if token.startswith(f"{KEEP_LOCAL_FLAG}="):
            raise ValueError(
                f"{KEEP_LOCAL_FLAG} is a boolean flag; got {token!r}. "
                f"Use {KEEP_LOCAL_FLAG} alone, or drop it."
            )
        remaining.append(token)
    return seen, remaining


def redirect_r2_to_local(local_root: Path) -> None:
    """Point rclone's ``r2:`` remote at ``local_root`` for the rest of the process.

    Materializes ``local_root`` (rclone's ``alias`` backend resolves paths
    lazily, but the parent dirs must exist when the first upload lands or
    rclone errors out) and sets ``RCLONE_CONFIG_R2_TYPE=alias`` +
    ``RCLONE_CONFIG_R2_REMOTE=<local_root>`` in process env. A URI of the
    form ``r2:<bucket>/<key>`` then materializes at
    ``<local_root>/<bucket>/<key>``.

    Idempotent — safe to call repeatedly with the same root, or after
    a previous keep-local session left the env vars set.

    :param local_root: Absolute filesystem path that becomes the fake-R2 root.
    """
    local_root.mkdir(parents=True, exist_ok=True)
    os.environ["RCLONE_CONFIG_R2_TYPE"] = "alias"
    os.environ["RCLONE_CONFIG_R2_REMOTE"] = str(local_root)
    logger.info(f"keep-local: R2 redirected to {local_root}")
