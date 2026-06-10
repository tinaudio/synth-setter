"""Publish a checkpoint as a real W&B model artifact for resolver round-trip tests.

The ``${wandb:...}`` resolver round-trip tests (``test_eval`` / ``test_train``) need a
genuine artifact in W&B that the resolver can download — a fake stub proves nothing about
the live download path. To keep these test writes out of the production model registry
(``model-<id>`` in the production project), everything lands in the dedicated
``synth-setter-citest`` project under ``model-citest-<id>`` names.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

# Dedicated test project so round-trip uploads never touch the production model registry.
# The entity is the key's default (read back from the run), so this works under whatever
# account owns the local / CI ``WANDB_API_KEY`` rather than a hardcoded team.
CITEST_PROJECT = "synth-setter-citest"


def _run_scoped_name(artifact_name: str, run_id: str) -> str:
    """Suffix a collection name with the run id so each run owns its own collection.

    A fixed shared name can wedge: deleting its versions leaves the collection in W&B's
    ``PENDING_DELETION`` state, which rejects new manifests (HTTP 400) until async GC
    finalizes, and concurrent runs race on its ``:latest`` alias. A run-scoped name avoids
    both — a brand-new collection per run, never colliding with a half-deleted one.

    :param artifact_name: Base name, e.g. ``model-citest-ffn_full-resume``.
    :param run_id: The W&B run id, unique per run.
    :returns: ``<artifact_name>-<run_id>``.
    """
    return f"{artifact_name}-{run_id}"


@contextlib.contextmanager
def published_checkpoint_artifact(
    ckpt_path: Path, artifact_name: str, run_dir: Path
) -> Iterator[str]:
    """Publish ``ckpt_path`` as ``model.ckpt`` in a W&B model artifact, yielding its resolver ref.

    Logs to :data:`CITEST_PROJECT` under the key's default entity (never the production
    registry) under a run-scoped name (see :func:`_run_scoped_name`), and blocks on
    ``artifact.wait()`` so the artifact is committed before the body resolves it. On exit
    the published version is deleted best-effort so its checkpoint bytes do not accumulate
    in the scratch project (an empty collection shell is left behind, which stores nothing).
    Requires a live ``WANDB_API_KEY`` in the environment — callers gate on it.

    :param ckpt_path: Local Lightning checkpoint embedded into the artifact as ``model.ckpt``.
    :param artifact_name: Base artifact name, e.g. ``model-citest-ffn_full``.
    :param run_dir: Directory for the wandb run's local files, kept off the repo tree.
    :yields: The ``entity/project/name:latest`` ref the resolver consumes — wrap in
        ``${wandb:<ref>}`` to pin it as a ``ckpt_path``.
    :ytype: str
    """
    import wandb

    # Pin a short host: W&B rejects a run whose machine hostname exceeds 64 chars
    # ("CommError: 64 limit exceeded for Host"), and self-hosted CI runners (e.g. the
    # MPS box) have hostnames well over that. None (the default) would record
    # socket.gethostname(), which is what trips the limit.
    run = wandb.init(
        project=CITEST_PROJECT,
        job_type="ckpt-roundtrip-smoke",
        dir=str(run_dir),
        settings=wandb.Settings(host="synth-setter-ci"),
    )
    name = _run_scoped_name(artifact_name, run.id)
    # Read the entity back from the run so the ref matches whatever account the key owns.
    ref = f"{run.entity}/{CITEST_PROJECT}/{name}:latest"
    try:
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(str(ckpt_path), name="model.ckpt")
        run.log_artifact(artifact)
        artifact.wait()
    finally:
        run.finish()
    try:
        yield ref
    finally:
        _delete_artifact_best_effort(ref)


def _delete_artifact_best_effort(ref: str) -> None:
    """Delete the published artifact version, never raising — cleanup must not fail the test.

    Passes ``delete_aliases=True`` because the version carries the ``:latest`` alias, which
    W&B otherwise refuses to delete. Any failure is logged and swallowed.

    :param ref: The ``entity/project/name:latest`` ref to delete.
    """
    import wandb

    try:
        wandb.Api().artifact(ref).delete(delete_aliases=True)
    except Exception:  # noqa: BLE001 - best-effort scratch cleanup, never fail the test
        logger.warning("failed to delete citest artifact %s", ref, exc_info=True)
