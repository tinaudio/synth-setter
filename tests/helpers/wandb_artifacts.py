"""Publish a checkpoint as a real W&B model artifact for resolver round-trip tests.

The ``${wandb:...}`` resolver round-trip tests (``test_eval`` / ``test_train``) need a
genuine artifact in W&B that the resolver can download — a fake stub proves nothing about
the live download path. To keep these test writes out of the production model registry
(``model-<id>`` in the production project) and off the shared 5 GB W&B storage budget,
everything lands in the dedicated ``synth-setter-citest`` project under a per-call-unique
``model-citest-<id>-<token>`` name and is deleted — artifact and run — when the context exits.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from pathlib import Path

# Dedicated test project so round-trip uploads never touch the production model registry.
# The entity is the key's default (read back from the run), so this works under whatever
# account owns the local / CI ``WANDB_API_KEY`` rather than a hardcoded team.
CITEST_PROJECT = "synth-setter-citest"


@contextlib.contextmanager
def publish_checkpoint_artifact(
    ckpt_path: Path, artifact_name: str, run_dir: Path
) -> Iterator[str]:
    """Publish ``ckpt_path`` as ``model.ckpt`` in a W&B model artifact, yielding its resolver ref.

    Logs to :data:`CITEST_PROJECT` under the key's default entity (never the production
    registry) and blocks on ``artifact.wait()`` so the artifact is committed before the
    caller resolves it. The name carries a per-call random suffix so concurrent CI runs never
    collide on ``:latest`` and teardown targets exactly one version. On exit the artifact and
    its run are deleted (best-effort) so each round-trip leaves no W&B storage behind. Requires
    a live ``WANDB_API_KEY`` in the environment — callers gate on it.

    :param ckpt_path: Local Lightning checkpoint embedded into the artifact as ``model.ckpt``.
    :param artifact_name: Base artifact name (e.g. ``model-citest-ffn_full``); a random suffix
        is appended for per-call uniqueness.
    :param run_dir: Directory for the wandb run's local files, kept off the repo tree.
    :yields str: The ``entity/project/name:latest`` ref the resolver consumes — wrap in
        ``${wandb:<ref>}`` to pin it as a ``ckpt_path``.
    """
    import wandb

    unique_name = f"{artifact_name}-{uuid.uuid4().hex[:8]}"
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
    # Read the entity back from the run so refs match whatever account the key owns.
    entity, run_id = run.entity, run.id
    try:
        artifact = wandb.Artifact(name=unique_name, type="model")
        artifact.add_file(str(ckpt_path), name="model.ckpt")
        run.log_artifact(artifact)
        artifact.wait()
        yield f"{entity}/{CITEST_PROJECT}/{unique_name}:latest"
    finally:
        # Teardown is best-effort: a finish() comm error must neither fail the test nor
        # skip deletion, so suppress it and always reach the artifact/run cleanup.
        with contextlib.suppress(Exception):
            run.finish()
        _delete_citest_artifact_and_run(entity, unique_name, run_id)


def _delete_citest_artifact_and_run(entity: str, artifact_name: str, run_id: str) -> None:
    """Best-effort delete of a citest artifact and its run so a round-trip leaves no storage.

    Each deletion is independently suppressed: cleanup runs in a test ``finally`` and must
    never mask the test's own result, and a partial failure (e.g. a transient W&B 5xx) is
    recovered by the next run reusing the same project, not by aborting here.

    :param entity: W&B entity owning :data:`CITEST_PROJECT`.
    :param artifact_name: Unique artifact name whose ``:latest`` version is removed.
    :param run_id: Run id deleted after its artifact.
    """
    import wandb

    api = wandb.Api()
    project = f"{entity}/{CITEST_PROJECT}"
    with contextlib.suppress(Exception):
        api.artifact(f"{project}/{artifact_name}:latest").delete(delete_aliases=True)
    with contextlib.suppress(Exception):
        api.run(f"{project}/{run_id}").delete()
