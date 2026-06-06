"""Publish a checkpoint as a real W&B model artifact for resolver round-trip tests.

The ``${wandb:...}`` resolver round-trip tests (``test_eval`` / ``test_train``) need a
genuine artifact in W&B that the resolver can download — a fake stub proves nothing about
the live download path. To keep these test writes out of the production model registry
(``model-<id>`` in the production project), everything lands in the dedicated
``synth-setter-citest`` project under ``model-citest-<id>`` names.
"""

from __future__ import annotations

from pathlib import Path

# Dedicated test project so round-trip uploads never touch the production model registry.
# The entity is the key's default (read back from the run), so this works under whatever
# account owns the local / CI ``WANDB_API_KEY`` rather than a hardcoded team.
CITEST_PROJECT = "synth-setter-citest"


def publish_checkpoint_artifact(ckpt_path: Path, artifact_name: str, run_dir: Path) -> str:
    """Upload ``ckpt_path`` as ``model.ckpt`` in a W&B model artifact, returning its resolver ref.

    Logs to :data:`CITEST_PROJECT` under the key's default entity (never the production
    registry) and blocks on ``artifact.wait()`` so the artifact is committed before the
    caller resolves it. Requires a live ``WANDB_API_KEY`` in the environment — callers gate on it.

    :param ckpt_path: Local Lightning checkpoint embedded into the artifact as ``model.ckpt``.
    :param artifact_name: Artifact name, e.g. ``model-citest-ffn_full``.
    :param run_dir: Directory for the wandb run's local files, kept off the repo tree.
    :returns: The ``entity/project/name:latest`` ref the resolver consumes — wrap in
        ``${wandb:<ref>}`` to pin it as a ``ckpt_path``.
    """
    import wandb

    run = wandb.init(
        project=CITEST_PROJECT,
        job_type="ckpt-roundtrip-smoke",
        dir=str(run_dir),
    )
    try:
        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(str(ckpt_path), name="model.ckpt")
        run.log_artifact(artifact)
        artifact.wait()
        # Read the entity back from the run so the ref matches whatever account the key owns.
        ref = f"{run.entity}/{CITEST_PROJECT}/{artifact_name}:latest"
    finally:
        run.finish()
    return ref
