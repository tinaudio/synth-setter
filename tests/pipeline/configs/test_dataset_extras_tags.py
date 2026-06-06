"""Pin ``dataset.yaml``'s ``extras`` + ``tags`` parity with ``train.yaml`` / ``eval.yaml``.

Composing ``dataset.yaml`` must surface the ``extras: default`` group (so the
entrypoints' ``extras(cfg)`` call has a config to act on) and a non-empty
top-level ``tags`` (so ``enforce_tags`` never prompts/raises on the default
compose). ``tags`` is not a ``DatasetSpec`` field, so the spec round-trip must
still succeed with it masked out.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_module

from synth_setter.pipeline.schemas.spec import DatasetSpec

# Local checkout root — pins ``cfg.paths.*`` so the composed
# ``${oc.env:PROJECT_ROOT}`` / ``${hydra:runtime.output_dir}`` interpolations
# resolve to a real on-disk path during unit tests (mirrors test_experiment_yamls).
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_dataset_compose_enforce_tags_true_and_default_tags() -> None:
    """Composing ``dataset.yaml`` carries ``extras.enforce_tags`` and the default ``tags``.

    Parity with ``train.yaml`` / ``eval.yaml``: ``extras: default`` sets
    ``enforce_tags: True`` and a non-empty ``tags`` keeps that path from
    prompting/raising on a default compose.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=["experiment=generate_dataset/smoke-shard"])

    assert cfg.extras.enforce_tags is True
    assert cfg.tags == ["dev", "generate_dataset"]


def test_dataset_compose_with_tags_still_builds_dataset_spec() -> None:
    """``from_hydra_cfg`` masks ``tags`` out so the spec still builds with it present."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(config_name="dataset", overrides=["experiment=generate_dataset/smoke-shard"])
    cfg.paths.root_dir = str(REPO_ROOT)
    cfg.paths.output_dir = str(REPO_ROOT)
    cfg.paths.work_dir = str(REPO_ROOT)

    spec = DatasetSpec.from_hydra_cfg(cfg)

    assert isinstance(spec, DatasetSpec)
    assert not hasattr(spec, "tags")
