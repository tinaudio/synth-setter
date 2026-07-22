"""Config-layer tests that compose the real ``add_embeddings.yaml`` Hydra config.

The endpoint (``add_embeddings.main``) composes its shipped YAML and validates
it into :class:`AddEmbeddingsConfig`; a break in the ``defaults`` / ``paths`` /
``hydra`` composition would pass every unit test and only fail in production.
These bare-compose tests pin the knobs the endpoint reads and prove
``from_hydra_cfg`` validates the composed cfg, plus one live ``main()`` run of
the Hydra shell (success + failure→exit-1) with the real encoders swapped for
fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lance
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from synth_setter.data.vst.shapes import AUDIO_FIELD, CLAP_FIELD, M2L_FIELD
from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    DEFAULT_CLAP_CHECKPOINT,
    main,
)
from synth_setter.pipeline.schemas.add_embeddings_config import AddEmbeddingsConfig
from synth_setter.workspace import operator_workspace

_LANCE_URI = "r2://bucket/run/train.lance"

_M2L_TIME = 3


def _fake_m2l(audio: np.ndarray) -> np.ndarray:
    """Tile the per-channel mean into a constant-shape ``(B, C*4, 3)`` latent.

    :param audio: ``(B, C, T)`` audio batch.
    :returns: ``(B, C*4, 3)`` stand-in latent batch.
    """
    per_channel = np.repeat(audio.mean(axis=2), 4, axis=1)
    return np.repeat(per_channel[:, :, None], _M2L_TIME, axis=2)


def _fake_clap(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Broadcast each row's grand mean into a ``(B, CLAP_EMBEDDING_DIM)`` embedding.

    :param mono: ``(B, T)`` mono batch.
    :param sample_rate: Ignored.
    :returns: ``(B, CLAP_EMBEDDING_DIM)`` stand-in embedding batch.
    """
    del sample_rate
    return np.repeat(mono.mean(axis=1, keepdims=True), CLAP_EMBEDDING_DIM, axis=1)


def _compose_add_embeddings():
    """Compose ``add_embeddings.yaml`` with the required ``lance_uri``.

    :returns: The composed cfg; the caller must clear ``GlobalHydra``.
    """
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        return compose(
            config_name="add_embeddings",
            return_hydra_config=True,
            overrides=[f"lance_uri={_LANCE_URI}"],
        )


def test_add_embeddings_config_surfaces_uri_and_knob_defaults() -> None:
    """The composed cfg carries the ``lance_uri`` override and the shipped knob defaults."""
    cfg = _compose_add_embeddings()
    try:
        assert cfg.lance_uri == _LANCE_URI
        assert cfg.clap_checkpoint == DEFAULT_CLAP_CHECKPOINT
        assert cfg.device is None
        assert cfg.batch_size == 128
        assert cfg.build_index is True
        assert cfg.num_partitions is None
        assert cfg.num_sub_vectors == 16
        assert cfg.metric == "cosine"
        assert cfg.resume_cache is None
        assert cfg.debug is False
    finally:
        GlobalHydra.instance().clear()


def test_add_embeddings_config_from_hydra_cfg_validates_composed_cfg() -> None:
    """``AddEmbeddingsConfig.from_hydra_cfg`` masks non-spec groups and validates the knobs."""
    cfg = _compose_add_embeddings()
    try:
        config = AddEmbeddingsConfig.from_hydra_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()
    assert config == AddEmbeddingsConfig(lance_uri=_LANCE_URI)


def _write_audio_shard(uri: Path) -> None:
    """Write a minimal Lance audio shard for endpoint runs.

    :param uri: Output ``.lance`` directory.
    """
    from tests.helpers.finalize_shards import build_lance_smoke_spec, write_minimal_lance_shard

    write_minimal_lance_shard(uri, build_lance_smoke_spec())


@pytest.mark.slow
def test_add_embeddings_main_writes_columns_through_hydra_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live ``main()`` run composes the shell, validates the cfg, and writes both columns.

    :param tmp_path: Scratch dir for the shard and the Hydra run dir.
    :param monkeypatch: Swaps the real encoders for fakes and pins argv.
    """
    uri = tmp_path / "shell.lance"
    _write_audio_shard(uri)
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_m2l_audio_encoder",
        lambda device=None: _fake_m2l,
    )
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_clap_audio_encoder",
        lambda checkpoint, device=None: _fake_clap,
    )
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            f"lance_uri={uri}",
            "build_index=false",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
    )

    main()

    assert {M2L_FIELD, CLAP_FIELD} <= set(lance.dataset(str(uri)).schema.names)
    assert AUDIO_FIELD in lance.dataset(str(uri)).schema.names


def test_add_embeddings_main_exits_1_when_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Hydra shell maps a dataset-open failure to a clean ``sys.exit(1)``.

    :param tmp_path: Scratch dir for the Hydra run dir.
    :param monkeypatch: Breaks the open path and pins argv.
    """

    def boom(uri: str) -> object:
        raise RuntimeError("missing R2 credentials")

    monkeypatch.setattr("synth_setter.pipeline.data.add_embeddings._open_lance_dataset", boom)
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            "lance_uri=s3://bucket/missing.lance",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
