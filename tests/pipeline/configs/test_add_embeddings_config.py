"""Config-layer tests that compose the real ``add_embeddings.yaml`` Hydra config.

The endpoint (``add_embeddings.main``) composes its shipped YAML and validates
it into :class:`AddEmbeddingsConfig`; a break in the ``defaults`` / ``paths`` /
``hydra`` composition would pass every unit test and only fail in production.
These bare-compose tests pin the knobs the endpoint reads and prove
``from_hydra_cfg`` validates the composed cfg, plus live ``main()`` runs of the
Hydra shell (m2l+clap success + failure→exit-1, and the SAME dispatch) with the
real encoders swapped for fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lance
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from synth_setter.data.vst.shapes import (
    AUDIO_FIELD,
    CLAP_FIELD,
    M2L_FIELD,
    SAME_L_FIELD,
    SAME_S_FIELD,
)
from synth_setter.pipeline.data.add_embeddings import (
    CLAP_EMBEDDING_DIM,
    DEFAULT_CLAP_CHECKPOINT,
    DEFAULT_SAME_L_CHECKPOINT,
    DEFAULT_SAME_S_CHECKPOINT,
    SAME_EMBEDDING_DIM,
    SAME_SAMPLE_RATE,
    SameEncodeFn,
    main,
    same_num_latent_frames,
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


def _compose_add_embeddings() -> DictConfig:
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


def test_add_embeddings_config_coerces_resume_cache_string_to_path() -> None:
    """A Hydra string ``resume_cache`` override is coerced to ``Path`` under strict."""
    config = AddEmbeddingsConfig(lance_uri=_LANCE_URI, resume_cache="cache/embed.cache")  # type: ignore[arg-type]
    assert config.resume_cache == Path("cache/embed.cache")


@pytest.mark.parametrize("bad", [0, -1, 15])
def test_add_embeddings_config_rejects_bad_num_sub_vectors(bad: int) -> None:
    """``num_sub_vectors`` must be positive and divide the clap dim.

    :param bad: A non-positive or non-dividing sub-vector count.
    """
    with pytest.raises(ValueError):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, num_sub_vectors=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_add_embeddings_config_rejects_nonpositive_num_partitions(bad: int) -> None:
    """``num_partitions`` must be positive when set.

    :param bad: A non-positive partition count.
    """
    with pytest.raises(ValueError):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, num_partitions=bad)


def test_add_embeddings_config_rejects_unknown_metric() -> None:
    """``metric`` is constrained to the metrics Lance's IVF_PQ accepts."""
    with pytest.raises(ValueError):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, metric="banana")


def test_add_embeddings_config_surfaces_same_field_defaults() -> None:
    """The composed cfg ships an empty ``same_variants`` and the R2-mirror checkpoints."""
    cfg = _compose_add_embeddings()
    try:
        assert list(cfg.same_variants) == []
        assert cfg.same_s_checkpoint == DEFAULT_SAME_S_CHECKPOINT
        assert cfg.same_l_checkpoint == DEFAULT_SAME_L_CHECKPOINT
    finally:
        GlobalHydra.instance().clear()


def test_add_embeddings_config_from_hydra_cfg_coerces_same_variants_to_tuple() -> None:
    """A Hydra ``same_variants=[s,l]`` override validates into an ordered ``tuple``."""
    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        cfg = compose(
            config_name="add_embeddings",
            return_hydra_config=True,
            overrides=[f"lance_uri={_LANCE_URI}", "same_variants=[s,l]"],
        )
    try:
        config = AddEmbeddingsConfig.from_hydra_cfg(cfg)
    finally:
        GlobalHydra.instance().clear()
    assert config.same_variants == ("s", "l")


@pytest.mark.parametrize("bad", [["x"], ["s", "x"]])
def test_add_embeddings_config_rejects_unknown_same_variant(bad: list[str]) -> None:
    """``same_variants`` tokens are constrained to the ``s``/``l`` SAME choices.

    :param bad: A variant list carrying an unknown token.
    """
    with pytest.raises(ValueError):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, same_variants=bad)  # type: ignore[arg-type]


def test_add_embeddings_config_rejects_duplicate_same_variant() -> None:
    """A repeated variant is rejected: each SAME column is written at most once."""
    with pytest.raises(ValueError):
        AddEmbeddingsConfig(lance_uri=_LANCE_URI, same_variants=["s", "s"])  # type: ignore[arg-type]


def _fake_same(fill: float) -> SameEncodeFn:
    """Build a SAME encoder stub emitting a constant ``(B, 256, T)`` latent.

    :param fill: Constant latent value, distinguishing variants in round-trips.
    :returns: Encoder mapping prepared ``(B, 2, T)`` audio to a constant latent.
    """

    def encode(stereo: np.ndarray) -> np.ndarray:
        frames = same_num_latent_frames(stereo.shape[2], SAME_SAMPLE_RATE)
        return np.full((stereo.shape[0], SAME_EMBEDDING_DIM, frames), fill, dtype=np.float32)

    return encode


def _same_fill_for(checkpoint: str) -> float:
    """Map a SAME checkpoint to a distinct stub fill so a mis-routed column is caught.

    :param checkpoint: The per-variant checkpoint the endpoint passed the loader.
    :returns: ``0.25`` for the SAME-S checkpoint, ``0.75`` otherwise (SAME-L).
    """
    return 0.25 if checkpoint == DEFAULT_SAME_S_CHECKPOINT else 0.75


@pytest.mark.slow
def test_add_embeddings_main_writes_same_columns_through_hydra_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live SAME-mode ``main()`` run dispatches on ``same_variants`` and writes both columns.

    :param tmp_path: Scratch dir for the shard and the Hydra run dir.
    :param monkeypatch: Swaps the real SAME encoder loader for a fake and pins argv.
    """
    uri = tmp_path / "same_shell.lance"
    _write_audio_shard(uri)
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_same_audio_encoder",
        lambda checkpoint, device=None: _fake_same(_same_fill_for(checkpoint)),
    )
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            f"lance_uri={uri}",
            "same_variants=[s,l]",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
    )

    main()

    names = set(lance.dataset(str(uri)).schema.names)
    assert {SAME_S_FIELD, SAME_L_FIELD} <= names
    # The SAME path must not also write the m2l+clap columns.
    assert not ({M2L_FIELD, CLAP_FIELD} & names)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("variant", "present", "absent", "checkpoint"),
    [
        ("s", SAME_S_FIELD, SAME_L_FIELD, DEFAULT_SAME_S_CHECKPOINT),
        ("l", SAME_L_FIELD, SAME_S_FIELD, DEFAULT_SAME_L_CHECKPOINT),
    ],
)
def test_add_embeddings_main_single_same_variant_writes_only_that_column(
    variant: str,
    present: str,
    absent: str,
    checkpoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single ``same_variants`` entry writes only its column, loaded from its checkpoint.

    Guards the per-variant dispatch: a regression that ignored ``same_variants`` and
    always wrote both SAME columns (or loaded the wrong checkpoint) would fail here.
    The stub fill is keyed on the checkpoint, so the written column also proves the
    matching checkpoint fed the matching column.

    :param variant: The single SAME variant token requested (``"s"``/``"l"``).
    :param present: The column that must land.
    :param absent: The sibling SAME column that must not land.
    :param checkpoint: The checkpoint the requested variant resolves to.
    :param tmp_path: Scratch dir for the shard and the Hydra run dir.
    :param monkeypatch: Swaps the real SAME encoder loader for a fake and pins argv.
    """
    uri = tmp_path / "same_single.lance"
    _write_audio_shard(uri)
    monkeypatch.setattr(
        "synth_setter.pipeline.data.add_embeddings.load_same_audio_encoder",
        lambda checkpoint, device=None: _fake_same(_same_fill_for(checkpoint)),
    )
    monkeypatch.setenv("PROJECT_ROOT", str(operator_workspace()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synth-setter-add-embeddings",
            f"lance_uri={uri}",
            f"same_variants=[{variant}]",
            f"paths.log_dir={tmp_path}",
            f"hydra.run.dir={tmp_path / 'run'}",
        ],
    )

    main()

    dataset = lance.dataset(str(uri))
    names = set(dataset.schema.names)
    assert present in names
    assert absent not in names
    assert not ({M2L_FIELD, CLAP_FIELD} & names)
    values = dataset.to_table(columns=[present]).combine_chunks().column(present).chunk(0)
    assert float(values.to_numpy_ndarray().flat[0]) == _same_fill_for(checkpoint)
