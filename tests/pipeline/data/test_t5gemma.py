"""Checkpoint resolution, config parsing, and real-weight behavior of the SA3 T5Gemma encoder."""

import json
from pathlib import Path

import numpy as np
import pytest

from synth_setter.model_cache import embedding_model_dir
from synth_setter.pipeline.data.t5gemma import (
    DEFAULT_T5GEMMA_CHECKPOINT,
    T5GEMMA_EMBEDDING_DIM,
    T5GEMMA_MAX_LENGTH,
    _load_padding_embedding,
    _read_conditioner_config,
    _resolve_t5gemma_checkpoint_dir,
    TextEncodeFn,
    load_t5gemma_text_encoder,
)

_CHECKPOINT_DIR = embedding_model_dir("sa3-small-music")
_NEEDS_WEIGHTS = pytest.mark.skipif(
    not (_CHECKPOINT_DIR / "model_config.json").is_file(),
    reason=(
        f"SA3 checkpoint absent at {_CHECKPOINT_DIR}; hydrate it with "
        f"`rclone copy --checksum {DEFAULT_T5GEMMA_CHECKPOINT} {_CHECKPOINT_DIR}`"
    ),
)


def _write_model_config(
    directory: Path, *, padding_mode: str = "learned", max_length: int = 256
) -> Path:
    """Write a minimal SA3 ``model_config.json`` carrying a prompt conditioner.

    :param directory: Directory receiving the config.
    :param padding_mode: Padding mode recorded on the prompt conditioner.
    :param max_length: Token budget recorded on the prompt conditioner.
    :returns: Path to the written config.
    """
    config = {
        "model": {
            "conditioning": {
                "configs": [
                    {
                        "id": "prompt",
                        "type": "t5gemma",
                        "config": {
                            "max_length": max_length,
                            "padding_mode": padding_mode,
                            "repo_id": "stabilityai/stable-audio-3-small-music",
                            "subfolder": "t5gemma-b-b-ul2",
                        },
                    },
                    {"id": "seconds_total", "type": "number", "config": {"min_val": 0}},
                ],
                "cond_dim": 768,
            }
        }
    }
    path = directory / "model_config.json"
    path.write_text(json.dumps(config))
    return path


def test_resolve_t5gemma_checkpoint_dir_with_default_r2_source_uses_canonical_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default R2 source hydrates into its stable model directory.

    :param monkeypatch: Fixture isolating cache location and R2 download.
    :param tmp_path: XDG cache root for the assertion.
    """
    downloads: list[tuple[str, Path]] = []
    expected = tmp_path / "synth-setter/models/embeddings" / "sa3-small-music"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, destination: downloads.append((uri, destination)),
    )

    resolved = _resolve_t5gemma_checkpoint_dir(DEFAULT_T5GEMMA_CHECKPOINT)

    assert resolved == expected
    assert downloads == [(DEFAULT_T5GEMMA_CHECKPOINT, expected)]


def test_resolve_t5gemma_checkpoint_dir_with_existing_local_path_returns_it(
    tmp_path: Path,
) -> None:
    """An existing local checkpoint directory needs no download.

    :param tmp_path: Existing local checkpoint directory.
    """
    assert _resolve_t5gemma_checkpoint_dir(str(tmp_path)) == tmp_path


def test_read_conditioner_config_returns_prompt_conditioner_settings(tmp_path: Path) -> None:
    """Conditioner settings come from the checkpoint, never from hardcoded defaults.

    :param tmp_path: Directory holding the written config.
    """
    _write_model_config(tmp_path)

    config = _read_conditioner_config(tmp_path)

    assert (config.max_length, config.padding_mode, config.subfolder, config.cond_dim) == (
        256,
        "learned",
        "t5gemma-b-b-ul2",
        768,
    )


def test_read_conditioner_config_with_zero_padding_mode_raises(tmp_path: Path) -> None:
    """A non-learned padding mode is rejected rather than silently substituting zeros.

    :param tmp_path: Directory holding the written config.
    """
    _write_model_config(tmp_path, padding_mode="zero")

    with pytest.raises(ValueError, match="padding_mode"):
        _read_conditioner_config(tmp_path)


def test_read_conditioner_config_without_prompt_conditioner_raises(tmp_path: Path) -> None:
    """A checkpoint carrying no prompt conditioner cannot produce text embeddings.

    :param tmp_path: Directory holding the written config.
    """
    config = {"model": {"conditioning": {"configs": [{"id": "seconds_total"}], "cond_dim": 768}}}
    (tmp_path / "model_config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="prompt"):
        _read_conditioner_config(tmp_path)


@pytest.fixture(scope="module")
def t5gemma_encoder() -> TextEncodeFn:
    """Load the SA3 prompt conditioner once for the real-weight tests.

    :returns: Encoder over prompt batches.
    """
    return load_t5gemma_text_encoder(str(_CHECKPOINT_DIR), "cpu")


@pytest.fixture(scope="module")
def padding_embedding() -> np.ndarray:
    """Read the checkpoint's learned padding embedding.

    :returns: The learned padding embedding as a numpy vector.
    """
    return _load_padding_embedding(_CHECKPOINT_DIR, "cpu").float().numpy()


@pytest.mark.slow
@_NEEDS_WEIGHTS
def test_encode_any_prompt_batch_returns_fixed_dim_by_max_length(t5gemma_encoder: TextEncodeFn) -> None:
    """Every prompt yields the same fixed-shape embedding regardless of length.

    :param t5gemma_encoder: Loaded prompt conditioner.
    """
    embeddings = t5gemma_encoder(["warm analog pad", "", "a " * 400])

    assert embeddings.shape == (3, T5GEMMA_EMBEDDING_DIM, T5GEMMA_MAX_LENGTH)


@pytest.mark.slow
@_NEEDS_WEIGHTS
def test_encode_empty_prompt_returns_only_learned_padding(
    t5gemma_encoder: TextEncodeFn, padding_embedding: np.ndarray
) -> None:
    """The unconditional prompt is every position replaced by the learned padding.

    :param t5gemma_encoder: Loaded prompt conditioner.
    :param padding_embedding: Checkpoint's learned padding embedding.
    """
    embeddings = t5gemma_encoder([""])

    expected = np.broadcast_to(
        padding_embedding[:, None], (T5GEMMA_EMBEDDING_DIM, T5GEMMA_MAX_LENGTH)
    )
    np.testing.assert_array_equal(embeddings[0], expected)


@pytest.mark.slow
@_NEEDS_WEIGHTS
def test_encode_short_prompt_pads_some_positions_and_keeps_others(
    t5gemma_encoder: TextEncodeFn, padding_embedding: np.ndarray
) -> None:
    """Real tokens survive while unused positions become the learned padding.

    :param t5gemma_encoder: Loaded prompt conditioner.
    :param padding_embedding: Checkpoint's learned padding embedding.
    """
    embeddings = t5gemma_encoder(["warm analog pad"])[0]

    is_padding = np.all(embeddings == padding_embedding[:, None], axis=0)
    assert is_padding.any(), "a short prompt must leave padded positions"
    assert not is_padding.all(), "a non-empty prompt must retain real token positions"


@pytest.mark.slow
@_NEEDS_WEIGHTS
def test_encode_overlong_prompt_truncates_to_the_same_first_tokens(t5gemma_encoder: TextEncodeFn) -> None:
    """Text past the token budget is dropped, so a longer suffix changes nothing.

    :param t5gemma_encoder: Loaded prompt conditioner.
    """
    overlong = "cutoff, resonance, " * 200

    embeddings = t5gemma_encoder([overlong, overlong + " ignored trailing text"])

    np.testing.assert_array_equal(embeddings[0], embeddings[1])
