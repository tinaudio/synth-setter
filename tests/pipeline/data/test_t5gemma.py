"""Checkpoint resolution, config parsing, and real-weight behavior of the SA3 T5Gemma encoder."""

import json
import sys
import types
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from synth_setter.model_cache import embedding_model_dir
from synth_setter.pipeline.data.t5gemma import (
    _PADDING_EMBEDDING_KEY,
    DEFAULT_T5GEMMA_CHECKPOINT,
    T5GEMMA_EMBEDDING_DIM,
    T5GEMMA_MAX_LENGTH,
    TextEncodeFn,
    _load_padding_embedding,
    _read_conditioner_config,
    _resolve_t5gemma_checkpoint_dir,
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
    directory: Path,
    *,
    conditioner_type: str = "t5gemma",
    cond_dim: int = T5GEMMA_EMBEDDING_DIM,
    padding_mode: str = "learned",
    max_length: int = T5GEMMA_MAX_LENGTH,
) -> None:
    """Write a minimal SA3 ``model_config.json`` carrying a prompt conditioner.

    :param directory: Directory receiving the config.
    :param conditioner_type: Conditioner implementation recorded in the config.
    :param cond_dim: Conditioner output width recorded in the config.
    :param padding_mode: Padding mode recorded on the prompt conditioner.
    :param max_length: Token budget recorded on the prompt conditioner.
    """
    config = {
        "model": {
            "conditioning": {
                "configs": [
                    {
                        "id": "prompt",
                        "type": conditioner_type,
                        "config": {
                            "max_length": max_length,
                            "padding_mode": padding_mode,
                            "repo_id": "stabilityai/stable-audio-3-small-music",
                            "subfolder": "t5gemma-b-b-ul2",
                        },
                    },
                    {"id": "seconds_total", "type": "number", "config": {"min_val": 0}},
                ],
                "cond_dim": cond_dim,
            }
        }
    }
    (directory / "model_config.json").write_text(json.dumps(config))


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


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"conditioner_type": "t5"}, "type"),
        ({"cond_dim": T5GEMMA_EMBEDDING_DIM // 2}, "cond_dim"),
        ({"cond_dim": float(T5GEMMA_EMBEDDING_DIM)}, "cond_dim"),
        ({"max_length": T5GEMMA_MAX_LENGTH // 2}, "max_length"),
        ({"max_length": float(T5GEMMA_MAX_LENGTH)}, "max_length"),
    ],
)
def test_read_conditioner_config_with_noncanonical_shape_contract_raises(
    tmp_path: Path, override: dict[str, object], expected_error: str
) -> None:
    """Reject checkpoints that change the persisted T5Gemma column contract.

    :param tmp_path: Directory holding the written config.
    :param override: Noncanonical conditioner setting.
    :param expected_error: Setting named by the validation error.
    """
    _write_model_config(tmp_path, **override)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=expected_error):
        _read_conditioner_config(tmp_path)


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
def test_encode_any_prompt_batch_returns_fixed_dim_by_max_length(
    t5gemma_encoder: TextEncodeFn,
) -> None:
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
def test_encode_overlong_prompt_truncates_to_the_same_first_tokens(
    t5gemma_encoder: TextEncodeFn,
) -> None:
    """Text past the token budget is dropped, so a longer suffix changes nothing.

    :param t5gemma_encoder: Loaded prompt conditioner.
    """
    overlong = "cutoff, resonance, " * 200

    # Encoded separately: batch position perturbs SDPA reductions by ~2e-4, which
    # would mask the exact equality truncation is supposed to produce.
    truncated = t5gemma_encoder([overlong])
    with_suffix = t5gemma_encoder([overlong + " ignored trailing text"])

    np.testing.assert_array_equal(truncated, with_suffix)


@pytest.mark.slow
@_NEEDS_WEIGHTS
def test_encode_matches_upstream_conditioner_in_float32(t5gemma_encoder: TextEncodeFn) -> None:
    """Embeddings reproduce SA3's own conditioner run at float32, transposed.

    Pinned at float32 because the checkpoint's declared bfloat16 is not stable across torch
    releases, which a persisted column cannot tolerate.

    :param t5gemma_encoder: Loaded prompt conditioner.
    """
    from stable_audio_3.models.conditioners import T5GemmaConditioner

    from synth_setter.pipeline.data.t5gemma import (
        _T5GEMMA_MODEL_NAME,
        _load_padding_embedding,
        _read_conditioner_config,
    )

    prompts = ["warm analog pad", "", "cutoff, resonance, " * 200]
    config = _read_conditioner_config(_CHECKPOINT_DIR)
    reference = T5GemmaConditioner(
        output_dim=config.cond_dim,
        model_name=_T5GEMMA_MODEL_NAME,
        max_length=config.max_length,  # pyright: ignore[reportArgumentType]
        padding_mode=config.padding_mode,
        model_path=str(_CHECKPOINT_DIR),
        subfolder=config.subfolder,
    )
    reference.padding_embedding.data.copy_(_load_padding_embedding(_CHECKPOINT_DIR, "cpu"))
    cast("torch.nn.Module", reference.model).to(torch.float32)
    reference = reference.eval().requires_grad_(False)
    with torch.no_grad():
        expected = torch.cat([reference([prompt], "cpu")[0] for prompt in prompts])

    embeddings = t5gemma_encoder(prompts)

    np.testing.assert_array_equal(embeddings, expected.float().numpy().transpose(0, 2, 1))


@pytest.mark.slow
@_NEEDS_WEIGHTS
@pytest.mark.parametrize(
    ("param_spec_name", "total_names", "retained_names"),
    [("surge_simple", 91, 31), ("surge_xt", 164, 31)],
)
def test_param_names_caption_retains_only_its_leading_names_after_truncation(
    param_spec_name: str, total_names: int, retained_names: int
) -> None:
    """Pin how much of each wide spec's caption survives SA3's 256-token budget.

    Documented in ``docs/design/data-pipeline.md``; the counts move with the
    tokenizer, the checkpoint's max_length, or a spec's parameter names.

    :param param_spec_name: Registered param spec under test.
    :param total_names: Parameter count the spec encodes.
    :param retained_names: Leading names that fit the token budget.
    """
    from transformers import AutoTokenizer

    from synth_setter.data.vst.param_spec_registry import resolve_param_spec
    from synth_setter.param_spec_name import ParamSpecName

    names = resolve_param_spec(ParamSpecName(param_spec_name)).names
    tokenizer = AutoTokenizer.from_pretrained(
        str(_CHECKPOINT_DIR), subfolder=_read_conditioner_config(_CHECKPOINT_DIR).subfolder
    )
    fits = [
        count
        for count in range(1, len(names) + 1)
        if len(tokenizer(", ".join(names[:count]))["input_ids"]) <= T5GEMMA_MAX_LENGTH
    ]

    assert (len(names), max(fits)) == (total_names, retained_names)


class _FakeConditioner(torch.nn.Module):
    """Stand-in for SA3's conditioner that records the prompt batches it receives."""

    def __init__(
        self,
        *,
        output_dim: int,
        max_length: int,
        model_name: str,
        padding_mode: str,
        model_path: str,
        subfolder: str,
    ) -> None:
        """Accept the real conditioner's keyword contract.

        :param output_dim: Embedding width.
        :param max_length: Token budget.
        :param model_name: Encoder identifier the real class asserts on.
        :param padding_mode: Padding strategy.
        :param model_path: Local checkpoint directory.
        :param subfolder: Checkpoint-relative encoder directory.
        """
        super().__init__()
        self.padding_embedding = torch.nn.Parameter(torch.zeros(output_dim))
        self.model = torch.nn.Linear(1, 1)
        self.output_dim = output_dim
        self.max_length = max_length
        del model_name, padding_mode, model_path, subfolder
        self.batches: list[list[str]] = []

    def forward(self, prompts: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic sequence embeddings for one prompt batch.

        :param prompts: Prompt batch.
        :param device: Ignored torch device.
        :returns: ``(embeddings, attention_mask)`` as the real conditioner does.
        """
        del device
        self.batches.append(list(prompts))
        rows = len(prompts)
        size = rows * self.max_length * self.output_dim
        embeddings = torch.arange(size, dtype=torch.float32).reshape(
            rows, self.max_length, self.output_dim
        )
        return embeddings, torch.ones(rows, self.max_length, dtype=torch.bool)


def _fake_checkpoint(directory: Path) -> torch.Tensor:
    """Write a config and safetensors carrying a learned padding embedding.

    :param directory: Checkpoint directory to populate.
    :returns: The padding embedding written to the checkpoint.
    """
    from safetensors.torch import save_file

    config = {
        "model": {
            "conditioning": {
                "configs": [
                    {
                        "id": "prompt",
                        "type": "t5gemma",
                        "config": {
                            "max_length": T5GEMMA_MAX_LENGTH,
                            "padding_mode": "learned",
                            "subfolder": "t5gemma-b-b-ul2",
                        },
                    }
                ],
                "cond_dim": T5GEMMA_EMBEDDING_DIM,
            }
        }
    }
    (directory / "model_config.json").write_text(json.dumps(config))
    padding = torch.arange(T5GEMMA_EMBEDDING_DIM, dtype=torch.float32) + 0.5
    save_file({_PADDING_EMBEDDING_KEY: padding}, str(directory / "model.safetensors"))
    return padding


def _install_fake_conditioner_module(
    monkeypatch: pytest.MonkeyPatch, built: list["_FakeConditioner"]
) -> None:
    """Stand in for ``stable_audio_3`` so the unit test avoids the real model.

    The injected module chain keeps external model weights out of the unit test.

    :param monkeypatch: Fixture restoring ``sys.modules`` after the test.
    :param built: List receiving each conditioner the loader constructs.
    """

    def factory(**kwargs: str | int) -> _FakeConditioner:
        conditioner = _FakeConditioner(**kwargs)  # type: ignore[arg-type]
        built.append(conditioner)
        return conditioner

    conditioners = types.ModuleType("stable_audio_3.models.conditioners")
    conditioners.T5GemmaConditioner = factory  # type: ignore[attr-defined]
    models = types.ModuleType("stable_audio_3.models")
    models.conditioners = conditioners  # type: ignore[attr-defined]
    package = types.ModuleType("stable_audio_3")
    package.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_audio_3", package)
    monkeypatch.setitem(sys.modules, "stable_audio_3.models", models)
    monkeypatch.setitem(sys.modules, "stable_audio_3.models.conditioners", conditioners)


def test_load_padding_embedding_returns_the_checkpoint_tensor(tmp_path: Path) -> None:
    """The learned padding embedding is read straight from the checkpoint.

    :param tmp_path: Checkpoint directory.
    """
    expected = _fake_checkpoint(tmp_path)

    torch.testing.assert_close(_load_padding_embedding(tmp_path, "cpu"), expected)


def test_load_padding_embedding_without_the_key_raises(tmp_path: Path) -> None:
    """A checkpoint lacking the learned padding embedding cannot be substituted.

    :param tmp_path: Checkpoint directory.
    """
    from safetensors.torch import save_file

    _fake_checkpoint(tmp_path)
    save_file({"unrelated": torch.zeros(2)}, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match="padding_embedding"):
        _load_padding_embedding(tmp_path, "cpu")


def test_load_t5gemma_text_encoder_returns_canonical_dim_major_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return canonical dim-major Lance embeddings from singleton encodes.

    :param tmp_path: Checkpoint directory.
    :param monkeypatch: Fixture installing a stand-in conditioner class.
    """
    _fake_checkpoint(tmp_path)
    built: list[_FakeConditioner] = []
    _install_fake_conditioner_module(monkeypatch, built)
    prompts = [f"prompt {index}" for index in range(20)]

    embeddings = load_t5gemma_text_encoder(str(tmp_path), "cpu")(prompts)

    assert embeddings.shape == (
        len(prompts),
        T5GEMMA_EMBEDDING_DIM,
        T5GEMMA_MAX_LENGTH,
    )


def test_load_t5gemma_text_encoder_with_no_prompts_returns_empty_canonical_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve the canonical shape for an empty prompt batch.

    :param tmp_path: Checkpoint directory.
    :param monkeypatch: Fixture installing a stand-in conditioner class.
    """
    _fake_checkpoint(tmp_path)
    built: list[_FakeConditioner] = []
    _install_fake_conditioner_module(monkeypatch, built)

    embeddings = load_t5gemma_text_encoder(str(tmp_path), "cpu")([])

    assert embeddings.shape == (0, T5GEMMA_EMBEDDING_DIM, T5GEMMA_MAX_LENGTH)


def test_load_t5gemma_text_encoder_with_duplicate_prompts_returns_identical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse one singleton encoding for duplicate prompts.

    :param tmp_path: Checkpoint directory.
    :param monkeypatch: Fixture installing a stand-in conditioner class.
    """
    _fake_checkpoint(tmp_path)
    built: list[_FakeConditioner] = []
    _install_fake_conditioner_module(monkeypatch, built)

    embeddings = load_t5gemma_text_encoder(str(tmp_path), "cpu")(["same", "other", "same"])

    np.testing.assert_array_equal(embeddings[0], embeddings[2])
    assert built[0].batches == [["same"], ["other"]]


def test_load_t5gemma_text_encoder_substitutes_the_checkpoint_padding_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conditioner runs with the checkpoint's learned padding, not its random init.

    :param tmp_path: Checkpoint directory.
    :param monkeypatch: Fixture installing a stand-in conditioner class.
    """
    expected = _fake_checkpoint(tmp_path)
    built: list[_FakeConditioner] = []
    _install_fake_conditioner_module(monkeypatch, built)

    load_t5gemma_text_encoder(str(tmp_path), "cpu")(["a"])

    torch.testing.assert_close(built[0].padding_embedding.detach(), expected)


def test_resolve_t5gemma_checkpoint_dir_with_non_default_r2_prefix_uses_a_distinct_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2 mirrors outside the default keep their own cache directories.

    :param monkeypatch: Fixture replacing credential loading and download.
    """
    downloads: list[tuple[str, Path]] = []
    monkeypatch.setattr("synth_setter.pipeline.r2_io.ensure_r2_env_loaded", lambda: None)
    monkeypatch.setattr(
        "synth_setter.pipeline.r2_io.download_dir_no_overwrite",
        lambda uri, destination: downloads.append((uri, destination)),
    )

    resolved = _resolve_t5gemma_checkpoint_dir("r2://bucket/team-a/sa3-small-music")

    assert resolved != embedding_model_dir("sa3-small-music")
    assert downloads == [("r2://bucket/team-a/sa3-small-music", resolved)]


def test_resolve_t5gemma_checkpoint_dir_with_repo_id_falls_back_to_hub_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-path, non-R2 source resolves through the HuggingFace hub.

    :param monkeypatch: Fixture replacing the hub download.
    :param tmp_path: Directory the fake download returns.
    """
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda repo_id: str(tmp_path / repo_id.split("/")[-1])
    )

    assert _resolve_t5gemma_checkpoint_dir("stabilityai/stable-audio-3-small-music") == (
        tmp_path / "stable-audio-3-small-music"
    )
