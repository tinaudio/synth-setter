"""Combined sketches filter pitch distributions before projecting signed TIV."""

from typing import cast
from unittest import mock

import pytest
import torch

from synth_setter.conditioning import NUM_SKETCH_CONTROLS, TIV_SKETCH_CONTROLS, SketchControlSpec
from synth_setter.features.tiv import chroma_to_tiv
from synth_setter.models.components.sketch_tokens import SketchControlTokens
from synth_setter.sketch import median_filter_sketch_controls, prepare_music_tiv_tokens


@pytest.fixture
def chroma_controls() -> torch.Tensor:
    """Return changing pitch mixtures whose coordinate medians do not commute.

    :returns: Music channels followed by normalized chroma over three frames.
    """
    controls = torch.zeros(1, NUM_SKETCH_CONTROLS + TIV_SKETCH_CONTROLS, 3)
    controls[:, NUM_SKETCH_CONTROLS : NUM_SKETCH_CONTROLS + 3] = torch.tensor(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    )
    return controls


@pytest.fixture
def tiv_tokens() -> SketchControlTokens:
    """Expose the TIV projection input through an identity projection.

    :returns: Tokenizer with only its TIV projection active.
    """
    module = SketchControlTokens(
        d_model=TIV_SKETCH_CONTROLS,
        num_control_tokens=3,
        profile="music_tiv",
        median_filter_max_kernel_size=3,
    )
    with torch.no_grad():
        cast(torch.nn.Linear, module.projections["tiv"]).weight.copy_(
            torch.eye(TIV_SKETCH_CONTROLS)
        )
    return module


def test_training_chroma_median_precedes_tiv_projection(
    chroma_controls: torch.Tensor, tiv_tokens: SketchControlTokens
) -> None:
    """A changing mixture produces TIV of the median distribution.

    :param chroma_controls: Three distinct pitch mixtures.
    :param tiv_tokens: Tokenizer exposing signed TIV.
    """
    expected_chroma = torch.zeros(1, TIV_SKETCH_CONTROLS, 1)
    expected_chroma[:, :3] = 1 / 3
    expected = chroma_to_tiv(expected_chroma)[0, :, 0]
    wrong = median_filter_sketch_controls(
        chroma_to_tiv(chroma_controls[:, NUM_SKETCH_CONTROLS:]), 3
    )[0, :, 1]
    assert not torch.allclose(expected, wrong)
    with mock.patch.object(torch, "randint", return_value=torch.tensor([3])):
        result = tiv_tokens(chroma_controls, torch.ones(1, 4, dtype=torch.bool))
    actual = (result - tiv_tokens.unconditional(1))[0, 1]
    torch.testing.assert_close(actual, expected)
    assert (actual < 0).any()


def test_evaluation_converts_chroma_without_random_filtering(
    chroma_controls: torch.Tensor, tiv_tokens: SketchControlTokens
) -> None:
    """Evaluation still converts chroma while leaving its temporal detail intact.

    :param chroma_controls: Distinct pitch mixtures.
    :param tiv_tokens: Tokenizer exposing signed TIV.
    """
    tiv_tokens.eval()
    with mock.patch.object(torch, "randint", side_effect=AssertionError("random eval filter")):
        result = tiv_tokens(chroma_controls, torch.ones(1, 4, dtype=torch.bool))
    expected = chroma_to_tiv(chroma_controls[:, NUM_SKETCH_CONTROLS:]).transpose(1, 2)
    torch.testing.assert_close(result - tiv_tokens.unconditional(1), expected)


def test_training_erased_chroma_retains_input_pitch(
    chroma_controls: torch.Tensor, tiv_tokens: SketchControlTokens
) -> None:
    """Changing one-hot notes cannot become artificial silence under a median.

    :param chroma_controls: Input control storage.
    :param tiv_tokens: Tokenizer exposing signed TIV.
    """
    chroma_controls[:, NUM_SKETCH_CONTROLS:] = 0
    chroma_controls[:, NUM_SKETCH_CONTROLS : NUM_SKETCH_CONTROLS + 3] = torch.eye(3)
    with mock.patch.object(torch, "randint", return_value=torch.tensor([3])):
        result = tiv_tokens(chroma_controls, torch.ones(1, 4, dtype=torch.bool))
    expected = chroma_to_tiv(chroma_controls[:, NUM_SKETCH_CONTROLS:]).transpose(1, 2)
    torch.testing.assert_close(result - tiv_tokens.unconditional(1), expected)


def test_training_silent_frame_stays_silent(
    chroma_controls: torch.Tensor, tiv_tokens: SketchControlTokens
) -> None:
    """Neighboring pitched frames must not fill a truly silent input frame.

    :param chroma_controls: Input control storage.
    :param tiv_tokens: Tokenizer exposing signed TIV.
    """
    chroma_controls[:, NUM_SKETCH_CONTROLS:] = 0
    chroma_controls[:, NUM_SKETCH_CONTROLS, (0, 2)] = 1
    with mock.patch.object(torch, "randint", return_value=torch.tensor([3])):
        result = tiv_tokens(chroma_controls, torch.ones(1, 4, dtype=torch.bool))
    torch.testing.assert_close(
        (result - tiv_tokens.unconditional(1))[0, 1], torch.zeros(TIV_SKETCH_CONTROLS)
    )


def test_combined_spec_online_profile_roundtrips() -> None:
    """The combined online profile needs no additional input-format selector."""
    spec = SketchControlSpec(
        profile="music_tiv", source="online", sample_rate=44100, num_frames=96
    )
    assert SketchControlSpec.model_validate_json(spec.model_dump_json()) == spec


def test_shared_row_kernel_filters_music_and_chroma_together(
    chroma_controls: torch.Tensor,
) -> None:
    """One row keeps detail while another applies the same median to all four groups.

    :param chroma_controls: Distinct pitch mixtures for the chroma group.
    """
    controls = chroma_controls.expand(2, -1, -1).clone()
    controls[:, :NUM_SKETCH_CONTROLS] = torch.tensor([0.8, 0.1, 0.1])
    expected_chroma = torch.zeros(1, TIV_SKETCH_CONTROLS, 1)
    expected_chroma[:, :3] = 1 / 3
    with mock.patch.object(torch, "randint", side_effect=[torch.tensor([1, 3])]):
        result = prepare_music_tiv_tokens(controls, 3)
    torch.testing.assert_close(result[0, :NUM_SKETCH_CONTROLS], controls[0, :NUM_SKETCH_CONTROLS])
    torch.testing.assert_close(
        result[1, :NUM_SKETCH_CONTROLS, 1], torch.full((NUM_SKETCH_CONTROLS,), 0.1)
    )
    torch.testing.assert_close(
        result[1, NUM_SKETCH_CONTROLS:, 1], chroma_to_tiv(expected_chroma)[0, :, 0]
    )
    torch.testing.assert_close(
        result[:1, NUM_SKETCH_CONTROLS:], chroma_to_tiv(controls[:1, NUM_SKETCH_CONTROLS:])
    )


@pytest.mark.gpu
def test_preprocessing_cuda_matches_cpu(chroma_controls: torch.Tensor) -> None:
    """Filtering and Fourier conversion agree across CPU and CUDA.

    :param chroma_controls: Distinct pitch mixtures.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    with mock.patch.object(torch, "randint", return_value=torch.tensor([3])):
        cpu = prepare_music_tiv_tokens(chroma_controls, 3)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # Match the production trainer; CUDA median indices lack deterministic support.
        torch.use_deterministic_algorithms(False)
        with mock.patch.object(torch, "randint", return_value=torch.tensor([3], device="cuda")):
            cuda = prepare_music_tiv_tokens(chroma_controls.cuda(), 3)
        torch.testing.assert_close(cuda.cpu(), cpu, atol=2e-5, rtol=2e-5)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_inference_mode_training_module_converts_without_filtering(
    chroma_controls: torch.Tensor, tiv_tokens: SketchControlTokens
) -> None:
    """Inference sampling skips augmentation even if train mode remains enabled.

    :param chroma_controls: Distinct pitch mixtures.
    :param tiv_tokens: Tokenizer exposing signed TIV.
    """
    with (
        torch.inference_mode(),
        mock.patch.object(torch, "randint", side_effect=AssertionError("random inference filter")),
    ):
        result = tiv_tokens(chroma_controls, torch.ones(1, 4, dtype=torch.bool))
    expected = chroma_to_tiv(chroma_controls[:, NUM_SKETCH_CONTROLS:]).transpose(1, 2)
    torch.testing.assert_close(result - tiv_tokens.unconditional(1), expected)
