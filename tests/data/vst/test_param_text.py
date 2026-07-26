"""Behavior of the param-spec text normalizers feeding T5Gemma conditioning."""

import numpy as np
import pytest

from synth_setter.data.vst.param_spec import ContinuousParameter, DiscreteLiteralParameter, ParamSpec
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.data.vst.param_text import (
    DEFAULT_PARAM_TEXT_NORMALIZER,
    PARAM_TEXT_NORMALIZERS,
    param_names_normalizer,
    resolve_param_text_normalizer,
)


def _two_by_one_spec() -> ParamSpec:
    """Build a spec with two synth params and one note param.

    :returns: Spec whose encoded rows are three columns wide.
    """
    return ParamSpec(
        synth_params=[ContinuousParameter(name="cutoff"), ContinuousParameter(name="resonance")],
        note_params=[DiscreteLiteralParameter(name="pitch", min=21, max=108)],
    )


def test_param_names_normalizer_with_three_rows_returns_one_caption_per_row() -> None:
    """Every encoded row gets its own caption."""
    rows = np.zeros((3, 3), dtype=np.float32)

    captions = param_names_normalizer(_two_by_one_spec(), rows)

    assert len(captions) == 3


def test_param_names_normalizer_joins_synth_then_note_names_with_commas() -> None:
    """The caption enumerates the encoded parameter space in encode order."""
    rows = np.zeros((1, 3), dtype=np.float32)

    captions = param_names_normalizer(_two_by_one_spec(), rows)

    assert captions[0] == "cutoff, resonance, pitch"


@pytest.mark.parametrize(
    ("parameter_names", "expected_caption"),
    [
        pytest.param(
            [
                "oscillator_one_wavetable_position",
                "oscillator_one_frequency_modulation_depth",
                "oscillator_two_unison_voice_detune_amount",
                "filter_one_frequency_modulation_envelope_amount",
                "filter_two_key_tracking_modulation_amount",
                "amplifier_envelope_release_time_in_seconds",
            ],
            "oscillator_one_wavetable_position, "
            "oscillator_one_frequency_modulation_depth, "
            "oscillator_two_unison_voice_detune_amount, "
            "filter_one_frequency_modulation_envelope_amount, "
            "filter_two_key_tracking_modulation_amount, "
            "amplifier_envelope_release_time_in_seconds",
            id="long-synth-parameter-names",
        ),
        pytest.param(
            [
                "macro_control_brightness_and_harmonic_complexity",
                "macro_control_movement_and_modulation_intensity",
                "effect_chain_pre_delay_time_and_tempo_synchronization",
                "effect_chain_reverb_room_size_and_diffusion_amount",
                "effect_chain_stereo_width_and_mid_side_balance",
                "note_expression_polyphonic_aftertouch_pressure_amount",
            ],
            "macro_control_brightness_and_harmonic_complexity, "
            "macro_control_movement_and_modulation_intensity, "
            "effect_chain_pre_delay_time_and_tempo_synchronization, "
            "effect_chain_reverb_room_size_and_diffusion_amount, "
            "effect_chain_stereo_width_and_mid_side_balance, "
            "note_expression_polyphonic_aftertouch_pressure_amount",
            id="long-macro-effect-and-note-names",
        ),
    ],
)
def test_param_names_normalizer_with_long_names_returns_exact_caption(
    parameter_names: list[str], expected_caption: str
) -> None:
    """Long captions preserve every name, delimiter, and input position.

    :param parameter_names: Ordered names used to build a parameter spec.
    :param expected_caption: Complete caption expected from the normalizer.
    """
    spec = ParamSpec(
        synth_params=[ContinuousParameter(name=name) for name in parameter_names[:-1]],
        note_params=[ContinuousParameter(name=parameter_names[-1])],
    )
    rows = np.zeros((2, spec.encoded_width), dtype=np.float32)

    captions = param_names_normalizer(spec, rows)

    assert captions == [expected_caption, expected_caption]


def test_param_names_normalizer_with_differing_values_returns_identical_captions() -> None:
    """This normalizer describes the parameter space, so values do not change it."""
    rows = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]], dtype=np.float32)

    captions = param_names_normalizer(_two_by_one_spec(), rows)

    assert captions[0] == captions[1]


def test_param_names_normalizer_with_no_rows_returns_empty_list() -> None:
    """An empty batch produces no captions rather than one stray caption."""
    rows = np.zeros((0, 3), dtype=np.float32)

    captions = param_names_normalizer(_two_by_one_spec(), rows)

    assert captions == []


def test_param_names_normalizer_with_registered_spec_covers_every_parameter() -> None:
    """A real spec's caption names each of its encoded parameters."""
    spec = resolve_param_spec(ParamSpecName("surge_4"))
    rows = np.zeros((1, spec.encoded_width), dtype=np.float32)

    caption = param_names_normalizer(spec, rows)[0]

    assert caption.split(", ") == spec.names


def test_resolve_param_text_normalizer_with_default_name_returns_names_normalizer() -> None:
    """The default normalizer name resolves to the param-names strategy."""
    assert resolve_param_text_normalizer(DEFAULT_PARAM_TEXT_NORMALIZER) is param_names_normalizer


def test_resolve_param_text_normalizer_with_unknown_name_raises() -> None:
    """An unregistered strategy fails loudly instead of silently defaulting."""
    with pytest.raises(KeyError, match="nonexistent"):
        resolve_param_text_normalizer("nonexistent")


def test_param_text_normalizers_registry_contains_the_default_name() -> None:
    """The default name is always resolvable through the registry."""
    assert DEFAULT_PARAM_TEXT_NORMALIZER in PARAM_TEXT_NORMALIZERS
