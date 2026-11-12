"""Surge XT FX units probed by an impulse through the in-process SurgePy engine.

Introspected from SurgePy ``1.3.master.f7b97c68``. The patch topology is Audio In in global
slot G1 (injecting the impulse) feeding the effect in G2. ``fx_g2_fx_type`` selects the
effect; ``SURGE_FX_TYPE_LAYOUTS`` routes the selected type's native slots onto the spec.

- Continuous params share ``FX_SLOT_COUNT`` generic slots: slot ``i`` means the selected
  type's ``i``-th continuous native slot, so it is positional conditional on the type, and
  slots past a type's own count are inert. Every slot spans its full native range; long
  tails (Reverb 1 at maximum decay rings past 30 s) need a render window chosen for them.
- Discrete selectors stay per type, except two shared ones whose keys are canonical
  labels, not verbatim SurgePy displays: ``fx_g2_lfo_waveform`` (Phaser displays
  ``Sawtooth`` as ``Saw``) and ``fx_g2_drive_model`` (Waveshaper orders the same eight
  shapes differently). Shared raw values are canonical positions; map a label to its
  native index with ``shared_native_index``, never by position.
- Phaser stage count, Ring Mod shape and Ring Mod unison voices are scalar native integers.
  Ring Mod shape indices 0-27 are wavetables in an order that is not perceptually monotonic
  and index 28 is Audio In, which swaps the carrier for the external input.
- Waveshaper is limited to the eight drive-model shapes it shares with Rotary and
  Distortion; its Additive, Fold, Sine-sum and Cycle families are not selectable.

Off, Audio In (the injector), Vocoder (silent without a synth carrier), Tape (differs across
fresh engines), and Airwindows and Nimbus (labels depend on a nested mode selector) are not
selectable. Other one-hot keys and labels are verbatim SurgePy displays.
"""

from dataclasses import dataclass

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    ParamSpec,
)

FX_SLOT_COUNT = 12
LFO_WAVEFORMS = ("Sine", "Triangle", "Sawtooth", "Noise", "Sample & Hold", "Square")
DRIVE_MODELS = (
    "Soft",
    "Hard",
    "Asymmetric",
    "Sine",
    "Digital",
    "OJD",
    "Full Wave",
    "Fuzz Soft Clip",
)
# Waveshaper's native indices for the drive models; every other owner of a shared
# selector uses canonical order natively.
_WAVESHAPER_DRIVE_INDEX = {
    "Soft": 1,
    "Hard": 2,
    "Asymmetric": 3,
    "Sine": 4,
    "Digital": 5,
    "OJD": 41,
    "Full Wave": 10,
    "Fuzz Soft Clip": 25,
}
_SHARED_LABELS = {"fx_g2_lfo_waveform": LFO_WAVEFORMS, "fx_g2_drive_model": DRIVE_MODELS}


@dataclass(frozen=True)
class FxSlot:
    """One native SurgePy slot of the G2 effect and the name it carries in the spec.

    .. attribute :: native_slot

       Index into the effect's native ``p`` parameter list.

    .. attribute :: name

       Native label slug for a continuous slot; spec parameter name otherwise.
    """

    native_slot: int
    name: str


@dataclass(frozen=True)
class FxTypeLayout:
    """Routing of one FX type's native slots onto the spec.

    .. attribute :: continuous

       Native float slots in order; the ``i``-th fills ``fx_g2_slot_{i+1}``.

    .. attribute :: discrete

       Native discrete slots and the spec selector each one reads.
    """

    continuous: tuple[FxSlot, ...]
    discrete: tuple[FxSlot, ...]


SURGE_FX_TYPE_LAYOUTS = {
    "Delay": FxTypeLayout(
        continuous=(
            FxSlot(0, "left"),
            FxSlot(1, "right"),
            FxSlot(2, "feedback"),
            FxSlot(3, "crossfeed"),
            FxSlot(4, "low_cut"),
            FxSlot(5, "high_cut"),
            FxSlot(6, "rate"),
            FxSlot(7, "depth"),
            FxSlot(8, "channel"),
            FxSlot(10, "mix"),
            FxSlot(11, "width"),
        ),
        discrete=(),
    ),
    "Reverb 1": FxTypeLayout(
        continuous=(
            FxSlot(0, "pre_delay"),
            FxSlot(2, "size"),
            FxSlot(3, "decay_time"),
            FxSlot(4, "hf_damping"),
            FxSlot(5, "low_cut"),
            FxSlot(6, "peak_freq"),
            FxSlot(7, "peak_gain"),
            FxSlot(8, "high_cut"),
            FxSlot(9, "mix"),
            FxSlot(10, "width"),
        ),
        discrete=(FxSlot(1, "fx_g2_reverb_1_room_shape"),),
    ),
    "Phaser": FxTypeLayout(
        continuous=(
            FxSlot(0, "center"),
            FxSlot(1, "feedback"),
            FxSlot(2, "sharpness"),
            FxSlot(3, "rate"),
            FxSlot(4, "depth"),
            FxSlot(5, "stereo"),
            FxSlot(6, "mix"),
            FxSlot(7, "width"),
            FxSlot(9, "spread"),
            FxSlot(11, "tone"),
        ),
        discrete=(
            FxSlot(8, "fx_g2_phaser_count"),
            FxSlot(10, "fx_g2_lfo_waveform"),
        ),
    ),
    "Rotary": FxTypeLayout(
        continuous=(
            FxSlot(0, "horn_rate"),
            FxSlot(1, "doppler"),
            FxSlot(2, "tremolo"),
            FxSlot(3, "rotor_rate"),
            FxSlot(4, "drive"),
            FxSlot(6, "width"),
            FxSlot(7, "mix"),
        ),
        discrete=(FxSlot(5, "fx_g2_drive_model"),),
    ),
    "Distortion": FxTypeLayout(
        continuous=(
            FxSlot(0, "gain_p1"),
            FxSlot(1, "frequency_p2"),
            FxSlot(2, "bandwidth_p3"),
            FxSlot(3, "high_cut_p4"),
            FxSlot(4, "drive"),
            FxSlot(5, "feedback"),
            FxSlot(6, "gain_p7"),
            FxSlot(7, "frequency_p8"),
            FxSlot(8, "bandwidth_p9"),
            FxSlot(9, "high_cut_p10"),
            FxSlot(10, "gain_p11"),
        ),
        discrete=(FxSlot(11, "fx_g2_drive_model"),),
    ),
    "EQ": FxTypeLayout(
        continuous=(
            FxSlot(0, "gain_1"),
            FxSlot(1, "frequency_1"),
            FxSlot(2, "bandwidth_1"),
            FxSlot(3, "gain_2"),
            FxSlot(4, "frequency_2"),
            FxSlot(5, "bandwidth_2"),
            FxSlot(6, "gain_3"),
            FxSlot(7, "frequency_3"),
            FxSlot(8, "bandwidth_3"),
            FxSlot(9, "gain"),
            FxSlot(10, "mix"),
        ),
        discrete=(),
    ),
    "Freq Shift": FxTypeLayout(
        continuous=(
            FxSlot(0, "left"),
            FxSlot(1, "right"),
            FxSlot(2, "time"),
            FxSlot(3, "feedback"),
            FxSlot(4, "mix"),
        ),
        discrete=(),
    ),
    "Conditioner": FxTypeLayout(
        continuous=(
            FxSlot(0, "bass"),
            FxSlot(1, "treble"),
            FxSlot(2, "width"),
            FxSlot(3, "balance"),
            FxSlot(4, "threshold"),
            FxSlot(5, "attack_rate"),
            FxSlot(6, "release_rate"),
            FxSlot(7, "gain"),
            FxSlot(8, "side_low_cut"),
        ),
        discrete=(),
    ),
    "Chorus": FxTypeLayout(
        continuous=(
            FxSlot(0, "time"),
            FxSlot(1, "rate"),
            FxSlot(2, "depth"),
            FxSlot(3, "feedback"),
            FxSlot(4, "low_cut"),
            FxSlot(5, "high_cut"),
            FxSlot(6, "mix"),
            FxSlot(7, "width"),
        ),
        discrete=(),
    ),
    "Reverb 2": FxTypeLayout(
        continuous=(
            FxSlot(0, "pre_delay"),
            FxSlot(1, "room_size"),
            FxSlot(2, "decay_time"),
            FxSlot(3, "diffusion"),
            FxSlot(4, "buildup"),
            FxSlot(5, "modulation"),
            FxSlot(6, "lf_damping"),
            FxSlot(7, "hf_damping"),
            FxSlot(8, "width"),
            FxSlot(9, "mix"),
        ),
        discrete=(),
    ),
    "Flanger": FxTypeLayout(
        continuous=(
            FxSlot(2, "rate"),
            FxSlot(3, "depth"),
            FxSlot(4, "count"),
            FxSlot(5, "base_pitch"),
            FxSlot(6, "spacing"),
            FxSlot(7, "feedback"),
            FxSlot(8, "hf_damping"),
            FxSlot(9, "width"),
            FxSlot(10, "mix"),
        ),
        discrete=(
            FxSlot(0, "fx_g2_flanger_mode"),
            FxSlot(1, "fx_g2_lfo_waveform"),
        ),
    ),
    "Ring Mod": FxTypeLayout(
        continuous=(
            FxSlot(1, "frequency"),
            FxSlot(2, "unison_detune"),
            FxSlot(4, "forward_bias"),
            FxSlot(5, "linear_region"),
            FxSlot(6, "low_cut"),
            FxSlot(7, "high_cut"),
            FxSlot(8, "mix"),
        ),
        discrete=(
            FxSlot(0, "fx_g2_ring_mod_shape"),
            FxSlot(3, "fx_g2_ring_mod_unison_voices"),
        ),
    ),
    "Neuron": FxTypeLayout(
        continuous=(
            FxSlot(0, "drive"),
            FxSlot(1, "squash"),
            FxSlot(2, "stab"),
            FxSlot(3, "asymmetry"),
            FxSlot(4, "bias"),
            FxSlot(5, "frequency"),
            FxSlot(6, "separation"),
            FxSlot(8, "rate"),
            FxSlot(9, "depth"),
            FxSlot(10, "width"),
            FxSlot(11, "gain"),
        ),
        discrete=(FxSlot(7, "fx_g2_lfo_waveform"),),
    ),
    "Graphic EQ": FxTypeLayout(
        continuous=(
            FxSlot(0, "30_hz"),
            FxSlot(1, "60_hz"),
            FxSlot(2, "120_hz"),
            FxSlot(3, "250_hz"),
            FxSlot(4, "500_hz"),
            FxSlot(5, "1_khz"),
            FxSlot(6, "2_khz"),
            FxSlot(7, "4_khz"),
            FxSlot(8, "8_khz"),
            FxSlot(9, "12_khz"),
            FxSlot(10, "16_khz"),
            FxSlot(11, "gain"),
        ),
        discrete=(),
    ),
    "Resonator": FxTypeLayout(
        continuous=(
            FxSlot(0, "frequency_1"),
            FxSlot(1, "resonance_1"),
            FxSlot(2, "gain_1"),
            FxSlot(3, "frequency_2"),
            FxSlot(4, "resonance_2"),
            FxSlot(5, "gain_2"),
            FxSlot(6, "frequency_3"),
            FxSlot(7, "resonance_3"),
            FxSlot(8, "gain_3"),
            FxSlot(10, "gain"),
            FxSlot(11, "mix"),
        ),
        discrete=(FxSlot(9, "fx_g2_resonator_mode"),),
    ),
    "CHOW": FxTypeLayout(
        continuous=(
            FxSlot(0, "threshold"),
            FxSlot(1, "ratio"),
            FxSlot(3, "mix"),
        ),
        discrete=(FxSlot(2, "fx_g2_chow_flip"),),
    ),
    "Exciter": FxTypeLayout(
        continuous=(
            FxSlot(0, "drive"),
            FxSlot(1, "tone"),
            FxSlot(2, "attack"),
            FxSlot(3, "release"),
            FxSlot(4, "mix"),
        ),
        discrete=(),
    ),
    "Ensemble": FxTypeLayout(
        continuous=(
            FxSlot(0, "anti_alias_filter"),
            FxSlot(1, "frequency_1"),
            FxSlot(2, "depth_1"),
            FxSlot(3, "frequency_2"),
            FxSlot(4, "depth_2"),
            FxSlot(6, "clock_rate"),
            FxSlot(7, "saturation"),
            FxSlot(8, "feedback"),
            FxSlot(9, "width"),
            FxSlot(10, "mix"),
            FxSlot(11, "reconstruction_filter"),
        ),
        discrete=(FxSlot(5, "fx_g2_ensemble_type"),),
    ),
    "Combulator": FxTypeLayout(
        continuous=(
            FxSlot(0, "extra_noise"),
            FxSlot(1, "center"),
            FxSlot(2, "offset_2"),
            FxSlot(3, "offset_3"),
            FxSlot(4, "feedback"),
            FxSlot(5, "tone"),
            FxSlot(6, "comb_1"),
            FxSlot(7, "comb_2"),
            FxSlot(8, "comb_3"),
            FxSlot(9, "pan_2"),
            FxSlot(10, "pan_3"),
            FxSlot(11, "mix"),
        ),
        discrete=(),
    ),
    "Treemonster": FxTypeLayout(
        continuous=(
            FxSlot(0, "threshold"),
            FxSlot(1, "speed"),
            FxSlot(2, "low_cut"),
            FxSlot(3, "high_cut"),
            FxSlot(4, "pitch"),
            FxSlot(5, "ring_modulation"),
            FxSlot(6, "width"),
            FxSlot(7, "mix"),
        ),
        discrete=(),
    ),
    "Waveshaper": FxTypeLayout(
        continuous=(
            FxSlot(0, "low_cut_p1"),
            FxSlot(1, "high_cut_p2"),
            FxSlot(3, "bias"),
            FxSlot(4, "drive"),
            FxSlot(5, "low_cut_p6"),
            FxSlot(6, "high_cut_p7"),
            FxSlot(7, "gain"),
            FxSlot(8, "mix"),
        ),
        discrete=(FxSlot(2, "fx_g2_drive_model"),),
    ),
    "Mid-Side Tool": FxTypeLayout(
        continuous=(
            FxSlot(1, "low_cut_p2"),
            FxSlot(2, "gain_p3"),
            FxSlot(3, "frequency_p4"),
            FxSlot(4, "high_cut_p5"),
            FxSlot(5, "low_cut_p6"),
            FxSlot(6, "gain_p7"),
            FxSlot(7, "frequency_p8"),
            FxSlot(8, "high_cut_p9"),
            FxSlot(9, "mid_gain"),
            FxSlot(10, "side_gain"),
            FxSlot(11, "balance"),
        ),
        discrete=(FxSlot(0, "fx_g2_mid_side_tool_matrix"),),
    ),
    "Spring Reverb": FxTypeLayout(
        continuous=(
            FxSlot(0, "size"),
            FxSlot(1, "decay"),
            FxSlot(2, "reflections"),
            FxSlot(3, "hf_damping"),
            FxSlot(4, "spin"),
            FxSlot(5, "chaos"),
            FxSlot(6, "knock"),
            FxSlot(7, "mix"),
        ),
        discrete=(),
    ),
    "Bonsai": FxTypeLayout(
        continuous=(
            FxSlot(0, "gain_p1"),
            FxSlot(1, "amount_p2"),
            FxSlot(2, "distort"),
            FxSlot(5, "amount_p6"),
            FxSlot(6, "sensitivity"),
            FxSlot(7, "gain_p8"),
            FxSlot(8, "dull"),
            FxSlot(9, "gain_p10"),
            FxSlot(10, "mix"),
        ),
        discrete=(
            FxSlot(3, "fx_g2_bonsai_bias_filter"),
            FxSlot(4, "fx_g2_bonsai_type"),
        ),
    ),
}


def shared_native_index(type_name: str, param_name: str, label: str) -> int:
    """Map a shared selector's canonical label to the selected type's native index.

    :param type_name: SurgePy display name of the selected FX type.
    :param param_name: Shared selector, ``fx_g2_lfo_waveform`` or ``fx_g2_drive_model``.
    :param label: Canonical label from ``LFO_WAVEFORMS`` or ``DRIVE_MODELS``.
    :returns: Native integer value to write to that type's slot.
    :raises KeyError: The type has no such selector or the label is not canonical.
    """
    if not any(s.name == param_name for s in SURGE_FX_TYPE_LAYOUTS[type_name].discrete):
        raise KeyError(f"{type_name} has no {param_name}")
    labels = _SHARED_LABELS[param_name]
    if label not in labels:
        raise KeyError(f"{label!r} is not a canonical {param_name} label")
    if type_name == "Waveshaper":
        return _WAVESHAPER_DRIVE_INDEX[label]
    return labels.index(label)


SURGE_FX_PARAM_SPEC = ParamSpec(
    [
        CategoricalParameter(
            name="fx_g2_fx_type",
            values=[
                "Delay",
                "Reverb 1",
                "Phaser",
                "Rotary",
                "Distortion",
                "EQ",
                "Freq Shift",
                "Conditioner",
                "Chorus",
                "Reverb 2",
                "Flanger",
                "Ring Mod",
                "Neuron",
                "Graphic EQ",
                "Resonator",
                "CHOW",
                "Exciter",
                "Ensemble",
                "Combulator",
                "Treemonster",
                "Waveshaper",
                "Mid-Side Tool",
                "Spring Reverb",
                "Bonsai",
            ],
            raw_values=[
                0.039138,
                0.073276,
                0.107414,
                0.141552,
                0.17569,
                0.209828,
                0.243966,
                0.278103,
                0.312241,
                0.380517,
                0.414655,
                0.448793,
                0.517069,
                0.551207,
                0.585345,
                0.619483,
                0.653621,
                0.687759,
                0.721897,
                0.82431,
                0.858448,
                0.892586,
                0.926724,
                0.960862,
            ],
            encoding="onehot",
        ),
        *(ContinuousParameter(name=f"fx_g2_slot_{i}") for i in range(1, FX_SLOT_COUNT + 1)),
        CategoricalParameter(
            name="fx_g2_reverb_1_room_shape",
            values=["Type 1", "Type 2", "Type 3", "Type 4"],
            raw_values=[0.005, 0.335, 0.665, 0.995],
            encoding="onehot",
        ),
        DiscreteLiteralParameter(name="fx_g2_phaser_count", min=1, max=16),
        CategoricalParameter(name="fx_g2_lfo_waveform", values=LFO_WAVEFORMS, encoding="onehot"),
        CategoricalParameter(name="fx_g2_drive_model", values=DRIVE_MODELS, encoding="onehot"),
        CategoricalParameter(
            name="fx_g2_flanger_mode",
            values=["Dry + Combs", "Combs Only", "Dry + Arp Combs", "Arp Combs Only"],
            raw_values=[0.005, 0.335, 0.665, 0.995],
            encoding="onehot",
        ),
        DiscreteLiteralParameter(name="fx_g2_ring_mod_shape", min=0, max=28),
        DiscreteLiteralParameter(name="fx_g2_ring_mod_unison_voices", min=1, max=16),
        CategoricalParameter(
            name="fx_g2_resonator_mode",
            values=["Lowpass", "Bandpass", "Bandpass+Notch", "Highpass"],
            raw_values=[0.005, 0.335, 0.665, 0.995],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="fx_g2_chow_flip",
            values=["Off", "On"],
            raw_values=[0.25, 0.75],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="fx_g2_ensemble_type",
            values=[
                "Digital Delay",
                "BBD 128 Stages",
                "BBD 256 Stages",
                "BBD 512 Stages",
                "BBD 1024 Stages",
                "BBD 2048 Stages",
                "BBD 4096 Stages",
            ],
            raw_values=[0.005, 0.17, 0.335, 0.5, 0.665, 0.83, 0.995],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="fx_g2_mid_side_tool_matrix",
            values=["L-R > M-S > L-R", "L-R > M-S", "M-S > L-R"],
            raw_values=[0.005, 0.5, 0.995],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="fx_g2_bonsai_bias_filter",
            values=["Tilt", "Pull Mids"],
            raw_values=[0.005, 0.995],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="fx_g2_bonsai_type",
            values=["Gentle", "Normal", "Hard", "Harsh"],
            raw_values=[0.005, 0.335, 0.665, 0.995],
            encoding="onehot",
        ),
    ],
    note_params=[],
)
