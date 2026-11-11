"""Resonarium 0.1.1 control layout without importing the native extension.

Ranges transcribe upstream ``plugin/Source/Parameters.cpp`` at the pinned revision. Every control
is encoded as its host-normalized position in [0, 1]; :meth:`ResonariumControl.native_value`
applies the JUCE ``NormalisableRange`` mapping and snapping that host automation would.
Controls that cannot affect one fresh note (modulators, macros, host-tempo sync, external input,
sampler, voice count, controls only the editor reads) are omitted, as are the master and bank
input/output gains: they only rescale level, and sampling them over +/-100 dB makes clipping or
silence dominate.
"""

import math
from dataclasses import dataclass

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)

RESONARIUM_VERSION = "0.1.1"
RESONARIUM_NOTE_DURATION_SECONDS = 4.0

_FREQUENCY = (10.0, 20000.0, 0.0, 0.3)
_SWITCH = (0.0, 1.0, 1.0, 1.0)
_UNIT = (0.0, 1.0, 0.0, 1.0)
_UNIT_STEP = (0.0, 1.0, 0.01, 1.0)
_SECONDS = (0.0, 60.0, 0.0, 0.2)
_PERCENT = (0.0, 100.0, 0.0, 1.0)


@dataclass(frozen=True)
class ResonariumControl:
    """One native control and its JUCE ``NormalisableRange``.

    .. attribute :: uid
       Native parameter identifier accepted by ``Resonarium.set_param``.
    .. attribute :: minimum
       Native range start.
    .. attribute :: maximum
       Native range end.
    .. attribute :: interval
       Snapping step in native units; zero is continuous.
    .. attribute :: skew
       JUCE skew factor applied to the normalized position.
    """

    uid: str
    minimum: float
    maximum: float
    interval: float
    skew: float

    @property
    def steps(self) -> int:
        """Return the number of legal values for whole-number steps, else zero.

        :returns: Count of legal native values when the control is an integer enumeration.
        """
        if self.interval < 1:
            return 0
        return round((self.maximum - self.minimum) / self.interval) + 1

    def native_value(self, normalized: float) -> float:
        """Map a host-normalized position to the snapped native value.

        :param normalized: Position in [0, 1].
        :returns: The native value a host automation write of ``normalized`` stores.
        :raises ValueError: ``normalized`` is non-finite or outside [0, 1].
        """
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"{self.uid} must be a finite value in [0, 1]")
        proportion = normalized
        if self.skew != 1.0 and proportion > 0.0:
            proportion = math.exp(math.log(proportion) / self.skew)
        value = self.minimum + (self.maximum - self.minimum) * proportion
        if self.interval > 0.0:
            value = self.minimum + self.interval * math.floor(
                (value - self.minimum) / self.interval + 0.5
            )
        return min(self.maximum, max(self.minimum, value))


def _control(uid: str, bounds: tuple[float, float, float, float]) -> ResonariumControl:
    return ResonariumControl(uid, *bounds)


def _resonator(bank: int, index: int) -> list[ResonariumControl]:
    suffix = f" wb{bank}r{index}"
    return [
        _control("enabled" + suffix, _SWITCH),
        _control("pitch" + suffix, (0.1, 10.0, 0.01, 0.7)),
        _control("resonatorFrequency" + suffix, _FREQUENCY),
        _control("resonatorKeytrack" + suffix, _PERCENT),
        _control("decayTime" + suffix, _SECONDS),
        _control("dispersion" + suffix, _PERCENT),
        _control("decayFilterCutoff" + suffix, _FREQUENCY),
        _control("decayFilterPitch" + suffix, (-60.0, 60.0, 0.01, 1.0)),
        _control("decayFilterResonance" + suffix, (0.0, 100.0, 0.0, 0.25)),
        _control("decayFilterKeytrack" + suffix, _SWITCH),
        _control("svfMode" + suffix, _UNIT_STEP),
        _control("svfType" + suffix, (0.0, 2.0, 1.0, 1.0)),
        _control("postFilterCutoff" + suffix, _FREQUENCY),
        _control("postFilterPitch" + suffix, (-60.0, 60.0, 0.01, 1.0)),
        _control("postFilterResonance" + suffix, (0.01, 800.0, 0.0, 0.2)),
        _control("postFilterMode" + suffix, _UNIT),
        _control("postFilterKeytrack" + suffix, _SWITCH),
        _control("gain" + suffix, (-100.0, 0.0, 0.0, 4.0)),
    ]


def _bank(bank: int) -> list[ResonariumControl]:
    suffix = f" wb{bank}"
    return [
        _control("noteOffset" + suffix, (-36.0, 36.0, 0.0, 1.0)),
        _control("couplingMode" + suffix, (0.0, 2.0, 1.0, 1.0)),
        _control("inputMix" + suffix, _UNIT_STEP),
        _control("cascadeAmount" + suffix, (-1.0, 1.0, 0.01, 1.0)),
        _control("cascadeFilterCutoff" + suffix, _FREQUENCY),
        _control("cascadeFilterResonance" + suffix, (0.01, 800.0, 0.0, 0.4)),
        _control("cascadeFilterMode" + suffix, _UNIT),
    ]


def _exciter_filter(prefix: str) -> list[ResonariumControl]:
    return [
        _control(prefix + "filterType", (0.0, 5.0, 1.0, 1.0)),
        _control(prefix + "frequency", _FREQUENCY),
        _control(prefix + "_resonance", (0.01, 100.0, 0.0, 0.4)),
    ]


def _envelope(prefix: str) -> list[ResonariumControl]:
    return [
        _control(prefix + "enabled", _SWITCH),
        _control(prefix + "attack", _SECONDS),
        _control(prefix + "decay", _SECONDS),
        _control(prefix + "sustain", _PERCENT),
        _control(prefix + "release", _SECONDS),
    ]


def _exciters() -> list[ResonariumControl]:
    return [
        *_exciter_filter("impExciter0"),
        _control("impExciter0 enabled", _SWITCH),
        _control("impExciter0 thickness", (1.0, 10.0, 1.0, 1.0)),
        _control("impExciter0 pickPosition", _UNIT_STEP),
        _control("impExciter0 level", _UNIT_STEP),
        *_exciter_filter("noiseExciter0"),
        *_envelope("noiseExciterEnv0"),
        _control("noiseExciter0enabled", _SWITCH),
        _control("noiseExciter0type", (0.0, 2.0, 1.0, 1.0)),
        _control("noiseExciter0density", _UNIT_STEP),
        _control("noiseExciter0level", _UNIT_STEP),
        *_exciter_filter("impTrainExciter0"),
        *_envelope("impTrainExciterEnv0"),
        _control("impTrainExciter0enabled", _SWITCH),
        _control("impTrainExciter0mode", (0.0, 2.0, 1.0, 1.0)),
        _control("impTrainExciter0rate", (0.1, 1000.0, 0.01, 0.4)),
        _control("impTrainExciter0impulseLength", (1.0, 10.0, 1.0, 1.0)),
        _control("impTrainExciter0staticDensity", _UNIT_STEP),
        _control("impTrainExciter0level", _UNIT_STEP),
    ]


def _global_filter(prefix: str) -> list[ResonariumControl]:
    return [
        _control(prefix + " enable", _SWITCH),
        _control(prefix + " cutoff", _FREQUENCY),
        _control(prefix + " mode", _UNIT),
        _control(prefix + " resonance", (0.0, 100.0, 0.0, 1.0)),
    ]


def _effects() -> list[ResonariumControl]:
    return [
        _control("chorusEnable", _SWITCH),
        _control("chorusRate", (0.0, 100.0, 0.0, 0.3)),
        _control("chorusDepth", _UNIT),
        _control("chorusDelay", (0.0, 0.1, 0.0, 1.0)),
        _control("chorusFeedback", (-1.0, 1.0, 0.0, 1.0)),
        _control("chorusMix", _UNIT_STEP),
        _control("delayEnable", _SWITCH),
        _control("delayTimeL", (0.01, 2.0, 0.0, 0.4)),
        _control("delayTimeR", (0.01, 2.0, 0.0, 0.4)),
        _control("delayPingPong", _SWITCH),
        _control("delayFeedback", _UNIT),
        _control("delayLock", _SWITCH),
        _control("delayMix", _UNIT_STEP),
        _control("distEnable", _SWITCH),
        _control("distType", (0.0, 5.0, 1.0, 1.0)),
        _control("distDrive", (-36.0, 36.0, 0.0, 1.0)),
        _control("distOutputGain", (-36.0, 36.0, 0.0, 1.0)),
        _control("distMix", _UNIT_STEP),
        _control("distCutoff", _FREQUENCY),
        _control("distResonance", (0.01, 15.0, 0.0, 0.4)),
        _control("distFilterMode", _UNIT),
        _control("ampEnable", _SWITCH),
        _control("ampMode", (0.0, 5.0, 1.0, 1.0)),
        _control("ampParamA", _UNIT),
        _control("ampParamB", _UNIT),
        _control("ampParamC", _UNIT),
        _control("ampParamD", _UNIT),
        *_global_filter("Filter 1"),
        *_global_filter("Filter 2"),
        _control("compressorEnable", _SWITCH),
        _control("compressorThreshold", (-60.0, 12.0, 0.0, 1.0)),
        _control("compressorRatio", (1.0, 9.0, 0.0, 1.0)),
        _control("compressorAttack", (1.0, 1000.0, 0.0, 1.0)),
        _control("compressorRelease", (1.0, 1000.0, 0.0, 1.0)),
        _control("phaserEnable", _SWITCH),
        _control("phaserRate", (0.0, 99.99, 0.0, 0.3)),
        _control("phaserDepth", _UNIT),
        _control("phaserCentreFreq", _FREQUENCY),
        _control("phaserFeedback", (-1.0, 1.0, 0.0, 1.0)),
        _control("phaserMix", _UNIT_STEP),
        _control("reverbEnable", _SWITCH),
        _control("reverbDampingFreq", _UNIT),
        _control("reverbDensity", _UNIT),
        _control("reverbBandwidthFreq", _UNIT),
        _control("reverbDecay", _UNIT),
        _control("reverbPredelay", _UNIT),
        _control("reverbSize", _UNIT),
        _control("reverbEarlyMix", _UNIT),
        _control("reverbMix", _UNIT_STEP),
    ]


def _bank_with_resonators(bank: int) -> list[ResonariumControl]:
    resonators = [control for index in range(8) for control in _resonator(bank, index)]
    return [*resonators, *_bank(bank)]


RESONARIUM_CONTROLS: tuple[ResonariumControl, ...] = (
    *_exciters(),
    *(control for bank in range(4) for control in _bank_with_resonators(bank)),
    *_effects(),
)


def _parameter(control: ResonariumControl) -> CategoricalParameter | ContinuousParameter:
    if control.steps:
        positions = [step / (control.steps - 1) for step in range(control.steps)]
        return CategoricalParameter(control.uid, values=positions, raw_values=positions)
    return ContinuousParameter(control.uid)


RESONARIUM_PARAM_SPEC = ParamSpec(
    synth_params=[_parameter(control) for control in RESONARIUM_CONTROLS],
    note_params=[
        DiscreteLiteralParameter("pitch", min=48, max=72),
        NoteDurationParameter(
            "note_start_and_end", max_note_duration_seconds=RESONARIUM_NOTE_DURATION_SECONDS
        ),
    ],
)
