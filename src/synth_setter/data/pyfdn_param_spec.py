"""Fixed parameter distributions for order-8 mono pyFDN instruments.

Example:
    ``PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC.sample(rng)`` draws one native patch.
"""

import numpy as np
from pyFDN import householder_matrix
from scipy.linalg import expm

from synth_setter.data.vst.param_spec import (
    ContinuousArrayParameter,
    ContinuousParameter,
    DiscreteArrayParameter,
    DiscreteLiteralParameter,
    Parameter,
    ParameterValues,
    ParamSpec,
)


PYFDN_ORDER = 8
PYFDN_RT_CROSSOVER_HZ = 6_000.0
PYFDN_RT_DC_NAME = "post_delay.rt_dc_seconds"
PYFDN_RT_MAX_SECONDS = 4.0
PYFDN_RT_MIN_SECONDS = 0.1
PYFDN_RT_NYQUIST_NAME = "post_delay.rt_nyquist_seconds"
PYFDN_RT_GEQ_SECONDS_NAME = "post_delay.geq.rt_seconds"
PYFDN_GEQ_RT_MAX_SECONDS = 5.0
PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME = "post_delay.pitch_shift.transpose_cents"
PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MIN = -1200.0
PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX = 1200.0
PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME = "post_delay.pitch_shift.window_size"
PYFDN_PITCHSHIFT_WINDOW_SIZE_MIN = 256
PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX = 4096
PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME = "post_delay.pitch_shift.active_channels"
# Götz et al., arXiv:2510.23158 §2.2: coprime, log-spaced lengths kept verbatim in samples.
PYFDN_GOTZ_DELAYS = np.array([809, 877, 937, 1049, 1151, 1249, 1373, 1499], dtype=np.int64)
PYFDN_FEEDBACK_SKEW_NAME = "feedback_skew"
PYFDN_FEEDBACK_SKEW_SIZE = PYFDN_ORDER * (PYFDN_ORDER - 1) // 2
PYFDN_FEEDBACK_SKEW_MAX = np.pi
PYFDN_GEQ_SECTIONS = 11
PYFDN_GEQ_GAIN_DB_NAME = "post_delay.geq.gain_db"
# Flat per-pass loss: -0.3 dB on the longest line is RT60 ~6.8 s, -20 dB is ~0.1 s.
PYFDN_GEQ_GAIN_DB_MIN = -20.0
PYFDN_GEQ_GAIN_DB_MAX = -0.3
PYFDN_GEQ_BAND_GAIN_DB_NAME = "post_delay.geq.band_gain_db"
PYFDN_GEQ_BAND_GAIN_DB_MIN = -6.0
PYFDN_GEQ_BAND_GAIN_DB_MAX = 0.0
PYFDN_TONE_GEQ_GAIN_DB_NAME = "input.tone_geq.command_gain_db"
PYFDN_TONE_GEQ_GAIN_DB_MAX = 12.0
PYFDN_DIRECT_DELAY_SAMPLES = 2


_PYFDN_MIDI_STUBS: ParameterValues = {
    "pitch": 0,
    "note_start_and_end": (0.0, 0.0),
}


class PyFDNParamSpec(ParamSpec):
    """Restore fixed feedback and MIDI values outside the learned coordinates."""

    def __init__(
        self,
        synth_params: list[Parameter],
        feedback_matrix: np.ndarray,
    ) -> None:
        """Bind learned parameters to one renderer-native feedback matrix.

        :param synth_params: Parameters represented in each encoded row.
        :param feedback_matrix: Fixed order-8 feedback matrix restored after decoding.
        """
        super().__init__(synth_params=synth_params, note_params=[])
        self._feedback_matrix = feedback_matrix.copy()

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Sample learned fields and restore fixed renderer values.

        :param rng: Optional caller-owned random generator.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().sample(rng)
        synth_params["feedback_matrix"] = self._feedback_matrix.copy()
        return synth_params, _PYFDN_MIDI_STUBS.copy()

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode learned fields and restore fixed renderer values.

        :param params: Encoded learned FDN parameter row shaped ``(self.encoded_width,)``.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().decode(params)
        synth_params["feedback_matrix"] = self._feedback_matrix.copy()
        return synth_params, _PYFDN_MIDI_STUBS.copy()


def _fdn_io_parameters() -> list[Parameter]:
    """Build fresh input, output, and direct gain parameters shared by every topology.

    :returns: B, C, and D parameter definitions in renderer encoding order.
    """
    return [
        ContinuousArrayParameter(
            name="input_matrix", shape=(PYFDN_ORDER, 1), min=-1.0, max=1.0
        ),
        ContinuousArrayParameter(
            name="output_matrix", shape=(1, PYFDN_ORDER), min=-1.0, max=1.0
        ),
        ContinuousArrayParameter(
            name="direct_matrix", shape=(1, 1), min=-1.0, max=1.0
        ),
    ]


def _fdn_matrix_parameters(*, delay_min: int, delay_max: int) -> list[Parameter]:
    """Build fresh common FDN parameters in renderer encoding order.

    :param delay_min: Inclusive delay-line lower bound in samples.
    :param delay_max: Inclusive delay-line upper bound in samples.
    :returns: Delay and B/C/D parameter definitions excluding fixed feedback.
    """
    return [
        DiscreteArrayParameter(
            name="delays", shape=(PYFDN_ORDER,), min=delay_min, max=delay_max
        ),
        *_fdn_io_parameters(),
    ]


_PYFDN_N8_HOUSEHOLDER_FEEDBACK = householder_matrix(np.ones(PYFDN_ORDER, dtype=np.float64))

PYFDN_N8_MONO_HOUSEHOLDER_PARAM_SPEC = PyFDNParamSpec(
    synth_params=[
        *_fdn_matrix_parameters(delay_min=400, delay_max=1200),
        ContinuousParameter(
            name=PYFDN_RT_DC_NAME,
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_RT_MAX_SECONDS,
        ),
        ContinuousParameter(
            name=PYFDN_RT_NYQUIST_NAME,
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_RT_MAX_SECONDS,
        ),
    ],
    feedback_matrix=_PYFDN_N8_HOUSEHOLDER_FEEDBACK,
)

PYFDN_PITCHSHIFT_N8_MONO_HOUSEHOLDER_PARAM_SPEC = PyFDNParamSpec(
    synth_params=[
        *_fdn_matrix_parameters(delay_min=1000, delay_max=6000),
        ContinuousArrayParameter(
            name=PYFDN_RT_GEQ_SECONDS_NAME,
            shape=(10,),
            min=PYFDN_RT_MIN_SECONDS,
            max=PYFDN_GEQ_RT_MAX_SECONDS,
        ),
        ContinuousParameter(
            name=PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_NAME,
            min=PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MIN,
            max=PYFDN_PITCHSHIFT_TRANSPOSE_CENTS_MAX,
        ),
        DiscreteLiteralParameter(
            name=PYFDN_PITCHSHIFT_WINDOW_SIZE_NAME,
            min=PYFDN_PITCHSHIFT_WINDOW_SIZE_MIN,
            max=PYFDN_PITCHSHIFT_WINDOW_SIZE_MAX,
        ),
        DiscreteArrayParameter(
            name=PYFDN_PITCHSHIFT_ACTIVE_CHANNELS_NAME,
            shape=(PYFDN_ORDER,),
            min=0,
            max=1,
        ),
    ],
    feedback_matrix=_PYFDN_N8_HOUSEHOLDER_FEEDBACK,
)


def skew_to_orthogonal(skew: np.ndarray) -> np.ndarray:
    """Map strictly-upper-triangular coordinates to an orthogonal matrix.

    Implements the paper's ``U = exp(Tr(P) - Tr(P)^T)`` parameterisation, so every
    decoded feedback matrix is orthogonal without projection.

    :param skew: Upper-triangle entries shaped ``(PYFDN_FEEDBACK_SKEW_SIZE,)`` in row order.
    :returns: Float64 orthogonal matrix shaped ``(PYFDN_ORDER, PYFDN_ORDER)``.
    """
    upper = np.zeros((PYFDN_ORDER, PYFDN_ORDER), dtype=np.float64)
    upper[np.triu_indices(PYFDN_ORDER, k=1)] = np.asarray(skew, dtype=np.float64)
    return np.asarray(expm(upper - upper.T), dtype=np.float64)


class PyFDNGotzParamSpec(ParamSpec):
    """Derive the orthogonal feedback matrix and restore fixed values outside the codec."""

    def __init__(
        self,
        synth_params: list[Parameter],
        fixed_delays: np.ndarray | None,
    ) -> None:
        """Bind learned parameters to optional fixed delay lengths.

        :param synth_params: Parameters represented in each encoded row.
        :param fixed_delays: Int64 delays ``(PYFDN_ORDER,)`` restored after decoding, or
            ``None`` when ``delays`` is itself a learned parameter.
        """
        super().__init__(synth_params=synth_params, note_params=[])
        self._fixed_delays = None if fixed_delays is None else fixed_delays.copy()

    def _restore(self, synth_params: ParameterValues) -> ParameterValues:
        synth_params["feedback_matrix"] = skew_to_orthogonal(
            np.asarray(synth_params[PYFDN_FEEDBACK_SKEW_NAME])
        )
        if self._fixed_delays is not None:
            synth_params["delays"] = self._fixed_delays.copy()
        return synth_params

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Sample learned fields and derive the renderer-native values.

        :param rng: Optional caller-owned random generator.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().sample(rng)
        return self._restore(synth_params), _PYFDN_MIDI_STUBS.copy()

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode learned fields and derive the renderer-native values.

        :param params: Encoded learned FDN parameter row shaped ``(self.encoded_width,)``.
        :returns: Complete native FDN values and fixed MIDI compatibility values.
        """
        synth_params, _ = super().decode(params)
        return self._restore(synth_params), _PYFDN_MIDI_STUBS.copy()


def _gotz_parameters() -> list[Parameter]:
    """Build fresh Götz-topology parameters shared by both delay variants.

    :returns: Skew feedback coordinates, B/C/D gains, per-line attenuation, and tone GEQ.
    """
    return [
        ContinuousArrayParameter(
            name=PYFDN_FEEDBACK_SKEW_NAME,
            shape=(PYFDN_FEEDBACK_SKEW_SIZE,),
            min=-PYFDN_FEEDBACK_SKEW_MAX,
            max=PYFDN_FEEDBACK_SKEW_MAX,
        ),
        *_fdn_io_parameters(),
        ContinuousArrayParameter(
            name=PYFDN_GEQ_GAIN_DB_NAME,
            shape=(PYFDN_ORDER,),
            min=PYFDN_GEQ_GAIN_DB_MIN,
            max=PYFDN_GEQ_GAIN_DB_MAX,
        ),
        ContinuousArrayParameter(
            name=PYFDN_GEQ_BAND_GAIN_DB_NAME,
            shape=(PYFDN_GEQ_SECTIONS - 1, PYFDN_ORDER),
            min=PYFDN_GEQ_BAND_GAIN_DB_MIN,
            max=PYFDN_GEQ_BAND_GAIN_DB_MAX,
        ),
        ContinuousArrayParameter(
            name=PYFDN_TONE_GEQ_GAIN_DB_NAME,
            shape=(PYFDN_GEQ_SECTIONS,),
            min=-PYFDN_TONE_GEQ_GAIN_DB_MAX,
            max=PYFDN_TONE_GEQ_GAIN_DB_MAX,
        ),
    ]


PYFDN_GOTZ_N8_MONO_FIXED_DELAYS_PARAM_SPEC = PyFDNGotzParamSpec(
    synth_params=_gotz_parameters(),
    fixed_delays=PYFDN_GOTZ_DELAYS,
)

PYFDN_GOTZ_N8_MONO_LEARNED_DELAYS_PARAM_SPEC = PyFDNGotzParamSpec(
    synth_params=[
        DiscreteArrayParameter(
            name="delays",
            shape=(PYFDN_ORDER,),
            min=int(PYFDN_GOTZ_DELAYS.min()),
            max=int(PYFDN_GOTZ_DELAYS.max()),
        ),
        *_gotz_parameters(),
    ],
    fixed_delays=None,
)
