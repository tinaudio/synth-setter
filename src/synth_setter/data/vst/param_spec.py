"""Synth/note parameter definitions, sampling, and encoding for VST param specs."""

import math
from collections.abc import Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import Literal, TypedDict

import numpy as np

# Synth values are renderer-native; semantic values are interpretable; encoded values use [0, 1].

type ParameterValue = float | int | tuple[float, ...] | np.ndarray
type ParameterValues = dict[str, ParameterValue]

_MAX_EXACT_FLOAT32_INTEGER_SPAN = (1 << 23) - 1
_MAX_EXACT_FLOAT64_INTEGER = 1 << 53


class Parameter:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __len__(self) -> int:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator) -> ParameterValue:
        raise NotImplementedError

    def encode(self, raw_value: object) -> np.ndarray:
        raise NotImplementedError

    def decode(self, encoded: np.ndarray) -> ParameterValue:
        raise NotImplementedError

    def encoded_names(self) -> tuple[str, ...]:
        """Return one stable label for each encoded coordinate.

        :returns: Logical name for a scalar or zero-based coordinate names for an expanded field.
        """
        if len(self) == 1:
            return (self.name,)
        return tuple(f"{self.name}.{index}" for index in range(len(self)))


class CategoricalParameter(Parameter):
    def __init__(
        self,
        name: str,
        values: Sequence[object],
        raw_values: list[float] | None = None,
        weights: list[float] | None = None,
        encoding: Literal["scalar", "onehot"] = "scalar",
    ) -> None:
        super().__init__(name)

        if raw_values is not None:
            assert len(values) == len(raw_values), (
                "values and raw_values must have the same length"
            )

        else:
            n = len(values)
            raw_values = [i / (n - 1) for i in range(n)]

        if weights is not None:
            assert len(values) == len(weights), "values and weights must have the same length"

        else:
            weights = [1.0] * len(values)

        self.values = list(values)
        self.raw_values = raw_values
        self.weights = weights
        self.encoding = encoding

    def __len__(self):
        if self.encoding == "scalar":
            return 1
        else:
            return len(self.raw_values)

    def sample(self, rng: np.random.Generator) -> float:
        p = np.array(self.weights)
        p /= p.sum()
        return float(rng.choice(self.raw_values, p=p))

    def _encode_onehot(self, raw_value: float) -> np.ndarray:
        # find index of nearest raw value
        # make one-hot encoding
        dists = np.abs(np.array(self.raw_values) - raw_value)
        idx = np.argmin(dists)
        onehot = np.zeros(len(self.raw_values))
        onehot[idx] = 1

        return onehot

    def _encode_scalar(self, raw_value: float) -> np.ndarray:
        return np.array([raw_value])

    def encode(self, raw_value: object) -> np.ndarray:
        if not isinstance(raw_value, Real):
            raise TypeError(f"{self.name} must be numeric")
        value = float(raw_value)
        if self.encoding == "scalar":
            return self._encode_scalar(value)
        else:
            return self._encode_onehot(value)

    def _decode_onehot(self, onehot: np.ndarray) -> float:
        idx = np.argmax(onehot)
        return self.raw_values[idx]

    def _decode_scalar(self, scalar: np.ndarray) -> float:
        return scalar.item()

    def decode(self, encoded: np.ndarray) -> float:
        if self.encoding == "scalar":
            return self._decode_scalar(encoded)
        else:
            return self._decode_onehot(encoded)

    def __repr__(self):
        return f'CategoricalParameter(name="{self.name}", values={self.values}, raw_values={self.raw_values})'


class DiscreteLiteralParameter(Parameter):
    def __init__(
        self,
        name: str,
        min: int,
        max: int,
        encoding: Literal["scalar", "onehot"] = "scalar",
    ):
        super().__init__(name)
        self.min = min
        self.max = max
        self.encoding = encoding

    def __len__(self):
        if self.encoding == "scalar":
            return 1
        else:
            return self.max - self.min + 1

    def sample(self, rng: np.random.Generator) -> int:
        # Native int, not np.int64: a sampled pitch flows into mido/pedalboard's
        # MIDI parser, which rejects numpy scalars ("must be bytes or lists of
        # byte values"). ``Generator.integers`` returns np.int64.
        return int(rng.integers(self.min, self.max + 1))

    def _encode_onehot(self, raw_value: int) -> np.ndarray:
        onehot = np.zeros(self.max - self.min + 1)
        onehot[raw_value - self.min] = 1

        return onehot

    def _encode_scalar(self, raw_value: int) -> np.ndarray:
        return (np.array([raw_value]) - self.min) / (self.max - self.min)

    def encode(self, raw_value: object) -> np.ndarray:
        if not isinstance(raw_value, Integral):
            raise TypeError(f"{self.name} must be an integer")
        value = int(raw_value)
        if self.encoding == "scalar":
            return self._encode_scalar(value)
        else:
            return self._encode_onehot(value)

    def _decode_onehot(self, onehot: np.ndarray) -> int:
        idx = np.argmax(onehot)
        return idx + self.min

    def _decode_scalar(self, scalar: np.ndarray) -> int:
        offset = scalar.item() * (self.max - self.min)
        lower_offset = math.floor(offset)
        return self.min + lower_offset + int(offset - lower_offset >= 0.5)

    def decode(self, encoded: np.ndarray) -> int:
        if self.encoding == "scalar":
            return self._decode_scalar(encoded)
        else:
            return self._decode_onehot(encoded)

    def __repr__(self):
        return f'DiscreteParameter(name="{self.name}", min={self.min}, max={self.max})'


class ContinuousParameter(Parameter):
    def __init__(
        self,
        name: str,
        min: float = 0.0,
        max: float = 1.0,
        constant_val_p: float = 0.0,
        constant_val: float = 0.0,
    ) -> None:
        super().__init__(name)

        if not np.isfinite(min) or not np.isfinite(max):
            raise ValueError("bounds must be finite")
        if max <= min:
            raise ValueError("max must be greater than min")
        # Finite bounds can still overflow when normalization subtracts them.
        if not np.isfinite(max - min):
            raise ValueError("span must be finite")
        if not 0.0 <= constant_val_p <= 1.0:
            raise ValueError("constant_val_p must be in [0, 1]")
        if constant_val_p > 0.0 and not min <= constant_val <= max:
            raise ValueError("constant_val must be within [min, max] when enabled")

        self.min = min
        self.max = max
        self.constant_val_p = constant_val_p
        self.constant_val = constant_val

    def __len__(self):
        return 1

    def sample(self, rng: np.random.Generator) -> float:
        if self.constant_val_p > 0.0 and rng.random() < self.constant_val_p:
            return self.constant_val

        return float(rng.uniform(self.min, self.max))

    def encode(self, raw_value: object) -> np.ndarray:
        if not isinstance(raw_value, Real):
            raise TypeError(f"{self.name} must be numeric")
        return (np.array([float(raw_value)]) - self.min) / (self.max - self.min)

    def decode(self, encoded: np.ndarray) -> float:
        return self.min + encoded.item() * (self.max - self.min)

    def __repr__(self):
        return f'ContinuousParameter(name="{self.name}", min={self.min}, max={self.max})'


class ContinuousArrayParameter(Parameter):
    """A fixed-shape continuous parameter encoded elementwise into ``[0, 1]``."""

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        min: float,
        max: float,
    ) -> None:
        """Bind the native shape and finite affine-encoding range.

        :param name: Logical parameter name.
        :param shape: Non-empty native array shape with positive dimensions.
        :param min: Inclusive native lower bound.
        :param max: Inclusive native upper bound.
        :raises ValueError: The shape or bounds cannot define a finite array domain.
        """
        super().__init__(name)
        if not shape or any(
            not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
            for size in shape
        ):
            raise ValueError("shape must contain positive integer dimensions")
        if not np.isfinite(min) or not np.isfinite(max):
            raise ValueError("bounds must be finite")
        if max <= min:
            raise ValueError("max must be greater than min")
        if not np.isfinite(max - min):
            raise ValueError("span must be finite")
        self.shape = tuple(int(size) for size in shape)
        self.min = min
        self.max = max

    def __len__(self) -> int:
        return math.prod(self.shape)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one native float64 array.

        :param rng: Generator that owns the deterministic sample stream.
        :returns: Values shaped ``self.shape`` within the configured native bounds.
        """
        return rng.uniform(self.min, self.max, size=self.shape)

    def encode(self, raw_value: object) -> np.ndarray:
        """Encode a native array as a flat C-order float32 vector.

        :param raw_value: Finite values shaped ``self.shape`` within the native bounds.
        :returns: Values shaped ``(len(self),)`` in ``[0, 1]``.
        :raises ValueError: The input has the wrong shape or invalid values.
        """
        raw = np.asarray(raw_value)
        if raw.shape != self.shape:
            raise ValueError(f"{self.name} must have shape {self.shape}, got {raw.shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"{self.name} must contain only finite values")
        if np.any((raw < self.min) | (raw > self.max)):
            raise ValueError(f"{self.name} values must be within [{self.min}, {self.max}]")
        encoded = (raw.astype(np.float64) - self.min) / (self.max - self.min)
        return encoded.reshape(-1, order="C").astype(np.float32)

    def encoded_names(self) -> tuple[str, ...]:
        """Return one C-order coordinate label per native element.

        :returns: Labels ordered identically to :meth:`encode`.
        """
        return tuple(
            f"{self.name}.{'.'.join(str(coordinate) for coordinate in index)}"
            for index in np.ndindex(self.shape)
        )

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode a flat unit-domain vector to its native float64 shape.

        :param encoded: Finite values shaped ``(len(self),)`` in ``[0, 1]``.
        :returns: Float64 values shaped ``self.shape``.
        :raises ValueError: The input has the wrong shape or invalid values.
        """
        values = np.asarray(encoded)
        expected_shape = (len(self),)
        if values.shape != expected_shape:
            raise ValueError(
                f"encoded {self.name} must have shape {expected_shape}, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"encoded {self.name} must contain only finite values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"encoded {self.name} values must be within [0, 1]")
        raw = self.min + values.astype(np.float64) * (self.max - self.min)
        return raw.reshape(self.shape, order="C")


class DiscreteArrayParameter(ContinuousArrayParameter):
    """A fixed-shape integer parameter encoded elementwise into ``[0, 1]``."""

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        min: int,
        max: int,
    ) -> None:
        """Bind the native shape and inclusive integer range.

        :param name: Logical parameter name.
        :param shape: Non-empty native array shape with positive dimensions.
        :param min: Inclusive native lower bound.
        :param max: Inclusive native upper bound.
        :raises ValueError: The bounds are non-integral or cannot round-trip through float32.
        """
        if any(
            not isinstance(bound, Integral) or isinstance(bound, bool)
            for bound in (min, max)
        ):
            raise ValueError("bounds must be integers")
        min = int(min)
        max = int(max)
        super().__init__(name=name, shape=shape, min=min, max=max)
        if min < -_MAX_EXACT_FLOAT64_INTEGER or max > _MAX_EXACT_FLOAT64_INTEGER:
            raise ValueError("bounds exceed exact float64 integer range")
        if max - min > _MAX_EXACT_FLOAT32_INTEGER_SPAN:
            raise ValueError("range is too wide for exact float32 encoding")

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one native int64 array from the inclusive range.

        :param rng: Generator that owns the deterministic sample stream.
        :returns: Int64 values shaped ``self.shape``.
        """
        return rng.integers(self.min, self.max + 1, size=self.shape, dtype=np.int64)

    def encode(self, raw_value: object) -> np.ndarray:
        """Encode native integers as a flat C-order float32 vector.

        :param raw_value: Integral values shaped ``self.shape`` within the native bounds.
        :returns: Values shaped ``(len(self),)`` in ``[0, 1]``.
        :raises ValueError: The input has the wrong shape or invalid values.
        """
        encoded = super().encode(raw_value)
        raw = np.asarray(raw_value)
        if not np.equal(raw, np.rint(raw)).all():
            raise ValueError(f"{self.name} must contain only integer values")
        return encoded

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode a flat unit-domain vector to its native int64 shape.

        :param encoded: Finite values shaped ``(len(self),)`` in ``[0, 1]``.
        :returns: Int64 values shaped ``self.shape``.
        """
        return np.rint(super().decode(encoded)).astype(np.int64)


class NoteDurationParameter(Parameter):
    """A special parameter for sampling note durations."""

    def __init__(self, name: str, max_note_duration_seconds: float):
        super().__init__(name)
        self.max_note_duration_seconds = max_note_duration_seconds

    def __len__(self):
        return 2

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        start, end = np.sort(rng.uniform(0.0, self.max_note_duration_seconds, size=2)).tolist()

        return start, end

    def encode(self, raw_value: object) -> np.ndarray:
        if not (
            isinstance(raw_value, tuple)
            and len(raw_value) == 2
            and all(isinstance(value, Real) for value in raw_value)
        ):
            raise TypeError(f"{self.name} must be a pair of numeric endpoints")
        return np.array(raw_value, dtype=np.float64) / self.max_note_duration_seconds

    def decode(self, encoded: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in encoded * self.max_note_duration_seconds)


# pydoclint check-class-attributes has no sphinx directive for TypedDict fields,
# so DOC601/DOC603 are unsatisfiable here.
class NoteParams(TypedDict):  # noqa: DOC601, DOC603
    """Complete MIDI note mapping required by VST render boundaries."""

    pitch: int
    note_start_and_end: tuple[float, float]


def require_scalar_synth_params(values: Mapping[str, ParameterValue]) -> dict[str, float]:
    """Validate and normalize synth values for a scalar-only renderer.

    :param values: Renderer-native synth values decoded by a generic parameter spec.
    :returns: Plain float values accepted by legacy VST and TorchSynth renderers.
    :raises TypeError: A synth value is not a real scalar.
    """
    result: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real scalar")
        result[name] = float(value)
    return result


def require_note_params(values: Mapping[str, ParameterValue]) -> NoteParams:
    """Validate and normalize the complete MIDI mapping for a renderer.

    :param values: Renderer-native note values decoded by a generic parameter spec.
    :returns: Canonical Python pitch and note-window values.
    :raises TypeError: The pitch or note window has an invalid native type.
    :raises ValueError: The mapping has missing or extra note fields.
    """
    if set(values) != {"pitch", "note_start_and_end"}:
        raise ValueError("note params must contain exactly pitch and note_start_and_end")

    pitch = values["pitch"]
    if not isinstance(pitch, Integral) or isinstance(pitch, bool):
        raise TypeError("pitch must be an integer")

    window = values["note_start_and_end"]
    if not (
        isinstance(window, tuple)
        and len(window) == 2
        and all(isinstance(endpoint, Real) for endpoint in window)
    ):
        raise TypeError("note_start_and_end must be two numeric endpoints")
    return {
        "pitch": int(pitch),
        "note_start_and_end": (float(window[0]), float(window[1])),
    }


class ParamSpec:
    def __init__(
        self,
        synth_params: list[Parameter],
        note_params: list[Parameter],
    ) -> None:
        self.synth_params = synth_params
        self.note_params = note_params

    @property
    def synth_param_length(self) -> int:
        return sum([len(p) for p in self.synth_params])

    @property
    def note_param_length(self) -> int:
        return sum([len(p) for p in self.note_params])

    @property
    def encoded_width(self) -> int:
        """Return the complete encoded synth-and-note vector width."""
        return self.synth_param_length + self.note_param_length

    def __len__(self) -> int:
        return self.encoded_width

    def encoded_slices(self) -> Iterator[tuple[Parameter, slice]]:
        """Pair each parameter with the columns it occupies in an encoded row.

        Spans are contiguous from 0, ordered ``synth_params`` then ``note_params``
        to match :meth:`encode`, and the final stop equals :attr:`encoded_width`.
        Callers must index encoded rows through these spans: a parameter may own
        several columns (onehot values, note start/end), so there is no positional
        correspondence between :attr:`names` and encoded columns.

        :yields: One ``(parameter, span)`` pair per parameter, in encoding order.
        :ytype: tuple[Parameter, slice]
        """
        pointer = 0
        for param in (*self.synth_params, *self.note_params):
            width = len(param)
            yield param, slice(pointer, pointer + width)
            pointer += width

    @property
    def synth_columns(self) -> slice:
        """Return the encoded columns the synth params occupy.

        Read off :meth:`encoded_slices` so callers splitting an encoded row never restate
        an assumed column order.

        :returns: Contiguous span covering every synth parameter's encoded columns.
        """
        spans = [span for _, span in self.encoded_slices()][: len(self.synth_params)]
        return slice(spans[0].start, spans[-1].stop) if spans else slice(0, 0)

    def sample(
        self, rng: np.random.Generator | None = None
    ) -> tuple[ParameterValues, ParameterValues]:
        """Draw one synth/note param set, every parameter drawing from ``rng``.

        :param rng: Generator all parameters draw from; ``None`` uses a fresh
            non-deterministic one (pass a seeded one for reproducible draws).
        :returns: Separate renderer-native synth and note value mappings; either may be empty.
        """
        if rng is None:
            rng = np.random.default_rng()
        synth_param_dict: ParameterValues = {
            p.name: p.sample(rng) for p in self.synth_params
        }
        note_param_dict: ParameterValues = {
            p.name: p.sample(rng) for p in self.note_params
        }
        return synth_param_dict, note_param_dict

    def encode(
        self,
        synth_param_dict: Mapping[str, object],
        note_param_dict: Mapping[str, object],
    ) -> np.ndarray:
        values = [
            parameter.encode(synth_param_dict[parameter.name])
            for parameter in self.synth_params
        ] + [
            parameter.encode(note_param_dict[parameter.name])
            for parameter in self.note_params
        ]
        if not values:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(values).astype(np.float32)

    def encoded_to_model(self, encoded: np.ndarray) -> np.ndarray:
        """Rescale encoded values onto the ``[-1, 1]`` scale the model predicts in.

        Elementwise, so it applies to a whole row or to any column span of one.

        :param encoded: Values in ``[0, 1]``.
        :returns: The same values on the model's ``[-1, 1]`` scale.
        """
        return encoded * 2 - 1

    def model_to_encoded(self, model: np.ndarray) -> np.ndarray:
        """Rescale model-space values onto the encoded ``[0, 1]`` domain, clipped.

        Inverse of :meth:`encoded_to_model` for in-range inputs; predictions
        overshooting ``[-1, 1]`` saturate rather than decode out of domain.

        :param model: Values on the model's ``[-1, 1]`` scale.
        :returns: The same values clipped into ``[0, 1]``.
        """
        calculation_dtype = np.result_type(model.dtype, np.float64)
        promoted = model.astype(calculation_dtype, copy=False)
        return ((promoted + 1) / 2).clip(0, 1)

    def decode(self, params: np.ndarray) -> tuple[ParameterValues, ParameterValues]:
        """Decode one encoded row of values in ``[0, 1]``.

        Raw model outputs live in ``[-1, 1]`` and must go through
        :func:`decode_model_output` instead. Width is not validated: the row
        is consumed in per-parameter slices, so a wrong-width row truncates or
        raises late (behavior pinned in ``tests/data/vst/test_param_spec.py``).

        :param params: Encoded row (output of :meth:`encode`), nominally ``len(self)`` wide.
        :returns: Separate renderer-native synth and note value mappings; either may be empty.
        """
        # Split positionally, not by name: encoded_slices() yields synth spans first,
        # and a synth and note parameter may legitimately share a name.
        spans = list(self.encoded_slices())
        synth_spans = spans[: len(self.synth_params)]
        note_spans = spans[len(self.synth_params) :]

        synth_params: ParameterValues = {
            param.name: param.decode(params[span]) for param, span in synth_spans
        }
        note_params: ParameterValues = {
            param.name: param.decode(params[span]) for param, span in note_spans
        }
        return synth_params, note_params

    @property
    def synth_param_names(self) -> list[str]:
        return [p.name for p in self.synth_params]

    @property
    def note_param_names(self) -> list[str]:
        return [p.name for p in self.note_params]

    @property
    def names(self) -> list[str]:
        return self.synth_param_names + self.note_param_names

    @property
    def encoded_names(self) -> list[str]:
        """Return labels for every encoded synth and note coordinate.

        :returns: Coordinate labels in the same order as :meth:`encode`.
        """
        names = []
        for parameter in (*self.synth_params, *self.note_params):
            names.extend(parameter.encoded_names())
        return names


def decode_model_output(
    row: np.ndarray, spec: ParamSpec
) -> tuple[ParameterValues, ParameterValues]:
    """Invert the model-output scale and decode one prediction row.

    Model prediction rows live in ``[-1, 1]``; the encoded param domain is
    ``[0, 1]``, so the row goes through :meth:`ParamSpec.model_to_encoded`
    before :meth:`ParamSpec.decode`.

    :param row: One prediction row, nominally ``(len(spec),)`` wide, values in
        ``[-1, 1]``; width is not enforced (see :meth:`ParamSpec.decode`).
    :param spec: Spec the model was trained against.
    :returns: Renderer-native synth and note value mappings; note values may be empty.
    """
    return spec.decode(spec.model_to_encoded(row))
