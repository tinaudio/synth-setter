"""Synth/note parameter definitions, sampling, and encoding for VST param specs."""

from collections.abc import Iterator, Mapping
from typing import Any, Literal, TypedDict, cast

import numpy as np

# Synth values are renderer-native; semantic values are interpretable; encoded values use [0, 1].


class Parameter:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def sample(self, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    def encode(self, raw_value: Any) -> np.ndarray:
        raise NotImplementedError


class CategoricalParameter(Parameter):
    def __init__(
        self,
        name: str,
        values: list[Any],
        raw_values: list[Any] | None = None,
        weights: list[float] | None = None,
        encoding: Literal["scalar", "onehot"] = "scalar",
    ):
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

        self.values = values
        self.raw_values = raw_values
        self.weights = weights
        self.encoding = encoding

    def __len__(self):
        if self.encoding == "scalar":
            return 1
        else:
            return len(self.raw_values)

    def sample(self, rng: np.random.Generator) -> Any:
        p = np.array(self.weights)
        p /= p.sum()
        return rng.choice(self.raw_values, p=p)

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

    def encode(self, raw_value: float) -> np.ndarray:
        if self.encoding == "scalar":
            return self._encode_scalar(raw_value)
        else:
            return self._encode_onehot(raw_value)

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

    def encode(self, raw_value: int) -> np.ndarray:
        if self.encoding == "scalar":
            return self._encode_scalar(raw_value)
        else:
            return self._encode_onehot(raw_value)

    def _decode_onehot(self, onehot: np.ndarray) -> int:
        idx = np.argmax(onehot)
        return idx + self.min

    def _decode_scalar(self, scalar: np.ndarray) -> int:
        scaled = scalar * (self.max - self.min) + self.min
        return int(scaled.item())

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

        return rng.uniform(self.min, self.max)

    def encode(self, raw_value: float) -> np.ndarray:
        return (np.array([raw_value]) - self.min) / (self.max - self.min)

    def decode(self, encoded: np.ndarray) -> float:
        return self.min + encoded.item() * (self.max - self.min)

    def __repr__(self):
        return f'ContinuousParameter(name="{self.name}", min={self.min}, max={self.max})'


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

    def encode(self, raw_value: tuple[float, float]) -> np.ndarray:
        return np.array(raw_value) / self.max_note_duration_seconds

    def decode(self, encoded: np.ndarray) -> tuple[float, float]:
        return tuple(encoded * self.max_note_duration_seconds)


# pydoclint check-class-attributes has no sphinx directive for TypedDict fields,
# so DOC601/DOC603 are unsatisfiable here.
class NoteParams(TypedDict):  # noqa: DOC601, DOC603
    """Note-conditioning params consumed by ``render_params``.

    Closed and total: ``ParamSpec.sample`` and ``ParamSpec.decode`` emit exactly these two keys.
    """

    pitch: int
    note_start_and_end: tuple[float, float]


class ParamSpec:
    def __init__(
        self, synth_params: list[Parameter], note_params: list[Parameter]
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
    ) -> tuple[dict[str, float], NoteParams]:
        """Draw one synth/note param set, every parameter drawing from ``rng``.

        :param rng: Generator all parameters draw from; ``None`` uses a fresh
            non-deterministic one (pass a seeded one for reproducible draws).
        :returns: ``(synth_param_dict, note_params)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        synth_param_dict = {p.name: p.sample(rng) for p in self.synth_params}
        note_param_dict = {p.name: p.sample(rng) for p in self.note_params}

        # Keys come from runtime ``Parameter.name`` values, so the checker can't
        # prove the NoteParams key->type mapping; assert it at this one source.
        return synth_param_dict, cast(NoteParams, note_param_dict)

    def encode(
        self, synth_param_dict: dict[str, float], note_param_dict: Mapping[str, object]
    ) -> np.ndarray:
        synth_params = [p.encode(synth_param_dict[p.name]) for p in self.synth_params]
        note_params = [p.encode(note_param_dict[p.name]) for p in self.note_params]

        synth_params = np.concatenate(synth_params).astype(np.float32)
        note_params = np.concatenate(note_params).astype(np.float32)

        return np.concatenate((synth_params, note_params))

    def decode(self, params: np.ndarray) -> tuple[dict[str, float], NoteParams]:
        """Decode one encoded row of values in ``[0, 1]``.

        Raw model outputs live in ``[-1, 1]`` and must go through
        :func:`decode_model_output` instead. Width is not validated: the row
        is consumed in per-parameter slices, so a wrong-width row truncates or
        raises late (behavior pinned in ``tests/data/vst/test_param_spec.py``).

        :param params: Encoded row (output of :meth:`encode`), nominally ``len(self)`` wide.
        :returns: ``(synth_param_dict, note_params)``.
        """
        # Split positionally, not by name: encoded_slices() yields synth spans first,
        # and a synth and note parameter may legitimately share a name.
        spans = list(self.encoded_slices())
        synth_spans = spans[: len(self.synth_params)]
        note_spans = spans[len(self.synth_params) :]

        synth_params = {param.name: param.decode(params[span]) for param, span in synth_spans}
        note_params = {param.name: param.decode(params[span]) for param, span in note_spans}

        # Same cast as sample(): keys come from runtime ``Parameter.name`` values,
        # so the checker can't prove the NoteParams key->type mapping.
        return synth_params, cast(NoteParams, note_params)

    @property
    def synth_param_names(self) -> list[str]:
        return [p.name for p in self.synth_params]

    @property
    def note_param_names(self) -> list[str]:
        return [p.name for p in self.note_params]

    @property
    def names(self) -> list[str]:
        return self.synth_param_names + self.note_param_names


def decode_model_output(row: np.ndarray, spec: ParamSpec) -> tuple[dict[str, float], NoteParams]:
    """Invert the model-output scale and decode one prediction row.

    Model prediction rows live in ``[-1, 1]``; the encoded param domain is
    ``[0, 1]``, so the row is rescaled via ``(x + 1) / 2`` and clipped before
    :meth:`ParamSpec.decode`.

    :param row: One prediction row, nominally ``(len(spec),)`` wide, values in
        ``[-1, 1]``; width is not enforced (see :meth:`ParamSpec.decode`).
    :param spec: Spec the model was trained against.
    :returns: ``(synth_param_dict, note_params)``; synth values are in the
        selected renderer's native parameter domains, while note params contain
        pitch as int and start/end seconds.
    """
    scaled = np.clip((row + 1) / 2, 0, 1)
    return spec.decode(scaled)
