import random
from itertools import chain
from typing import Any, List, Literal, Optional, Tuple

import numpy as np

# param representations:
# 1. Synth: dict of str -> float pairs, where the float is on [0, 1]
# 2. Semantic: dict of str -> representation pairs, where the representation takes on
#    the interpretable value of the parameter
# 3. Encoded: NumPy array of values on [0, 1]


class Parameter:
    name: str

    def __init__(self, name: str):
        self.name = name

    def sample(self) -> float:
        raise NotImplementedError


class CategoricalParameter(Parameter):
    def __init__(
        self,
        name: str,
        values: List[Any],
        raw_values: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
        encoding: Literal["scalar", "onehot"] = "scalar",
    ):
        super().__init__(name)

        if raw_values is not None:
            assert len(values) == len(
                raw_values
            ), "values and raw_values must have the same length"

        else:
            n = len(values)
            raw_values = [i / (n - 1) for i in range(n)]

        if weights is not None:
            assert len(values) == len(
                weights
            ), "values and weights must have the same length"

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

    def sample(self) -> float:
        p = np.array(self.weights)
        p /= p.sum()
        return np.random.choice(self.raw_values, p=p)

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

    def sample(self) -> float:
        return np.random.randint(self.min, self.max + 1)

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
    ):
        super().__init__(name)

        assert max > min, "max must be greater than min"
        assert min >= 0.0, "min must be greater than or equal to 0.0"
        assert max <= 1.0, "max must be less than or equal to 1.0"

        self.min = min
        self.max = max

        self.constant_val_p = constant_val_p
        self.constant_val = constant_val

    def __len__(self):
        return 1

    def sample(self) -> float:
        if self.constant_val_p > 0.0 and random.random() < self.constant_val_p:
            return self.constant_val

        return random.uniform(self.min, self.max)

    def encode(self, raw_value: float) -> np.ndarray:
        return (np.array([raw_value]) - self.min) / (self.max - self.min)

    def decode(self, encoded: np.ndarray) -> float:
        return self.min + encoded.item() * (self.max - self.min)

    def __repr__(self):
        return (
            f'ContinuousParameter(name="{self.name}", min={self.min}, max={self.max})'
        )


class NoteDurationParameter(Parameter):
    """A special parameter for sampling note durations."""

    def __init__(self, name: str, max_note_duration_seconds: float):
        super().__init__(name)
        self.max_note_duration_seconds = max_note_duration_seconds

    def __len__(self):
        return 2

    def sample(self) -> Tuple[float, float]:
        start, end = np.sort(
            np.random.uniform(0.0, self.max_note_duration_seconds, size=2)
        ).tolist()

        return start, end

    def encode(self, raw_value: Tuple[float, float]) -> np.ndarray:
        return np.array(raw_value) / self.max_note_duration_seconds

    def decode(self, encoded: np.ndarray) -> Tuple[float, float]:
        return tuple(encoded * self.max_note_duration_seconds)


class ParamSpec:
    def __init__(
        self,
        synth_params: List[Parameter],
        note_params: List[Parameter],
    ):
        self.synth_params = synth_params
        self.note_params = note_params

    @property
    def synth_param_length(self) -> int:
        return sum([len(p) for p in self.synth_params])

    @property
    def note_param_length(self) -> int:
        return sum([len(p) for p in self.note_params])

    def __len__(self):
        return self.synth_param_length + self.note_param_length

    def sample(self) -> Tuple[dict[str, float], dict[str, float]]:
        synth_param_dict = {p.name: p.sample() for p in self.synth_params}
        note_param_dict = {p.name: p.sample() for p in self.note_params}

        return synth_param_dict, note_param_dict

    def encode(
        self, synth_param_dict: dict[str, float], note_param_dict: dict[str, float]
    ) -> np.ndarray:
        synth_params = [p.encode(synth_param_dict[p.name]) for p in self.synth_params]
        note_params = [p.encode(note_param_dict[p.name]) for p in self.note_params]

        synth_params = np.concatenate(synth_params).astype(np.float32)
        note_params = np.concatenate(note_params).astype(np.float32)

        return np.concatenate((synth_params, note_params))

    def decode(self, params: np.ndarray) -> Tuple[dict[str, float], dict[str, float]]:
        synth_params_to_process = [(p, len(p)) for p in self.synth_params]
        note_params_to_process = [(p, len(p)) for p in self.note_params]

        synth_params = {}
        note_params = {}

        pointer = 0
        for param, length in synth_params_to_process:
            param_value = param.decode(params[pointer : pointer + length])
            synth_params[param.name] = param_value
            pointer += length

        for param, length in note_params_to_process:
            param_value = param.decode(params[pointer : pointer + length])
            note_params[param.name] = param_value
            pointer += length

        return synth_params, note_params

    @property
    def synth_param_names(self) -> List[str]:
        return [p.name for p in self.synth_params]

    @property
    def note_param_names(self) -> List[str]:
        return [p.name for p in self.note_params]

    @property
    def names(self) -> List[str]:
        return self.synth_param_names + self.note_param_names
