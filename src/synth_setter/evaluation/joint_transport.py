"""Exact joint transport on globally normalized time-frequency energy.

Usage: ``compute_joint_time_frequency_ot(target[None, :], pred[None, :], 44100)``
for one-dimensional float impulse responses. Gain is unrestricted, not PCM-clipped.
See ``docs/reference/reverb-metrics.md`` for the representation and ground cost.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.optimize import linprog
from scipy.signal import stft
from scipy.sparse import coo_matrix

_STFT_N_FFT = 2048
_STFT_HOP_LENGTH = 512
_MAX_FREQUENCY_BANDS = 32
_MAX_TIME_CELLS = 64
_DEFAULT_TIME_SCALE_SECONDS = 0.1
_DEFAULT_LOG_FREQUENCY_SCALE_OCTAVES = 1.0


def _validate_axis_coordinates(
    coordinates: np.ndarray,
    expected_size: int,
    name: str,
) -> np.ndarray:
    """Validate one physical coordinate per ordered grid cell.

    :param coordinates: Candidate one-dimensional coordinate array.
    :param expected_size: Required number of coordinates.
    :param name: Argument name included in validation errors.
    :return: Finite, strictly increasing float64 coordinates.
    :raises ValueError: If the coordinates violate the grid contract.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 1 or values.size != expected_size:
        raise ValueError(f"{name} must contain one coordinate per grid cell")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _validate_scale(scale: float, name: str) -> float:
    """Return a finite positive physical scale as a float.

    :param scale: Candidate physical cost scale.
    :param name: Argument name included in validation errors.
    :return: Validated scale as a float.
    :raises ValueError: If ``scale`` is not finite and positive.
    """
    if not isinstance(scale, Real) or isinstance(scale, bool):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(scale)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _normalized_mass_grid(mass: np.ndarray, name: str) -> np.ndarray:
    """Validate and globally normalize a nonnegative mass grid.

    :param mass: Candidate frequency-by-time mass matrix.
    :param name: Argument name included in validation errors.
    :return: Float64 grid with unit total mass.
    :raises ValueError: If the grid is invalid or has no finite positive mass.
    """
    grid = np.asarray(mass, dtype=np.float64)
    if grid.ndim != 2 or 0 in grid.shape:
        raise ValueError(f"{name} must be a non-empty frequency-by-time matrix")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be finite")
    if np.any(grid < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    total = float(np.sum(grid))
    if not np.isfinite(total):
        raise ValueError(f"{name} total mass must be finite")
    if total == 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return grid / total


def _grid_flow_problem(
    shape: tuple[int, int],
    time_coordinates: np.ndarray,
    log_frequency_coordinates: np.ndarray,
) -> tuple[np.ndarray, coo_matrix]:
    """Build sparse incidence constraints and costs for the grid graph.

    :param shape: Frequency and time cell counts.
    :param time_coordinates: Time-cell centres in transport cost units.
    :param log_frequency_coordinates: Frequency-cell centres in transport cost units.
    :return: Directed-edge costs and sparse node-edge incidence matrix.
    """
    frequency_cells, time_cells = shape
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    costs: list[float] = []

    def append_edge(first_node: int, second_node: int, edge_cost: float) -> None:
        """Append both nonnegative flow variables for an undirected edge.

        :param first_node: First endpoint's flattened grid index.
        :param second_node: Second endpoint's flattened grid index.
        :param edge_cost: Transport cost in either direction.
        """
        for source, destination in ((first_node, second_node), (second_node, first_node)):
            edge_index = len(costs)
            rows.extend((source, destination))
            columns.extend((edge_index, edge_index))
            values.extend((1.0, -1.0))
            costs.append(edge_cost)

    for frequency_index in range(frequency_cells):
        row_offset = frequency_index * time_cells
        for time_index in range(time_cells - 1):
            cost = time_coordinates[time_index + 1] - time_coordinates[time_index]
            append_edge(row_offset + time_index, row_offset + time_index + 1, cost)

    for frequency_index in range(frequency_cells - 1):
        cost = (
            log_frequency_coordinates[frequency_index + 1]
            - log_frequency_coordinates[frequency_index]
        )
        for time_index in range(time_cells):
            first_node = frequency_index * time_cells + time_index
            second_node = first_node + time_cells
            append_edge(first_node, second_node, cost)

    incidence = coo_matrix(
        (values, (rows, columns)),
        shape=(frequency_cells * time_cells, len(costs)),
        dtype=np.float64,
    )
    return np.asarray(costs, dtype=np.float64), incidence


def compute_grid_wasserstein_distance(
    target_mass: np.ndarray,
    pred_mass: np.ndarray,
    *,
    time_coordinates: np.ndarray,
    log_frequency_coordinates: np.ndarray,
    time_scale_seconds: float = _DEFAULT_TIME_SCALE_SECONDS,
    log_frequency_scale_octaves: float = _DEFAULT_LOG_FREQUENCY_SCALE_OCTAVES,
) -> float:
    """Return exact balanced W1 between two frequency-by-time mass grids.

    Each grid is globally normalized. Transport follows adjacent grid edges in
    both directions, with additive cost ``seconds / time_scale_seconds +
    octaves / log_frequency_scale_octaves``.

    :param target_mass: Nonnegative frequency-by-time target mass.
    :param pred_mass: Nonnegative matrix with the same shape as ``target_mass``.
    :param time_coordinates: Increasing time-cell centres in seconds.
    :param log_frequency_coordinates: Increasing frequency-cell centres in octaves.
    :param time_scale_seconds: Positive time displacement worth one cost unit.
    :param log_frequency_scale_octaves: Positive log-frequency displacement worth one cost unit.
    :returns: Globally normalized balanced Wasserstein-1 distance.
    :raises ValueError: If grids, coordinates, or scales violate the stated constraints.
    :raises RuntimeError: If the exact linear program cannot be solved.
    """
    target = _normalized_mass_grid(target_mass, "target_mass")
    pred = _normalized_mass_grid(pred_mass, "pred_mass")
    if target.shape != pred.shape:
        raise ValueError("target_mass and pred_mass must have the same shape")

    times = _validate_axis_coordinates(time_coordinates, target.shape[1], "time_coordinates")
    frequencies = _validate_axis_coordinates(
        log_frequency_coordinates,
        target.shape[0],
        "log_frequency_coordinates",
    )
    time_scale = _validate_scale(time_scale_seconds, "time_scale_seconds")
    frequency_scale = _validate_scale(
        log_frequency_scale_octaves,
        "log_frequency_scale_octaves",
    )
    if np.array_equal(target, pred):
        return 0.0

    costs, incidence = _grid_flow_problem(
        target.shape,
        times / time_scale,
        frequencies / frequency_scale,
    )
    balance = (target - pred).ravel()
    result = linprog(
        costs,
        A_eq=incidence.tocsr()[:-1],
        b_eq=balance[:-1],
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"joint transport optimization failed: {result.message}")
    return float(result.fun)


def _pool_time_cells(energy: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool contiguous frames by energy sum while retaining time centres.

    :param energy: Frequency-bin energy by STFT frame.
    :param times: STFT frame centres in seconds.
    :return: Energy and coordinates on at most ``_MAX_TIME_CELLS`` time cells.
    """
    if energy.shape[1] <= _MAX_TIME_CELLS:
        return energy, times

    time_groups = np.array_split(np.arange(energy.shape[1]), _MAX_TIME_CELLS)
    pooled_energy = np.stack([np.sum(energy[:, group], axis=1) for group in time_groups], axis=1)
    pooled_times = np.asarray([np.mean(times[group]) for group in time_groups])
    return pooled_energy, pooled_times


def _joint_energy_grid(
    audio: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate channel STFT energy onto the bounded transport grid.

    :param audio: Real channel-first audio.
    :param sample_rate: Sample rate in Hz.
    :return: Energy grid, time centres in seconds, and frequency centres in octaves.
    """
    missing_samples = max(0, _STFT_N_FFT - audio.shape[1])
    padded_audio = np.pad(audio, ((0, 0), (0, missing_samples)))
    frequencies, times, coefficients = stft(
        padded_audio,
        fs=sample_rate,
        window="hann",
        nperseg=_STFT_N_FFT,
        noverlap=_STFT_N_FFT - _STFT_HOP_LENGTH,
        nfft=_STFT_N_FFT,
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    energy = np.sum(np.abs(coefficients) ** 2, axis=0)
    energy[1:-1] *= 2.0
    energy, times = _pool_time_cells(energy, times)

    minimum_positive_frequency = sample_rate / _STFT_N_FFT
    log_edges = np.linspace(
        np.log2(minimum_positive_frequency),
        np.log2(sample_rate / 2.0),
        _MAX_FREQUENCY_BANDS + 1,
    )
    log_centres = (log_edges[:-1] + log_edges[1:]) / 2.0
    positive_log_frequencies = np.log2(frequencies[1:])
    band_indices = np.empty(frequencies.size, dtype=np.intp)
    band_indices[0] = 0
    band_indices[1:] = np.clip(
        np.searchsorted(log_edges, positive_log_frequencies, side="right") - 1,
        0,
        _MAX_FREQUENCY_BANDS - 1,
    )
    grid = np.zeros((_MAX_FREQUENCY_BANDS, energy.shape[1]), dtype=np.float64)
    np.add.at(grid, band_indices, energy)
    return grid, times, log_centres


def _validated_audio_pair(
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate and coerce an equal-shape pair of real channel-first audio.

    :param target: Candidate target audio.
    :param pred: Candidate predicted audio.
    :param sample_rate: Candidate sample rate in Hz.
    :return: Float64 target and predicted audio plus validated sample rate.
    :raises ValueError: If either input violates the public audio contract.
    """
    rate = _validate_scale(sample_rate, "sample_rate")
    target_audio = np.asarray(target)
    pred_audio = np.asarray(pred)
    invalid_dimensions = target_audio.ndim != 2 or pred_audio.ndim != 2
    empty_audio = 0 in target_audio.shape or 0 in pred_audio.shape
    if invalid_dimensions or empty_audio:
        raise ValueError(
            "audio must be non-empty channel-first arrays with shape (channels, samples)"
        )
    if target_audio.shape != pred_audio.shape:
        raise ValueError("target and pred must have the same shape")
    if np.iscomplexobj(target_audio) or np.iscomplexobj(pred_audio):
        raise ValueError("target and pred audio must be real-valued")
    try:
        target_audio = target_audio.astype(np.float64, copy=False)
        pred_audio = pred_audio.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("target and pred audio must be real-valued numeric arrays") from error
    if not np.all(np.isfinite(target_audio)) or not np.all(np.isfinite(pred_audio)):
        raise ValueError("target and pred audio must be finite")
    return target_audio, pred_audio, rate


def compute_joint_time_frequency_ot(
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: float = 44100,
) -> float:
    """Compare channel-first audio by exact joint time-frequency transport.

    The metric globally normalizes linear STFT energy, making it invariant to
    independent nonzero gains. Two silent inputs have distance zero; balanced
    transport between silence and non-silence is undefined.

    :param target: Finite channel-first audio with shape ``(channels, samples)``.
    :param pred: Finite audio with the same shape as ``target``.
    :param sample_rate: Finite positive sample rate in Hz.
    :returns: Joint Wasserstein-1 distance in 0.1-second and one-octave cost units.
    :raises ValueError: If input validation fails or exactly one input is silent.
    """
    target_audio, pred_audio, rate = _validated_audio_pair(target, pred, sample_rate)
    target_peak = float(np.max(np.abs(target_audio)))
    pred_peak = float(np.max(np.abs(pred_audio)))
    if target_peak == 0.0 and pred_peak == 0.0:
        return 0.0
    if target_peak == 0.0 or pred_peak == 0.0:
        raise ValueError("balanced transport is undefined when exactly one input is silent")

    target_grid, times, log_frequencies = _joint_energy_grid(target_audio / target_peak, rate)
    pred_grid, _, _ = _joint_energy_grid(pred_audio / pred_peak, rate)
    return compute_grid_wasserstein_distance(
        target_grid,
        pred_grid,
        time_coordinates=times,
        log_frequency_coordinates=log_frequencies,
    )
