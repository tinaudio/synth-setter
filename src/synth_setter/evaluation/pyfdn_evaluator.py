"""Native predicted-patch evaluation for the fixed-source pyFDN instrument.

Example:
    ``PyFDNEvaluation(PyFDNRenderer(), output_dir)`` evaluates test-step predictions.
"""

import base64
import hashlib
import inspect
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from jaxtyping import Float32
from lightning import Callback, LightningModule, Trainer
from pydantic import BaseModel, ConfigDict

import synth_setter.data.pyfdn_param_spec as pyfdn_param_spec_module
from synth_setter.data.pyfdn_instrument import PyFDNRenderer, params_to_fdn_build
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.pyfdn_source import PYFDN_SOURCE_SAMPLE_RATE_HZ
from synth_setter.data.vst.param_spec import ParameterValues
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)

_PYFDN_SAMPLE_RATE = float(PYFDN_SOURCE_SAMPLE_RATE_HZ)
_AUDIO_METRICS = {
    "mss": compute_mss,
    "rms_cosine": compute_rms,
    "sot": compute_sot,
    "wmfcc": compute_wmfcc,
}
_PREDICTION_ERRORS = (TypeError, ValueError, RuntimeError, OverflowError, FloatingPointError)
type PyFDNRowStatus = Literal["decode_invalid", "build_invalid", "render_invalid", "finite_render"]


class _EvaluationRow(BaseModel):
    """Strict per-prediction record persisted to ``metrics.csv``.

    .. attribute :: model_config

       Strict extra-forbidding Pydantic boundary configuration.

    .. attribute :: sample_id

       Sequential row identity.

    .. attribute :: status

       Terminal prediction stage.

    .. attribute :: error

       Stage-qualified failure detail.

    .. attribute :: pred_min

       Minimum finite model prediction.

    .. attribute :: pred_max

       Maximum finite model prediction.

    .. attribute :: pred_out_of_range_count

       Number of finite coordinates outside ``[-1, 1]``.

    .. attribute :: target_peak

       Native target peak magnitude.

    .. attribute :: target_rms

       Native target RMS amplitude.

    .. attribute :: pred_peak

       Native prediction peak magnitude.

    .. attribute :: pred_rms

       Native prediction RMS amplitude.

    .. attribute :: mss

       Multi-scale spectral distance.

    .. attribute :: rms_cosine

       RMS-envelope cosine similarity.

    .. attribute :: sot

       Spectral-over-time distance.

    .. attribute :: wmfcc

       Weighted MFCC distance.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    sample_id: int
    status: PyFDNRowStatus
    error: str | None
    pred_min: float | None
    pred_max: float | None
    pred_out_of_range_count: int
    target_peak: float
    target_rms: float
    pred_peak: float | None
    pred_rms: float | None
    mss: float | None
    rms_cosine: float | None
    sot: float | None
    wmfcc: float | None


class _ArtifactDigests(BaseModel):
    """Digests for one finite render's native artifacts.

    .. attribute :: model_config

       Strict extra-forbidding Pydantic boundary configuration.

    .. attribute :: params_csv

       SHA-256 of the coordinate table.

    .. attribute :: pred_wav

       SHA-256 of the prediction waveform.

    .. attribute :: target_wav

       SHA-256 of the target waveform.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    params_csv: str
    pred_wav: str
    target_wav: str


class _RowCheckpoint(BaseModel):
    """Durable row commit used to resume interrupted evaluation.

    .. attribute :: model_config

       Strict extra-forbidding Pydantic boundary configuration.

    .. attribute :: source_provenance

       Canonical source identity used for this row.

    .. attribute :: evaluation_fingerprint

       Digest of evaluator, metric, codec, and renderer implementation files.

    .. attribute :: prediction_base64

       Exact prediction float32 bytes.

    .. attribute :: target_base64

       Exact target float32 bytes.

    .. attribute :: row

       Validated row result and metrics.

    .. attribute :: artifacts

       Digests for finite-render artifacts.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    source_provenance: dict[str, str | float | int | bool | None]
    evaluation_fingerprint: str
    prediction_base64: str
    target_base64: str
    row: _EvaluationRow
    artifacts: _ArtifactDigests | None


class _ParameterMetricRow(TypedDict):
    """Typed coordinate-error record persisted to ``parameter_metrics.csv``.

    .. attribute :: coordinate

       ParamSpec coordinate name.

    .. attribute :: mse

       Mean squared model-space error.
    """

    coordinate: str
    mse: float


@dataclass(frozen=True, slots=True)
class PyFDNRowEvaluation:
    """Result of evaluating one target/prediction pair through native pyFDN.

    .. attribute :: status

       Terminal prediction stage.

    .. attribute :: error

       Stage-qualified failure, or ``None`` after a finite render.

    .. attribute :: target_params

       Strictly decoded target patch.

    .. attribute :: predicted_params

       Strictly decoded prediction, or ``None`` after decode failure.

    .. attribute :: target_audio

       Native-amplitude target rerender.

    .. attribute :: predicted_audio

       Native-amplitude prediction rerender, or ``None`` after failure.

    .. attribute :: audio_metrics

       Native-waveform 48 kHz behavioral metrics.

    .. attribute :: target_peak

       Target peak magnitude.

    .. attribute :: target_rms

       Target root-mean-square amplitude.

    .. attribute :: predicted_peak

       Prediction peak magnitude, or ``None`` without finite audio.

    .. attribute :: predicted_rms

       Prediction root-mean-square amplitude, or ``None`` without finite audio.
    """

    status: PyFDNRowStatus
    error: str | None
    target_params: ParameterValues
    predicted_params: ParameterValues | None
    target_audio: Float32[np.ndarray, "1 192000"]
    predicted_audio: Float32[np.ndarray, "1 192000"] | None
    audio_metrics: dict[str, float]
    target_peak: float
    target_rms: float
    predicted_peak: float | None
    predicted_rms: float | None


def decode_pyfdn_model_output(
    row: Float32[np.ndarray, "91"],
) -> ParameterValues:
    """Decode one exact pyFDN model row without clipping or repair.

    :param row: Float32 model-space coordinates shaped ``(91,)`` in ``[-1, 1]``.
    :returns: Native pyFDN synth parameters; pyFDN has no note parameters.
    :raises TypeError: The row is not a float32 NumPy array.
    :raises ValueError: Shape, finiteness, range, or note parameters violate the contract.
    """
    if not isinstance(row, np.ndarray) or row.dtype != np.float32:
        raise TypeError("pyFDN model output must be a float32 NumPy array")
    if row.shape != (PYFDN_N8_MONO_PARAM_SPEC.encoded_width,):
        raise ValueError(
            "pyFDN model output must have shape "
            f"({PYFDN_N8_MONO_PARAM_SPEC.encoded_width},), got {row.shape}"
        )
    if not np.isfinite(row).all():
        raise ValueError("pyFDN model output must contain only finite values")
    if np.any((row < -1.0) | (row > 1.0)):
        raise ValueError("pyFDN model output values must be within [-1, 1]")

    encoded = (row.astype(np.float64) + 1.0) / 2.0
    synth_params, note_params = PYFDN_N8_MONO_PARAM_SPEC.decode(encoded)
    if note_params:
        raise ValueError("pyFDN ParamSpec must not decode note parameters")
    return synth_params


def _peak_and_rms(audio: np.ndarray) -> tuple[float, float]:
    """Return native peak magnitude and float64 RMS.

    :param audio: Finite channel-first waveform.
    :returns: Peak magnitude and root-mean-square amplitude.
    """
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    return peak, rms


def _prediction_error(stage: str, exc: Exception) -> str:
    """Format one stable row-level prediction failure.

    :param stage: Decode, build, or render stage name.
    :param exc: Contract exception raised by that stage.
    :returns: Stage-qualified exception type and message.
    """
    return f"{stage}: {type(exc).__name__}: {exc}"


def evaluate_pyfdn_row(
    prediction: Float32[np.ndarray, "91"],
    target: Float32[np.ndarray, "91"],
    *,
    renderer: PyFDNRenderer,
) -> PyFDNRowEvaluation:
    """Evaluate one exact prediction against a target rerendered through the same source.

    Target failures are dataset invariant violations and propagate. Prediction failures are
    classified at decode, build, or render without repair.

    :param prediction: Predicted model-space coordinates.
    :param target: Ground-truth model-space coordinates.
    :param renderer: Canonical-source native renderer.
    :returns: Native target audio plus prediction status, audio, and 48 kHz metrics.
    :raises ValueError: Target data, target render, or successful-pair metrics are invalid.
    """
    target_params = decode_pyfdn_model_output(target)
    target_build = params_to_fdn_build(target_params, sample_rate=_PYFDN_SAMPLE_RATE)
    target_audio = renderer.render_build(target_build)
    target_peak, target_rms = _peak_and_rms(target_audio)

    predicted_params: ParameterValues | None = None

    def failed(status: PyFDNRowStatus, stage: str, exc: Exception) -> PyFDNRowEvaluation:
        """Package a terminal prediction failure.

        :param status: Terminal evaluator status.
        :param stage: Failing stage name.
        :param exc: Boundary exception.
        :returns: Row result retaining all available target and prediction state.
        """
        return PyFDNRowEvaluation(
            status=status,
            error=_prediction_error(stage, exc),
            target_params=target_params,
            predicted_params=predicted_params,
            target_audio=target_audio,
            predicted_audio=None,
            audio_metrics={},
            target_peak=target_peak,
            target_rms=target_rms,
            predicted_peak=None,
            predicted_rms=None,
        )

    try:
        predicted_params = decode_pyfdn_model_output(prediction)
    except _PREDICTION_ERRORS as exc:
        return failed("decode_invalid", "decode", exc)

    try:
        predicted_build = params_to_fdn_build(predicted_params, sample_rate=_PYFDN_SAMPLE_RATE)
    except _PREDICTION_ERRORS as exc:
        return failed("build_invalid", "build", exc)

    try:
        predicted_audio = renderer.render_build(predicted_build)
    except _PREDICTION_ERRORS as exc:
        return failed("render_invalid", "render", exc)

    predicted_peak, predicted_rms = _peak_and_rms(predicted_audio)
    audio_metrics = {
        name: float(metric(target_audio, predicted_audio, _PYFDN_SAMPLE_RATE))
        for name, metric in _AUDIO_METRICS.items()
    }
    if not all(np.isfinite(value) for value in audio_metrics.values()):
        raise ValueError("pyFDN audio metrics must contain only finite values")
    return PyFDNRowEvaluation(
        status="finite_render",
        error=None,
        target_params=target_params,
        predicted_params=predicted_params,
        target_audio=target_audio,
        predicted_audio=predicted_audio,
        audio_metrics=audio_metrics,
        target_peak=target_peak,
        target_rms=target_rms,
        predicted_peak=predicted_peak,
        predicted_rms=predicted_rms,
    )


def _native_coordinate_values(params: ParameterValues) -> np.ndarray:
    """Flatten native values in ParamSpec coordinate order.

    :param params: Decoded pyFDN parameter mapping.
    :returns: Float64 values shaped ``(91,)``.
    """
    values: list[np.ndarray] = []
    for parameter, _ in PYFDN_N8_MONO_PARAM_SPEC.encoded_slices():
        values.append(np.asarray(params[parameter.name], dtype=np.float64).reshape(-1))
    return np.concatenate(values)


def _distribution_metrics(values: list[float]) -> dict[str, float]:
    """Return a population summary for one nonempty distribution.

    :param values: Nonempty finite observations.
    :returns: Mean and population standard deviation.
    """
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def _parameter_metrics(
    squared_error: np.ndarray, finite_count: int
) -> tuple[dict[str, float], list[_ParameterMetricRow]]:
    """Aggregate parameter errors over finite model predictions.

    :param squared_error: Per-coordinate accumulated squared error shaped ``(91,)``.
    :param finite_count: Number of finite prediction rows.
    :returns: Scalar metrics and per-coordinate CSV rows.
    """
    if not finite_count:
        return {}, []

    coordinate_mse = squared_error / finite_count
    metrics = {"pyfdn/parameter_mse": float(coordinate_mse.mean())}
    rows: list[_ParameterMetricRow] = []
    for name, value in zip(PYFDN_N8_MONO_PARAM_SPEC.encoded_names, coordinate_mse, strict=True):
        metrics[f"pyfdn/parameter_mse/coordinate/{name}"] = float(value)
        rows.append({"coordinate": name, "mse": float(value)})

    spans = {parameter.name: span for parameter, span in PYFDN_N8_MONO_PARAM_SPEC.encoded_slices()}
    for field in (
        "delays",
        "direct_matrix",
        "feedback_matrix",
        "input_matrix",
        "output_matrix",
    ):
        metrics[f"pyfdn/parameter_mse/field/{field}"] = float(coordinate_mse[spans[field]].mean())
    rt_slice = slice(
        spans["post_delay.rt_dc_seconds"].start,
        spans["post_delay.rt_nyquist_seconds"].stop,
    )
    metrics["pyfdn/parameter_mse/field/post_delay_rt_controls"] = float(
        coordinate_mse[rt_slice].mean()
    )
    return metrics, rows


def _distribution_aggregates(
    rows: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Aggregate behavioral and native-amplitude distributions.

    :param rows: Per-prediction evaluator records.
    :returns: Scalar metrics and rows for ``aggregated_metrics.csv``.
    """
    distributions = {
        "pyfdn/amplitude/pred_peak": rows["pred_peak"].dropna().tolist(),
        "pyfdn/amplitude/pred_rms": rows["pred_rms"].dropna().tolist(),
        "pyfdn/amplitude/target_peak": rows["target_peak"].dropna().tolist(),
        "pyfdn/amplitude/target_rms": rows["target_rms"].dropna().tolist(),
    }
    for column in _AUDIO_METRICS:
        values = rows[column].dropna().tolist() if column in rows else []
        if values:
            distributions[f"pyfdn/audio/{column}"] = values

    metrics: dict[str, float] = {}
    aggregated_rows: dict[str, dict[str, float]] = {}
    for name, values in distributions.items():
        if not values:
            continue
        summary = _distribution_metrics(values)
        metrics[f"{name}_mean"] = summary["mean"]
        metrics[f"{name}_std"] = summary["std"]
        aggregated_rows[name] = summary
    return metrics, aggregated_rows


def _sha256(path: Path) -> str:
    """Hash one evaluator-owned artifact.

    :param path: Existing file to hash.
    :returns: Lowercase SHA-256 hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_fingerprint() -> str:
    """Identify code that determines decoding, rendering, and metric values.

    Exact replayed inputs make checkpoint and Hydra identities redundant for row reuse; changes to
    evaluator, ParamSpec, instrument, or metric implementation bytes invalidate progress instead.

    :returns: SHA-256 over the relevant implementation files.
    """
    paths = (
        Path(__file__),
        Path(inspect.getfile(compute_mss)),
        Path(inspect.getfile(params_to_fdn_build)),
        Path(inspect.getfile(pyfdn_param_spec_module)),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _artifact_digests(sample_dir: Path) -> _ArtifactDigests:
    """Hash all native artifacts for one finite render.

    :param sample_dir: Completed sample artifact directory.
    :returns: Strict artifact digest record.
    """
    return _ArtifactDigests(
        params_csv=_sha256(sample_dir / "params.csv"),
        pred_wav=_sha256(sample_dir / "pred.wav"),
        target_wav=_sha256(sample_dir / "target.wav"),
    )


def _progress_row_index(path: Path) -> int:
    """Parse a row index from evaluator progress metadata.

    :param path: ``sample_<index>`` progress path.
    :returns: Parsed nonnegative row index.
    :raises ValueError: The filename has no valid row identity.
    """
    try:
        index = int(path.stem.removeprefix("sample_"))
    except ValueError as exc:
        raise ValueError(f"invalid pyFDN progress filename: {path.name}") from exc
    if index < 0:
        raise ValueError(f"invalid pyFDN progress filename: {path.name}")
    return index


def _recover_pending_artifacts(
    output_dir: Path,
    progress_dir: Path,
    committed_rows: list[_EvaluationRow],
) -> None:
    """Remove interrupted rows and reject untracked native artifacts.

    :param output_dir: Evaluation run directory.
    :param progress_dir: Durable row-metadata directory.
    :param committed_rows: Consecutive validated row commits.
    :raises ValueError: An audio directory has neither a commit nor a pending marker.
    """
    committed_count = len(committed_rows)
    for pending in progress_dir.glob("sample_*.pending"):
        sample_id = _progress_row_index(pending)
        if sample_id >= committed_count:
            sample_dir = output_dir / "audio" / f"sample_{sample_id}"
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
        pending.unlink()

    expected_audio = {row.sample_id for row in committed_rows if row.status == "finite_render"}
    audio_root = output_dir / "audio"
    if not audio_root.exists():
        return
    untracked = [
        path
        for path in audio_root.glob("sample_*")
        if _progress_row_index(path) not in expected_audio
    ]
    if untracked:
        raise ValueError(
            "output directory already contains pyFDN evaluation artifacts without "
            "resumable progress: " + ", ".join(str(path) for path in untracked)
        )


def _encode_model_row(row: np.ndarray) -> str:
    """Encode one float32 model row without losing non-finite bit patterns.

    :param row: Model-space row shaped ``(91,)``.
    :returns: Base64-encoded raw float32 bytes.
    """
    return base64.b64encode(row.tobytes()).decode("ascii")


def _evaluation_row(
    sample_id: int,
    prediction: np.ndarray,
    result: PyFDNRowEvaluation,
) -> _EvaluationRow:
    """Build the persisted scalar record for one evaluated prediction.

    :param sample_id: Sequential test-row identity.
    :param prediction: Exact float32 model prediction.
    :param result: Completed native evaluator result.
    :returns: Strict per-row scalar record.
    """
    finite_prediction = prediction[np.isfinite(prediction)]
    pred_min = float(finite_prediction.min()) if finite_prediction.size else None
    pred_max = float(finite_prediction.max()) if finite_prediction.size else None
    return _EvaluationRow(
        sample_id=sample_id,
        status=result.status,
        error=result.error,
        pred_min=pred_min,
        pred_max=pred_max,
        pred_out_of_range_count=int(
            np.count_nonzero(np.isfinite(prediction) & ((prediction < -1.0) | (prediction > 1.0)))
        ),
        target_peak=result.target_peak,
        target_rms=result.target_rms,
        pred_peak=result.predicted_peak,
        pred_rms=result.predicted_rms,
        mss=result.audio_metrics.get("mss"),
        rms_cosine=result.audio_metrics.get("rms_cosine"),
        sot=result.audio_metrics.get("sot"),
        wmfcc=result.audio_metrics.get("wmfcc"),
    )


def _decode_model_row(encoded: str) -> np.ndarray:
    """Decode one checkpointed float32 model row.

    :param encoded: Base64-encoded raw float32 bytes.
    :returns: Independent model-space row shaped ``(91,)``.
    :raises ValueError: The payload is malformed or has the wrong width.
    """
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("pyFDN progress row is not valid base64") from exc
    row = np.frombuffer(raw, dtype=np.float32).copy()
    expected_width = PYFDN_N8_MONO_PARAM_SPEC.encoded_width
    if row.shape != (expected_width,):
        raise ValueError(f"pyFDN progress row must have shape ({expected_width},)")
    return row


def _write_success_artifacts(
    output_dir: Path,
    *,
    sample_id: int,
    prediction: np.ndarray,
    target: np.ndarray,
    result: PyFDNRowEvaluation,
) -> _ArtifactDigests:
    """Write native float WAVs and coordinate parameters for one successful row.

    :param output_dir: Evaluation run directory.
    :param sample_id: Sequential test-row identity.
    :param prediction: Model-space prediction row.
    :param target: Model-space target row.
    :param result: Successful native evaluation result.
    :returns: SHA-256 digests for the completed artifacts.
    :raises ValueError: A successful result lacks predicted parameters or audio.
    """
    if result.predicted_audio is None or result.predicted_params is None:
        raise ValueError("finite_render result must carry predicted artifacts")
    sample_dir = output_dir / "audio" / f"sample_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sf.write(
        sample_dir / "pred.wav",
        result.predicted_audio.T,
        int(_PYFDN_SAMPLE_RATE),
        subtype="FLOAT",
    )
    sf.write(
        sample_dir / "target.wav",
        result.target_audio.T,
        int(_PYFDN_SAMPLE_RATE),
        subtype="FLOAT",
    )
    encoded_prediction = (prediction.astype(np.float64) + 1.0) / 2.0
    encoded_target = (target.astype(np.float64) + 1.0) / 2.0
    pd.DataFrame(
        {
            "coordinate": PYFDN_N8_MONO_PARAM_SPEC.encoded_names,
            "pred_model": prediction,
            "target_model": target,
            "pred_encoded": encoded_prediction,
            "target_encoded": encoded_target,
            "pred_native": _native_coordinate_values(result.predicted_params),
            "target_native": _native_coordinate_values(result.target_params),
        }
    ).to_csv(sample_dir / "params.csv", index=False)
    return _artifact_digests(sample_dir)


class PyFDNEvaluation(Callback):
    """Evaluate pyFDN predictions with durable per-row progress and native artifacts."""

    def __init__(
        self,
        renderer: PyFDNRenderer,
        output_dir: str | Path,
    ) -> None:
        """Bind an injected fixed-source renderer and output root.

        :param renderer: Native renderer whose source is owned by instrument configuration.
        :param output_dir: Evaluation run directory.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.renderer = renderer
        self._evaluation_fingerprint = _evaluation_fingerprint()
        self._reset()

    def _reset(self) -> None:
        """Reset row progress and parameter accumulators for one test epoch."""
        self.rows: list[_EvaluationRow] = []
        self.parameter_squared_error = np.zeros(
            PYFDN_N8_MONO_PARAM_SPEC.encoded_width, dtype=np.float64
        )
        self.parameter_finite_count = 0
        self._committed_inputs: list[tuple[np.ndarray, np.ndarray]] = []
        self._input_row_index = 0
        self._progress_loaded = False

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore durable progress and reject colliding distributed writers.

        :param trainer: Active Lightning trainer.
        :param pl_module: Unused model hook argument.
        :raises ValueError: Test evaluation uses more than one process or progress is invalid.
        """
        del pl_module
        if trainer.world_size != 1:
            raise ValueError("pyFDN evaluation artifacts require trainer.world_size == 1")
        self._reset()
        self._restore_progress()

    def _require_tracked_output(self) -> None:
        """Reject aggregate outputs that have no resumable row commits.

        :raises ValueError: Aggregate outputs exist without progress metadata.
        """
        owned_paths = (
            self.output_dir / "metrics" / "metrics.csv",
            self.output_dir / "metrics" / "aggregated_metrics.csv",
            self.output_dir / "metrics" / "parameter_metrics.csv",
        )
        existing = [path for path in owned_paths if path.exists()]
        if existing:
            raise ValueError(
                "output directory already contains pyFDN evaluation artifacts without "
                "resumable progress: " + ", ".join(str(path) for path in existing)
            )

    def _restore_progress(self) -> None:
        """Restore consecutive source-matched row commits from disk.

        :raises ValueError: Progress is malformed, nonconsecutive, or source-mismatched.
        """
        if self._progress_loaded:
            return
        progress_dir = self.output_dir / "metrics" / "rows"
        paths = list(progress_dir.glob("sample_*.json"))
        if not paths:
            _recover_pending_artifacts(self.output_dir, progress_dir, [])
            self._require_tracked_output()
            self._progress_loaded = True
            return

        paths.sort(key=_progress_row_index)
        indices = [_progress_row_index(path) for path in paths]
        if indices != list(range(len(paths))):
            raise ValueError("pyFDN progress rows must be consecutive from sample_0")

        for sample_id, path in enumerate(paths):
            checkpoint = _RowCheckpoint.model_validate_json(path.read_text())
            if checkpoint.source_provenance != self.renderer.source_provenance:
                raise ValueError("pyFDN progress source provenance does not match the renderer")
            if checkpoint.evaluation_fingerprint != self._evaluation_fingerprint:
                raise ValueError("pyFDN progress evaluator implementation does not match")
            if checkpoint.row.sample_id != sample_id:
                raise ValueError("pyFDN progress sample identity does not match its filename")
            prediction = _decode_model_row(checkpoint.prediction_base64)
            target = _decode_model_row(checkpoint.target_base64)
            if checkpoint.row.status == "finite_render":
                sample_dir = self.output_dir / "audio" / f"sample_{sample_id}"
                if (
                    checkpoint.artifacts is None
                    or _artifact_digests(sample_dir) != checkpoint.artifacts
                ):
                    raise ValueError(
                        f"pyFDN progress sample_{sample_id} artifacts do not match their digests"
                    )
            elif checkpoint.artifacts is not None:
                raise ValueError(
                    f"pyFDN progress sample_{sample_id} has artifacts for an invalid render"
                )
            self.rows.append(checkpoint.row)
            self._committed_inputs.append((prediction, target))
            self._accumulate_parameter_error(prediction, target)
        _recover_pending_artifacts(self.output_dir, progress_dir, self.rows)
        self._progress_loaded = True

    def _accumulate_parameter_error(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one finite prediction's model-space error.

        :param prediction: Float32 prediction row.
        :param target: Float32 target row.
        """
        if not np.isfinite(prediction).all():
            return
        self.parameter_squared_error += np.square(
            prediction.astype(np.float64) - target.astype(np.float64)
        )
        self.parameter_finite_count += 1

    def _begin_row(self, sample_id: int) -> None:
        """Mark one row in progress before rendering any artifact.

        :param sample_id: Sequential test-row identity.
        """
        progress_dir = self.output_dir / "metrics" / "rows"
        progress_dir.mkdir(parents=True, exist_ok=True)
        (progress_dir / f"sample_{sample_id}.pending").write_text(self._evaluation_fingerprint)

    def _commit_row(
        self,
        *,
        prediction: np.ndarray,
        target: np.ndarray,
        row: _EvaluationRow,
        artifacts: _ArtifactDigests | None,
    ) -> None:
        """Atomically commit one completed evaluator row.

        :param prediction: Exact float32 prediction input.
        :param target: Exact float32 target input.
        :param row: Completed per-row metrics and terminal status.
        :param artifacts: Digests for a finite render, otherwise ``None``.
        """
        progress_dir = self.output_dir / "metrics" / "rows"
        path = progress_dir / f"sample_{row.sample_id}.json"
        temporary = path.with_suffix(".json.tmp")
        checkpoint = _RowCheckpoint(
            source_provenance=self.renderer.source_provenance,
            evaluation_fingerprint=self._evaluation_fingerprint,
            prediction_base64=_encode_model_row(prediction),
            target_base64=_encode_model_row(target),
            row=row,
            artifacts=artifacts,
        )
        temporary.write_text(checkpoint.model_dump_json())
        temporary.replace(path)
        (progress_dir / f"sample_{row.sample_id}.pending").unlink()

    def _evaluate_new_row(
        self,
        *,
        sample_id: int,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Evaluate and durably commit one previously unseen row.

        :param sample_id: Sequential test-row identity.
        :param prediction: Exact float32 model prediction.
        :param target: Exact float32 model target.
        """
        self._begin_row(sample_id)
        result = evaluate_pyfdn_row(prediction, target, renderer=self.renderer)
        row = _evaluation_row(sample_id, prediction, result)
        artifacts = None
        if result.status == "finite_render":
            artifacts = _write_success_artifacts(
                self.output_dir,
                sample_id=sample_id,
                prediction=prediction,
                target=target,
                result=result,
            )
        self._commit_row(
            prediction=prediction,
            target=target,
            row=row,
            artifacts=artifacts,
        )
        self.rows.append(row)
        self._committed_inputs.append((prediction.copy(), target.copy()))
        self._accumulate_parameter_error(prediction, target)

    def evaluate_batch(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Evaluate, resume, and persist one ordered batch of model-space rows.

        :param predictions: Float32 prediction matrix shaped ``(batch, 91)``.
        :param targets: Float32 target matrix with the same shape.
        :raises TypeError: Either matrix is not float32.
        :raises ValueError: Batch shapes or persisted row identities violate the contract.
        """
        self._restore_progress()
        expected_width = PYFDN_N8_MONO_PARAM_SPEC.encoded_width
        if predictions.dtype != np.float32 or targets.dtype != np.float32:
            raise TypeError("pyFDN prediction and target batches must be float32")
        if (
            predictions.ndim != 2
            or targets.shape != predictions.shape
            or predictions.shape[1] != expected_width
        ):
            raise ValueError(
                "pyFDN prediction and target batches must have matching shape "
                f"(batch, {expected_width})"
            )

        for prediction, target in zip(predictions, targets, strict=True):
            sample_id = self._input_row_index
            if sample_id < len(self._committed_inputs):
                committed_prediction, committed_target = self._committed_inputs[sample_id]
                prediction_matches = np.array_equal(
                    prediction, committed_prediction, equal_nan=True
                )
                target_matches = np.array_equal(target, committed_target, equal_nan=True)
                if not prediction_matches or not target_matches:
                    raise ValueError(
                        f"pyFDN resumed input does not match committed sample_{sample_id}"
                    )
                self._input_row_index += 1
                continue

            self._evaluate_new_row(
                sample_id=sample_id,
                prediction=prediction,
                target=target,
            )
            self._input_row_index += 1

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Consume terminal predictions returned by the flow test step.

        :param trainer: Unused Lightning trainer hook argument.
        :param pl_module: Unused model hook argument.
        :param outputs: Test output carrying ``preds``.
        :param batch: Test batch carrying target ``params``.
        :param batch_idx: Unused ordered batch index.
        :param dataloader_idx: Must identify the sole test loader.
        :raises ValueError: Outputs or dataloader identity violate the callback contract.
        """
        del trainer, pl_module, batch_idx
        if dataloader_idx != 0:
            raise ValueError("pyFDN evaluation supports exactly one test dataloader")
        if not isinstance(outputs, dict) or "preds" not in outputs:
            raise ValueError("pyFDN evaluation requires test_step outputs['preds']")
        self.evaluate_batch(
            outputs["preds"].detach().cpu().numpy(),
            batch["params"].detach().cpu().numpy(),
        )

    def finalize(self) -> dict[str, float]:
        """Write CSV artifacts and return aggregate scalar metrics.

        :returns: Counts, rates, parameter errors, audio metrics, and amplitude summaries.
        :raises ValueError: No test rows were evaluated.
        """
        self._restore_progress()
        if not self.rows:
            raise ValueError("pyFDN evaluation received no test rows")
        if self._input_row_index != len(self.rows):
            raise ValueError("pyFDN resumed evaluation did not replay every committed row")
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rows = pd.DataFrame([row.model_dump() for row in self.rows])
        rows.to_csv(metrics_dir / "metrics.csv", index=False)

        total = len(self.rows)
        status_counts = Counter(row.status for row in self.rows)
        decode_invalid = status_counts["decode_invalid"]
        build_invalid = status_counts["build_invalid"]
        render_invalid = status_counts["render_invalid"]
        finite_render = status_counts["finite_render"]
        valid_build = render_invalid + finite_render
        metrics: dict[str, float] = {
            "pyfdn/rows_total": float(total),
            "pyfdn/invalid/decode_count": float(decode_invalid),
            "pyfdn/invalid/build_count": float(build_invalid),
            "pyfdn/invalid/render_count": float(render_invalid),
            "pyfdn/valid_build_count": float(valid_build),
            "pyfdn/finite_render_count": float(finite_render),
            "pyfdn/audio_pair_count": float(finite_render),
            "pyfdn/valid_build_rate": valid_build / total,
            "pyfdn/finite_render_rate": finite_render / total,
            "pyfdn/parameter_finite_count": float(self.parameter_finite_count),
            "pyfdn/parameter_finite_rate": self.parameter_finite_count / total,
        }

        parameter_metrics, parameter_rows = _parameter_metrics(
            self.parameter_squared_error, self.parameter_finite_count
        )
        metrics.update(parameter_metrics)
        pd.DataFrame(parameter_rows, columns=["coordinate", "mse"]).to_csv(
            metrics_dir / "parameter_metrics.csv", index=False
        )

        distribution_metrics, aggregated_rows = _distribution_aggregates(rows)
        metrics.update(distribution_metrics)
        pd.DataFrame.from_dict(aggregated_rows, orient="index").to_csv(
            metrics_dir / "aggregated_metrics.csv"
        )
        return metrics

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Persist aggregates and expose them through Lightning callback metrics.

        :param trainer: Unused Lightning trainer hook argument.
        :param pl_module: Model receiving aggregate scalar logs.
        """
        del trainer
        pl_module.log_dict(self.finalize(), on_step=False, on_epoch=True)
