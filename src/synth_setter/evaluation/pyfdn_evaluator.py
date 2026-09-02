"""Native predicted-patch evaluation for the fixed-source pyFDN instrument."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from jaxtyping import Float32
from lightning import Callback, LightningModule, Trainer

from synth_setter.data.pyfdn_instrument import PyFDNRenderer, params_to_fdn_build
from synth_setter.data.pyfdn_param_spec import PYFDN_N8_MONO_PARAM_SPEC
from synth_setter.data.vst.param_spec import ParameterValues
from synth_setter.evaluation.compute_audio_metrics import (
    compute_mss,
    compute_rms,
    compute_sot,
    compute_wmfcc,
)

_PYFDN_SAMPLE_RATE = 48_000.0
_PREDICTION_ERRORS = (TypeError, ValueError, RuntimeError, OverflowError, FloatingPointError)
type PyFDNRowStatus = Literal["decode_invalid", "build_invalid", "render_invalid", "finite_render"]


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
    :param renderer: Checksum-pinned fixed-source native renderer.
    :returns: Native target audio plus prediction status, audio, and 48 kHz metrics.
    :raises ValueError: Target data, target render, or successful-pair metrics are invalid.
    """
    target_params = decode_pyfdn_model_output(target)
    target_build = params_to_fdn_build(target_params, sample_rate=_PYFDN_SAMPLE_RATE)
    target_audio = renderer.render_build(target_build)
    target_peak, target_rms = _peak_and_rms(target_audio)

    try:
        predicted_params = decode_pyfdn_model_output(prediction)
    except _PREDICTION_ERRORS as exc:
        return PyFDNRowEvaluation(
            status="decode_invalid",
            error=_prediction_error("decode", exc),
            target_params=target_params,
            predicted_params=None,
            target_audio=target_audio,
            predicted_audio=None,
            audio_metrics={},
            target_peak=target_peak,
            target_rms=target_rms,
            predicted_peak=None,
            predicted_rms=None,
        )

    try:
        predicted_build = params_to_fdn_build(predicted_params, sample_rate=_PYFDN_SAMPLE_RATE)
    except _PREDICTION_ERRORS as exc:
        return PyFDNRowEvaluation(
            status="build_invalid",
            error=_prediction_error("build", exc),
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
        predicted_audio = renderer.render_build(predicted_build)
    except _PREDICTION_ERRORS as exc:
        return PyFDNRowEvaluation(
            status="render_invalid",
            error=_prediction_error("render", exc),
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

    predicted_peak, predicted_rms = _peak_and_rms(predicted_audio)
    audio_metrics = {
        "mss": float(compute_mss(target_audio, predicted_audio, _PYFDN_SAMPLE_RATE)),
        "wmfcc": float(compute_wmfcc(target_audio, predicted_audio, _PYFDN_SAMPLE_RATE)),
        "sot": float(compute_sot(target_audio, predicted_audio, _PYFDN_SAMPLE_RATE)),
        "rms_cosine": float(compute_rms(target_audio, predicted_audio, _PYFDN_SAMPLE_RATE)),
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


def _distribution_metrics(
    name: str, values: list[float]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return scalar and tabular population summaries for one nonempty distribution.

    :param name: Published metric name.
    :param values: Nonempty finite observations.
    :returns: Namespaced scalar metrics and one aggregated-metrics table row.
    """
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=0))
    return (
        {f"{name}_mean": mean, f"{name}_std": std},
        {name: {"mean": mean, "std": std}},
    )


class PyFDNEvaluation(Callback):
    """Evaluate test predictions through native pyFDN and persist row artifacts."""

    def __init__(
        self,
        source_audio_path: str | Path,
        source_audio_sha256: str,
        output_dir: str | Path,
        *,
        synth_version: str = "0.4.2",
        sample_rate: int = 48_000,
    ) -> None:
        """Bind the fixed source and output root.

        :param source_audio_path: Checksum-pinned lossless mono source.
        :param source_audio_sha256: Expected SHA-256 of the source bytes.
        :param output_dir: Evaluation run directory.
        :param synth_version: Required installed pyFDN version.
        :param sample_rate: Fixed source and metric rate in Hz.
        :raises ValueError: The sample rate differs from the fixed 48 kHz contract.
        """
        super().__init__()
        if sample_rate != int(_PYFDN_SAMPLE_RATE):
            raise ValueError("pyFDN evaluation sample_rate must be 48000")
        self.output_dir = Path(output_dir)
        self.renderer = PyFDNRenderer(
            source_audio_path,
            source_audio_sha256,
            synth_version=synth_version,
            sample_rate=sample_rate,
        )
        self._reset()

    def _reset(self) -> None:
        """Reset all row and parameter accumulators for one test epoch."""
        self.rows: list[dict[str, Any]] = []
        self.parameter_squared_error = np.zeros(
            PYFDN_N8_MONO_PARAM_SPEC.encoded_width, dtype=np.float64
        )
        self.parameter_finite_count = 0

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset state and reject colliding distributed artifact writers.

        :param trainer: Active Lightning trainer.
        :param pl_module: Unused model hook argument.
        :raises ValueError: Test evaluation uses more than one process.
        """
        del pl_module
        if trainer.world_size != 1:
            raise ValueError("pyFDN evaluation artifacts require trainer.world_size == 1")
        self._reset()

    def evaluate_batch(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Evaluate and persist one ordered batch of model-space rows.

        :param predictions: Float32 prediction matrix shaped ``(batch, 91)``.
        :param targets: Float32 target matrix with the same shape.
        :raises TypeError: Either matrix is not float32.
        :raises ValueError: Batch shapes violate the width-91 contract.
        """
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
            sample_id = len(self.rows)
            if np.isfinite(prediction).all():
                self.parameter_squared_error += np.square(
                    prediction.astype(np.float64) - target.astype(np.float64)
                )
                self.parameter_finite_count += 1
            result = evaluate_pyfdn_row(prediction, target, renderer=self.renderer)
            finite_prediction = prediction[np.isfinite(prediction)]
            pred_min = float(finite_prediction.min()) if finite_prediction.size else None
            pred_max = float(finite_prediction.max()) if finite_prediction.size else None
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "status": result.status,
                "error": result.error,
                "pred_min": pred_min,
                "pred_max": pred_max,
                "pred_out_of_range_count": int(
                    np.count_nonzero((prediction < -1.0) | (prediction > 1.0))
                ),
                "target_peak": result.target_peak,
                "target_rms": result.target_rms,
                "pred_peak": result.predicted_peak,
                "pred_rms": result.predicted_rms,
                **result.audio_metrics,
            }
            self.rows.append(row)
            if result.status == "finite_render":
                self._write_success_artifacts(sample_id, prediction, target, result)

    def _write_success_artifacts(
        self,
        sample_id: int,
        prediction: np.ndarray,
        target: np.ndarray,
        result: PyFDNRowEvaluation,
    ) -> None:
        """Write native float WAVs and coordinate parameters for one successful row.

        :param sample_id: Sequential test-row identity.
        :param prediction: Model-space prediction row.
        :param target: Model-space target row.
        :param result: Successful native evaluation result.
        :raises ValueError: A successful result lacks predicted parameters or audio.
        """
        if result.predicted_audio is None or result.predicted_params is None:
            raise ValueError("finite_render result must carry predicted artifacts")
        sample_dir = self.output_dir / "audio" / f"sample_{sample_id}"
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
            outputs["preds"].detach().to(device="cpu", dtype=torch.float32).numpy(),
            batch["params"].detach().to(device="cpu", dtype=torch.float32).numpy(),
        )

    def finalize(self) -> dict[str, float]:
        """Write CSV artifacts and return aggregate scalar metrics.

        :returns: Counts, rates, parameter errors, audio metrics, and amplitude summaries.
        :raises ValueError: No test rows were evaluated.
        """
        if not self.rows:
            raise ValueError("pyFDN evaluation received no test rows")
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rows = pd.DataFrame(self.rows)
        rows.to_csv(metrics_dir / "metrics.csv", index=False)

        total = len(self.rows)
        status_counts = Counter(row["status"] for row in self.rows)
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

        parameter_rows: list[dict[str, Any]] = []
        if self.parameter_finite_count:
            coordinate_mse = self.parameter_squared_error / self.parameter_finite_count
            metrics["pyfdn/parameter_mse"] = float(coordinate_mse.mean())
            for name, value in zip(
                PYFDN_N8_MONO_PARAM_SPEC.encoded_names, coordinate_mse, strict=True
            ):
                metrics[f"pyfdn/parameter_mse/coordinate/{name}"] = float(value)
                parameter_rows.append({"coordinate": name, "mse": float(value)})
            spans = {
                parameter.name: span
                for parameter, span in PYFDN_N8_MONO_PARAM_SPEC.encoded_slices()
            }
            for field in (
                "delays",
                "feedback_matrix",
                "input_matrix",
                "output_matrix",
                "direct_matrix",
            ):
                metrics[f"pyfdn/parameter_mse/field/{field}"] = float(
                    coordinate_mse[spans[field]].mean()
                )
            rt_slice = slice(
                spans["post_delay.rt_dc_seconds"].start,
                spans["post_delay.rt_nyquist_seconds"].stop,
            )
            metrics["pyfdn/parameter_mse/field/post_delay_rt_controls"] = float(
                coordinate_mse[rt_slice].mean()
            )
        pd.DataFrame(parameter_rows, columns=["coordinate", "mse"]).to_csv(
            metrics_dir / "parameter_metrics.csv", index=False
        )

        aggregated_rows: dict[str, dict[str, float]] = {}
        distributions = {
            "pyfdn/amplitude/target_peak": rows["target_peak"].dropna().tolist(),
            "pyfdn/amplitude/target_rms": rows["target_rms"].dropna().tolist(),
            "pyfdn/amplitude/pred_peak": rows["pred_peak"].dropna().tolist(),
            "pyfdn/amplitude/pred_rms": rows["pred_rms"].dropna().tolist(),
        }
        for column in ("mss", "wmfcc", "sot", "rms_cosine"):
            values = rows[column].dropna().tolist() if column in rows else []
            if values:
                distributions[f"pyfdn/audio/{column}"] = values
        for name, values in distributions.items():
            if not values:
                continue
            scalar_metrics, table_row = _distribution_metrics(name, values)
            metrics.update(scalar_metrics)
            aggregated_rows.update(table_row)
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
