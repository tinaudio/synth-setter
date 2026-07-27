"""Simulator-feedback finetuning of a frozen flow-matching module (arXiv 2410.22573).

SPIKE (#2554): trains a small control network on top of a frozen pretrained
:class:`VSTFlowMatchingModule` using ONLINE per-step VST rendering — no caching.
The base velocity ``v`` is corrected as ``v_eff = v + v_c(v, t, c)`` for
``t >= t_min``, where ``c`` encodes the mismatch between the rendered one-step
parameter estimate and the target audio representation.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from lightning import LightningModule

from synth_setter.data.vst.param_spec import decode_model_output
from synth_setter.data.vst.param_spec_registry import resolve_param_spec
from synth_setter.data.vst.renderers import AudioRenderer
from synth_setter.data.vst.shapes import (
    MEL_N_MELS,
    MEL_WINDOW,
    mel_hop_length,
    mel_n_fft,
)
from synth_setter.data.vst_datamodule import load_dataset_statistics
from synth_setter.models.components.vector_field import VectorFieldBlock
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import RenderConfig
from synth_setter.renderer_factory import make_audio_renderer
from synth_setter.synth_spec import SynthName, SynthSpec


class ControlEncoder(torch.nn.Module):
    """CNN over (target, rendered, difference) mel stacks producing the control vector."""

    def __init__(self, mel_channels: int, control_dim: int):
        """Build the conv stack for a ``3 * mel_channels``-channel input.

        :param mel_channels: Channels per mel spectrogram (audio channels).
        :param control_dim: Width of the produced control vector.
        """
        super().__init__()
        in_channels = 3 * mel_channels
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, control_dim),
        )

    def forward(self, rep_target: torch.Tensor, rep_rendered: torch.Tensor) -> torch.Tensor:
        """Encode the target/rendered mel pair into one control vector per sample.

        :param rep_target: Target mel conditioning shaped ``(B, C, n_mels, n_frames)``.
        :param rep_rendered: Rendered-estimate mel with the same shape.
        :returns: Control vectors shaped ``(B, control_dim)``.
        """
        x = torch.cat((rep_target, rep_rendered, rep_target - rep_rendered), dim=1)
        return self.net(x)


class ControlField(torch.nn.Module):
    """Residual-MLP correction ``v_c(v, t, c)`` with a zero-initialized output layer."""

    def __init__(self, field_dim: int, hidden_dim: int, control_dim: int, num_blocks: int):
        """Build the control field over the base velocity.

        :param field_dim: Width of the velocity vector being corrected.
        :param hidden_dim: Hidden width of the residual blocks.
        :param control_dim: Width of the control vector conditioning each block.
        :param num_blocks: Number of residual blocks.
        """
        super().__init__()
        self.input = torch.nn.Linear(field_dim, hidden_dim)
        # +1: time is concatenated onto the control vector, as in VectorField.
        self.blocks = torch.nn.ModuleList(
            [VectorFieldBlock(hidden_dim, control_dim + 1) for _ in range(num_blocks)]
        )
        self.output = torch.nn.Linear(hidden_dim, field_dim)
        # Zero-init so finetuning starts exactly at the frozen base flow.
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, v: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict the velocity correction for one batch.

        :param v: Base velocities shaped ``(B, field_dim)``.
        :param t: Flow times shaped ``(B, 1)``.
        :param c: Control vectors shaped ``(B, control_dim)``.
        :returns: Velocity corrections shaped ``(B, field_dim)``.
        """
        z = torch.cat((c, t), dim=-1)
        y = self.input(v)
        for block in self.blocks:
            y = block(y, z)
        return self.output(y)


class VSTFlowMatchingFinetuneModule(LightningModule):
    """Frozen base flow + trainable control encoder/field fed by online VST renders."""

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None,
        *,
        pretrained_ckpt: str | None = None,
        base_module: VSTFlowMatchingModule | None = None,
        t_min: float = 0.8,
        feedback_enabled: bool = True,
        control_dim: int = 256,
        control_hidden_dim: int = 512,
        control_num_blocks: int = 4,
        param_spec_name: str = "surge_simple",
        plugin_state_path: str = "presets/surge-simple.fxp",
        synth_version: str = "1.3.master.f7b97c68",
        sample_rate: int = 44100,
        channels: int = 2,
        velocity: int = 100,
        signal_duration_seconds: float = 4.0,
        min_note_seconds: float = 0.05,
    ):
        """Load and freeze the base flow; build the trainable control networks.

        :param optimizer: ``functools.partial``-style optimizer factory (Hydra
            ``_partial_: true``); receives only the trainable control parameters.
        :param scheduler: ``functools.partial``-style scheduler factory or ``None``.
        :param pretrained_ckpt: Checkpoint the frozen base flow is loaded from.
        :param base_module: Pre-built base flow (tests); mutually exclusive default
            path is ``pretrained_ckpt``.
        :param t_min: Lower edge of the finetune time window; the control field
            only acts on ``t in [t_min, 1]``.
        :param feedback_enabled: When ``False``, the control vector is zeros and no
            rendering happens (ablation with identical control-field capacity).
        :param control_dim: Width of the control vector.
        :param control_hidden_dim: Hidden width of the control field.
        :param control_num_blocks: Residual blocks in the control field.
        :param param_spec_name: Registry key decoding model rows to native params.
        :param plugin_state_path: Baseline preset for the surgepy renderer.
        :param synth_version: Expected surgepy engine version.
        :param sample_rate: Render sample rate in Hz.
        :param channels: Render channel count.
        :param velocity: MIDI velocity for every feedback render.
        :param signal_duration_seconds: Render duration in seconds.
        :param min_note_seconds: Minimum sanitized note length; predicted rows can
            decode to invalid note windows (start >= end), which the renderer rejects.
        :raises ValueError: If neither ``pretrained_ckpt`` nor ``base_module`` is given.
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["base_module"])

        if base_module is None:
            if pretrained_ckpt is None:
                raise ValueError("provide pretrained_ckpt or base_module")
            # weights_only=False unpickles the module graph; checkpoint provenance is
            # deployment-controlled (same trust stance as predict_capture).
            base_module = VSTFlowMatchingModule.load_from_checkpoint(
                pretrained_ckpt, map_location="cpu", weights_only=False
            )
        self.base = base_module
        self.base.freeze()

        self._optimizer_factory = optimizer
        self._scheduler_factory = scheduler
        self._t_min = t_min
        self._feedback_enabled = feedback_enabled
        self._control_dim = control_dim
        self._param_spec_name = ParamSpecName(param_spec_name)
        self._plugin_state_path = plugin_state_path
        self._synth_version = synth_version
        self._sample_rate = sample_rate
        self._channels = channels
        self._velocity = velocity
        self._signal_duration_seconds = signal_duration_seconds
        self._min_note_seconds = min_note_seconds

        field_dim = int(self.base.hparams["num_params"])
        self.control_encoder = ControlEncoder(channels, control_dim)
        self.control_field = ControlField(
            field_dim, control_hidden_dim, control_dim, control_num_blocks
        )

        self._param_spec = resolve_param_spec(self._param_spec_name)
        self._renderer: AudioRenderer | None = None
        self._mel_stats: tuple[np.ndarray, np.ndarray] | None = None
        self._runtime_ready = False

    def train(self, mode: bool = True) -> "VSTFlowMatchingFinetuneModule":
        """Keep the frozen base in eval mode across Lightning's train/eval flips.

        :param mode: Requested train mode for the trainable submodules.
        :returns: This module.
        """
        super().train(mode)
        self.base.eval()
        return self

    def _render_config(self) -> RenderConfig:
        synth = SynthSpec(
            name=SynthName(str(self._param_spec_name)),
            param_spec_name=self._param_spec_name,
            plugin_path="surgepy",
            plugin_state_path=self._plugin_state_path,
            synth_version=self._synth_version,
        )
        return RenderConfig(
            synth=synth,
            renderer_backend="surgepy",
            sample_rate=self._sample_rate,
            channels=self._channels,
            velocity=self._velocity,
            signal_duration_seconds=self._signal_duration_seconds,
            min_loudness=-55.0,
            samples_per_shard=1,
            plugin_reload_cadence="render",
            gui_toggle_cadence="never",
        )

    def _ensure_runtime(self) -> None:
        """Lazily build the renderer and load mel stats after the datamodule hydrates."""
        if self._runtime_ready or not self._feedback_enabled:
            return
        if self._renderer is None:
            self._renderer = make_audio_renderer(self._render_config())
        datamodule = getattr(self.trainer, "datamodule", None) if self._trainer else None
        if (
            self._mel_stats is None
            and datamodule is not None
            and getattr(datamodule, "use_saved_mean_and_variance", False)
        ):
            self._mel_stats = load_dataset_statistics(
                Path(datamodule.dataset_root) / "train.lance"
            )
        self._runtime_ready = True

    def _rendered_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute the dataset-matching normalized log-mel of one rendered note.

        :param audio: Channel-leading rendered audio.
        :returns: Normalized mel shaped ``(channels, n_mels, n_frames)``.
        """
        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self._sample_rate,
            n_mels=MEL_N_MELS,
            n_fft=mel_n_fft(self._sample_rate),
            hop_length=mel_hop_length(self._sample_rate),
            window=MEL_WINDOW,
            center=True,
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        if self._mel_stats is not None:
            mean, std = self._mel_stats
            spec_db = (spec_db - mean) / std
        return spec_db

    def _sanitized_note_window(self, start: float, end: float) -> tuple[float, float]:
        duration = self._signal_duration_seconds
        start = min(max(0.0, start), duration - self._min_note_seconds)
        end = min(max(end, start + self._min_note_seconds), duration)
        return start, end

    def _render_reps(self, theta_hat: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Decode, render, and mel-encode a batch of one-step parameter estimates.

        :param theta_hat: One-step estimates shaped ``(batch, num_params)`` in ``[-1, 1]``.
        :returns: ``(reps, render_seconds, rep_seconds)`` with reps shaped like the
            batch mel conditioning.
        """
        assert self._renderer is not None
        rows = theta_hat.detach().cpu().numpy()
        render_seconds = 0.0
        rep_seconds = 0.0
        reps = []
        for row in rows:
            synth_params, note_params = decode_model_output(row, self._param_spec)
            start, end = self._sanitized_note_window(*note_params["note_start_and_end"])
            t0 = time.perf_counter()
            audio = self._renderer.render(
                synth_params, int(note_params["pitch"]), self._velocity, (start, end)
            )
            t1 = time.perf_counter()
            reps.append(self._rendered_mel(audio))
            render_seconds += t1 - t0
            rep_seconds += time.perf_counter() - t1
        stacked = np.stack(reps).astype(np.float32)
        return torch.from_numpy(stacked).to(self.device), render_seconds, rep_seconds

    def _feedback_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Run one CFM step over the finetune time window with simulator feedback.

        :param batch: Model batch with ``mel_spec``, ``params``, and ``noise``.
        :returns: ``(loss, metrics)`` where metrics include the frozen-base loss and
            per-phase wall times.
        """
        self._ensure_runtime()
        mel = batch["mel_spec"]
        params = batch["params"]
        noise = batch["noise"]

        t0 = time.perf_counter()
        with torch.no_grad():
            t = self._t_min + (1.0 - self._t_min) * torch.rand(
                params.shape[0], 1, device=params.device
            )
            x_t = self.base._rectified_probability_path(noise, params, t)
            target = params - noise
            z = self.base.encoder(mel)
            v = self.base.vector_field(x_t, t, z)
            theta_hat = x_t + (1.0 - t) * v
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        flow_seconds = time.perf_counter() - t0

        render_seconds = 0.0
        rep_seconds = 0.0
        if self._feedback_enabled:
            rep_rendered, render_seconds, rep_seconds = self._render_reps(theta_hat)
            c = self.control_encoder(mel, rep_rendered)
        else:
            c = torch.zeros(params.shape[0], self._control_dim, device=params.device)

        t1 = time.perf_counter()
        v_eff = v + self.control_field(v, t, c)
        loss = (v_eff - target).square().mean(dim=-1).mean()
        control_seconds = time.perf_counter() - t1

        with torch.no_grad():
            base_loss = (v - target).square().mean(dim=-1).mean()

        metrics: dict[str, torch.Tensor | float] = {
            "base_loss": base_loss,
            "time/flow_fwd": flow_seconds,
            "time/render": render_seconds,
            "time/rep": rep_seconds,
            "time/control": control_seconds,
        }
        return loss, metrics

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Optimize the control networks with the feedback-corrected CFM loss.

        :param batch: Model batch with ``mel_spec``, ``params``, and ``noise``.
        :param batch_idx: Lightning batch index (unused).
        :returns: Scalar training loss.
        """
        step_start = time.perf_counter()
        loss, metrics = self._feedback_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/base_loss", metrics["base_loss"], on_step=True, on_epoch=True)
        for key in ("time/flow_fwd", "time/render", "time/rep", "time/control"):
            self.log(f"train/{key}", metrics[key], on_step=True, on_epoch=False)
        self.log("train/time/step", time.perf_counter() - step_start, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Report the feedback-corrected and frozen-base CFM losses on one val batch.

        :param batch: Model batch with ``mel_spec``, ``params``, and ``noise``.
        :param batch_idx: Lightning batch index (unused).
        :returns: Scalar validation loss.
        """
        loss, metrics = self._feedback_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/base_loss", metrics["base_loss"], on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Build the optimizer (and optional scheduler) over the control parameters only.

        :returns: Lightning optimizer configuration.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = self._optimizer_factory(params=trainable)
        if self._scheduler_factory is None:
            return {"optimizer": optimizer}
        scheduler = self._scheduler_factory(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
