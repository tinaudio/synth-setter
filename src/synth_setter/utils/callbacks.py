"""Lightning callbacks for plots, prediction dumps, and checkpoint mirroring.

Typical usage::

    checkpoint = ModelCheckpoint(save_last=True, save_on_exception=True)
    uploader = CheckpointUploader("r2://bucket/checkpoints/run", checkpoint)
    trainer = Trainer(callbacks=[checkpoint, uploader])
"""

import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback, Checkpoint, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib.figure import Figure

from synth_setter.data.vst import param_specs
from synth_setter.models.components.transformer import (
    ApproxEquivTransformer,
    LearntProjection,
)
from synth_setter.models.ksin_flow_matching_module import KSinFlowMatchingModule
from synth_setter.models.vst_flow_matching_module import VSTFlowMatchingModule
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.subprocess_stream import STDERR_TAIL_CHARS

log = logging.getLogger(__name__)

# Bound on consecutive failed uploads of one unchanged checkpoint before backing
# off until the file next changes — caps R2 retries when R2 is down.
_MAX_UPLOAD_ATTEMPTS = 3

# The single mirrored object name; the whole class contract hinges on this basename.
_LAST_CKPT_NAME = "last.ckpt"
_CheckpointRevision = tuple[str, float, int, int | None]


def _stderr_tail(exc: BaseException) -> str:
    """Return the trailing stderr a subprocess error carries.

    That tail is the child traceback that makes a probe failure diagnosable
    from the run log.

    :param exc: Exception whose optional ``stderr`` attribute to read.
    :returns: Up to the last ``STDERR_TAIL_CHARS`` characters, or ``""`` when absent.
    """
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        # TimeoutExpired attaches undecoded bytes even when the runner asked for text.
        stderr = stderr.decode(errors="replace")
    if not isinstance(stderr, str) or not stderr:
        return ""
    return stderr[-STDERR_TAIL_CHARS:]


class ValidationAlignedModelCheckpoint(ModelCheckpoint):
    """Align monitored weights with validation while preserving recovery cadence."""

    # Lightning has no public split-save API; lockfile upgrades must pass these hook regressions.

    @property
    def state_key(self) -> str:
        """Preserve compatibility with checkpoints written by ModelCheckpoint.

        :returns: Lightning callback-state key used by the base checkpoint class.
        """
        fields = {
            "monitor": self.monitor,
            "mode": self.mode,
            "every_n_train_steps": self._every_n_train_steps,
            "every_n_epochs": self._every_n_epochs,
            "train_time_interval": self._train_time_interval,
        }
        return f"{ModelCheckpoint.__qualname__}{fields!r}"

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Separate recovery saves from monitored top-k selection at step cadence.

        :param trainer: Supplies checkpoint cadence and loop state.
        :param pl_module: Ignored Lightning hook module.
        :param outputs: Ignored Lightning hook payload.
        :param batch: Ignored Lightning hook payload.
        :param batch_idx: Ignored Lightning hook payload.
        """
        if self.monitor is None or self._every_n_train_steps < 1:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            return
        if self._should_skip_saving_checkpoint(trainer):
            return
        if trainer.global_step % self._every_n_train_steps != 0:
            return
        self._save_recovery_checkpoint(trainer)
        self._defer_save_until_validation = True

    def _save_recovery_checkpoint(self, trainer: Trainer) -> None:
        """Write recovery weights even when ``save_last='link'`` selects an older best.

        :param trainer: Active trainer receiving the checkpoint save.
        """
        configured_save_last = self.save_last
        if configured_save_last == "link":
            self.save_last = True
        try:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
        finally:
            self.save_last = configured_save_last

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Rank monitored weights using the metric from this validation event.

        :param trainer: Supplies fresh validation metrics and loop state.
        :param pl_module: Unused Lightning hook module.
        """
        if self.monitor is None or self._every_n_train_steps < 1:
            super().on_validation_end(trainer, pl_module)
            return
        if (
            trainer.fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING
            or trainer.sanity_checking
        ):
            return
        if not self._defer_save_until_validation:
            return
        self._save_topk_checkpoint(trainer, self._monitor_candidates(trainer))
        self._defer_save_until_validation = False


def _checkpoint_save_token(checkpoint_callback: ModelCheckpoint) -> int | None:
    """Return Lightning's completed-save token when available.

    :param checkpoint_callback: Checkpoint writer exposing Lightning's compatibility field.
    :returns: Completed global step, or ``None`` when Lightning no longer exposes the field.
    """
    token = getattr(checkpoint_callback, "_last_global_step_saved", None)
    return token if isinstance(token, int) else None


class CheckpointUploader(Checkpoint):
    """Best-effort rank-0 R2 mirror ordered after ModelCheckpoint saves.

    Uploads are synchronous; prefer single-device or coarse-cadence DDP runs.
    """

    def __init__(
        self, prefix_uri: str, checkpoint_callback: ModelCheckpoint | None = None
    ) -> None:
        """Bind the upload target.

        :param prefix_uri: ``r2://`` directory the run's ``last.ckpt`` uploads under.
        :param checkpoint_callback: Configured writer whose final Lightning topology is verified.
        """
        super().__init__()
        self._dest_uri = f"{prefix_uri.rstrip('/')}/{_LAST_CKPT_NAME}"
        self._checkpoint_callback = checkpoint_callback
        self._uploaded_revision: _CheckpointRevision | None = None
        self._pending_revision: _CheckpointRevision | None = None
        self._attempts = 0
        self._saw_checkpoint = False
        self._warned_ddp = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Reject a replaced or reordered checkpoint writer.

        :param trainer: Trainer containing the final callback topology.
        :param pl_module: Lightning module, unused by this topology check.
        :param stage: Lightning stage, unused by this topology check.
        :raises ValueError: If the bound writer is not the sole preceding ModelCheckpoint.
        """
        if self._checkpoint_callback is None:
            return
        model_checkpoints = [
            callback for callback in trainer.callbacks if isinstance(callback, ModelCheckpoint)
        ]
        if model_checkpoints != [self._checkpoint_callback] or trainer.callbacks.index(
            self._checkpoint_callback
        ) > trainer.callbacks.index(self):
            raise ValueError(
                "Checkpoint durability writer was replaced or reordered by model callbacks"
            )

    def _checkpoint_revision(self, trainer: Trainer) -> tuple[Path, _CheckpointRevision] | None:
        """Return the completed checkpoint path and revision.

        :param trainer: Trainer exposing the active checkpoint writer.
        :returns: Path and revision, or ``None`` when no readable checkpoint exists.
        """
        checkpoint_callback = self._checkpoint_callback or trainer.checkpoint_callback
        source = getattr(checkpoint_callback, "last_model_path", "") or ""
        if not source:
            return None
        self._saw_checkpoint = True
        try:
            stat = Path(source).stat()
        except OSError as exc:  # checkpoint pruned/rotated between save and this hook
            log.debug("Checkpoint %s vanished before upload: %s", source, exc)
            return None
        revision = (
            source,
            stat.st_mtime,
            stat.st_size,
            _checkpoint_save_token(checkpoint_callback),
        )
        return Path(source), revision

    def _try_upload(self, source: Path) -> bool:
        """Upload one revision while containing expected R2 failures.

        :param source: Local checkpoint to upload.
        :returns: Whether the upload succeeded.
        """
        try:
            r2_io.ensure_r2_env_loaded()
        except (RuntimeError, OSError) as exc:
            log.warning("Mid-run checkpoint upload to %s failed: %s", self._dest_uri, exc)
            return False
        try:
            r2_io.upload_to_uri(source, self._dest_uri)
        except (subprocess.CalledProcessError, OSError) as exc:
            log.warning("Mid-run checkpoint upload to %s failed: %s", self._dest_uri, exc)
            return False
        return True

    def _maybe_upload(self, trainer: Trainer) -> None:
        """Upload each newly completed rank-0 checkpoint save with bounded retries.

        :param trainer: The active trainer; supplies the rank flag and the
            ``ModelCheckpoint`` whose ``last_model_path`` is mirrored.
        """
        if not trainer.is_global_zero:
            return
        if not self._warned_ddp and getattr(trainer, "world_size", 1) > 1:
            self._warned_ddp = True
            log.warning(
                "Mid-run checkpoint upload runs synchronously on rank 0 and stalls "
                "other DDP ranks at the next collective while each %s is copied to R2; "
                "prefer a coarse checkpoint cadence.",
                _LAST_CKPT_NAME,
            )
        checkpoint = self._checkpoint_revision(trainer)
        if checkpoint is None:
            return
        source, revision = checkpoint
        if revision == self._uploaded_revision:
            return
        if revision != self._pending_revision:
            self._pending_revision = revision
            self._attempts = 0
        if self._attempts >= _MAX_UPLOAD_ATTEMPTS:
            return
        self._attempts += 1
        if not self._try_upload(source):
            return
        self._uploaded_revision = revision
        log.info("Mid-run checkpoint uploaded to %s", self._dest_uri)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule | None,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Mirror a training-step checkpoint.

        :param trainer: Active trainer.
        :param pl_module: Active Lightning module.
        :param outputs: Unused training-step output.
        :param batch: Unused training batch.
        :param batch_idx: Unused batch index.
        """
        self._maybe_upload(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule | None) -> None:
        """Mirror a validation-end checkpoint.

        :param trainer: Active trainer.
        :param pl_module: Active Lightning module.
        """
        self._maybe_upload(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule | None) -> None:
        """Mirror a training-epoch checkpoint.

        :param trainer: Active trainer.
        :param pl_module: Active Lightning module.
        """
        self._maybe_upload(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule | None) -> None:
        """Flush the final checkpoint.

        :param trainer: Active trainer.
        :param pl_module: Active Lightning module.
        """
        self._maybe_upload(trainer)
        if trainer.is_global_zero and not self._saw_checkpoint:
            log.warning(
                "Mid-run checkpoint upload enabled but ModelCheckpoint wrote no "
                "last_model_path to mirror to %s; set save_last on the ModelCheckpoint.",
                self._dest_uri,
            )

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule | None, exception: BaseException
    ) -> None:
        """Mirror the crash checkpoint.

        :param trainer: Active trainer.
        :param pl_module: Active Lightning module.
        :param exception: Exception that interrupted training.
        """
        self._maybe_upload(trainer)


def _log_figure(trainer: Trainer, key: str, fig: Figure) -> None:
    """Log a matplotlib figure to whichever Lightning loggers support images.

    Lightning's base ``Logger`` only standardizes scalar metrics; image APIs
    differ per backend, so dispatch is required:
    ``WandbLogger.log_image`` vs ``TensorBoardLogger.experiment.add_figure``.

    Any logger outside the two handled here is **intentionally skipped**
    (silent no-op) — ``CSVLogger`` has no image API, and callers treat that
    as expected. If a new image-capable logger is introduced (MLflow,
    Comet, Neptune, ...), add an ``isinstance`` branch below rather than
    adding the call at the callback site.

    Only rank 0 emits — TensorBoard's ``SummaryWriter`` is not rank-safe,
    and duplicate emissions from every DDP worker would corrupt event
    files or log the same figure N times.
    """
    if not trainer.is_global_zero:
        return
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            logger.log_image(key=key, images=[fig], step=trainer.global_step)
        elif isinstance(logger, TensorBoardLogger):
            logger.experiment.add_figure(key, fig, global_step=trainer.global_step)


class PlotLossPerTimestep(Callback):
    """Plot validation loss as a function of the flow-matching timestep ``t``.

    Runs a single validation batch through the model at ``num_timesteps`` different ``t``
    values and logs a loss-vs-``t`` figure to each attached logger.
    """

    def __init__(self, num_timesteps: int = 100):
        super().__init__()
        self.num_timesteps = num_timesteps

    def _get_val_batch(self, trainer):
        val_dl = trainer.val_dataloaders
        return next(iter(val_dl))

    def _compute_losses(self, trainer, pl_module):
        batch = self._get_val_batch(trainer)
        signal, params, _ = batch

        # Get conditioning vector
        conditioning = pl_module.encoder(signal)
        z = pl_module.vector_field.apply_dropout(conditioning, pl_module.hparams.cfg_dropout_rate)

        x0, x1, z = pl_module._sample_x0_and_x1(params, z)

        losses = []
        for n in range(self.num_timesteps):
            t = torch.full(
                (signal.shape[0], 1), n / (self.num_timesteps - 1), device=signal.device
            )
            x_t = pl_module._sample_probability_path(x0, x1, t)
            target = pl_module._evaluate_target_field(x0, x1, x_t, t)

            prediction = pl_module.vector_field(x_t, t, z)
            loss = (prediction - target).square().mean(dim=-1)
            losses.append(loss)

        return torch.stack(losses, dim=-1)

    def _aggregate_losses(self, losses):
        mean = losses.mean(dim=0)
        std = losses.std(dim=0)
        lower_ci = mean - 2 * std
        upper_ci = mean + 2 * std
        return mean, lower_ci, upper_ci

    def _plot_losses(self, losses):
        t = np.linspace(0, 1, self.num_timesteps)
        mean, lower_ci, upper_ci = self._aggregate_losses(losses)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t, mean.cpu().numpy())
        ax.fill_between(t, lower_ci.cpu().numpy(), upper_ci.cpu().numpy(), alpha=0.2)
        ax.set_xlabel("t")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per noise level / timestep")
        return fig

    def _log_plot(self, fig, trainer):
        _log_figure(trainer, "plot", fig)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        losses = self._compute_losses(trainer, pl_module)
        fig = self._plot_losses(losses)
        self._log_plot(fig, trainer)


def _self_similarity(x):
    y = x.permute(1, 0, 2)
    sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return sim


class PlotPositionalEncodingSimilarity(Callback):
    """Log a cosine-similarity heatmap of the vector field's positional encoding."""

    def _compute_similarity(self, pl_module):
        if pl_module.vector_field.pe_type == "initial":
            return _self_similarity(pl_module.vector_field.pe.pe)
        elif pl_module.vector_field.pe_type == "layerwise":
            return [_self_similarity(pe.pe) for pe in pl_module.vector_field.pe]

    def _plot_single_similarity(self, sim):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.imshow(sim.cpu().numpy(), vmin=-1, vmax=1, aspect="equal")
        fig.tight_layout()
        fig.suptitle("Positional Encoding Similarity")

        return fig

    def _plot_multiple_similarities(self, sims):
        n_pe = len(sims)
        n_rows = int(np.sqrt(n_pe))
        n_cols = int(np.ceil(n_pe / n_rows))

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

        for i, sim in enumerate(sims):
            ax[i // n_cols, i % n_cols].imshow(sim.cpu().numpy(), vmin=-1, vmax=1, aspect="equal")
            ax[i // n_cols, i % n_cols].set_title(f"PE {i // n_cols}-{i % n_cols}", fontsize=8)

        for i in range(n_pe, n_rows * n_cols):
            ax[i // n_cols, i % n_cols].axis("off")

        fig.tight_layout()
        fig.suptitle("Positional Encoding Similarities")

        return fig

    def _plot_similarity(self, sim):
        if isinstance(sim, torch.Tensor):
            return self._plot_single_similarity(sim)
        else:
            return self._plot_multiple_similarities(sim)

    def _log_plot(self, fig, trainer):
        _log_figure(trainer, "pos_enc_similarity", fig)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not isinstance(pl_module, KSinFlowMatchingModule):
            return

        if not isinstance(pl_module.vector_field, ApproxEquivTransformer):
            return

        if pl_module.vector_field.pe_type == "none":
            return

        pe_sims = self._compute_similarity(pl_module)
        fig = self._plot_similarity(pe_sims)
        self._log_plot(fig, trainer)


class PlotLearntProjection(Callback):
    """Log the learnt parameter-to-token projection matrix as an image."""

    def __init__(
        self,
        after_val: bool = True,
        every_n_steps: int | None = None,
        sort_assignments: bool = True,
    ):
        super().__init__()
        self.after_val = after_val
        self.every_n_steps = every_n_steps
        self.sort_assignments = sort_assignments

    def _get_assignment(self, pl_module):
        return pl_module.vector_field.projection.assignment

    def _sort_assignments(self, assignment):
        assignment = assignment.abs()
        k = torch.arange(assignment.shape[-1], device=assignment.device)[None]
        positional_average = torch.sum(assignment * k, dim=-1) / torch.sum(assignment, dim=-1)
        sorted_idxs = torch.argsort(positional_average)
        assignment = assignment[sorted_idxs]
        return assignment

    def _plot_assignments(self, pl_module):
        assignment = self._get_assignment(pl_module)

        if self.sort_assignments:
            assignment = self._sort_assignments(assignment)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        maxval = assignment.abs().max().item()
        img = ax.imshow(
            assignment.cpu().numpy(),
            aspect="equal",
            vmin=-maxval,
            vmax=maxval,
            cmap="RdBu",
        )
        fig.colorbar(img, ax=ax)

        ax.set_title("Assignment")

        ax.set_xlabel("params")
        ax.set_ylabel("tokens")
        fig.tight_layout()
        fig.suptitle("Learnt Assignment")

        return fig

    def _get_value_similarity(self, pl_module):
        proj = pl_module.vector_field.projection.in_projection  # num_params x d_embed x d_model

        sim_proj = torch.nn.functional.cosine_similarity(proj[None], proj[:, None], dim=-1)

        return sim_proj

    def _get_output_similarity(self, pl_module):
        proj = pl_module.vector_field.projection.out_projection.T  # num_params x d_embed x d_model

        sim_proj = torch.nn.functional.cosine_similarity(proj[None], proj[:, None], dim=-1)

        return sim_proj

    def _plot_projections(self, pl_module):
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))

        val_sim = self._get_value_similarity(pl_module)
        out_sim = self._get_output_similarity(pl_module)

        val_max = val_sim.abs().max()
        out_max = out_sim.abs().max()

        val_im = ax[0].imshow(
            val_sim.cpu().numpy(),
            aspect="equal",
            vmin=-val_max,
            vmax=val_max,
            cmap="RdBu",
        )
        ax[0].set_title("Value Projection")
        ax[0].set_xlabel("params")
        ax[0].set_ylabel("params")

        out_im = ax[1].imshow(
            out_sim.cpu().numpy(),
            aspect="equal",
            vmin=-out_max,
            vmax=out_max,
            cmap="RdBu",
        )
        ax[1].set_title("Out Projection")
        ax[1].set_xlabel("params")
        ax[1].set_ylabel("params")

        # show colorbar
        fig.colorbar(val_im, ax=ax[0])
        fig.colorbar(out_im, ax=ax[1])

        fig.tight_layout()

        return fig

    def _log_plots(self, fig_ass, fig_value, trainer):
        _log_figure(trainer, "assignment", fig_ass)
        _log_figure(trainer, "value", fig_value)

        plt.close(fig_ass)
        plt.close(fig_value)

    def _do_plotting(self, trainer, pl_module):
        if not (
            isinstance(pl_module, KSinFlowMatchingModule)
            or isinstance(pl_module, VSTFlowMatchingModule)
        ):
            return

        if not hasattr(pl_module.vector_field, "projection"):
            return

        if not isinstance(pl_module.vector_field, LearntProjection):
            return

        fig_ass = self._plot_assignments(pl_module)
        fig_value = self._plot_projections(pl_module)
        self._log_plots(fig_ass, fig_value, trainer)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.after_val:
            return

        self._do_plotting(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps is None:
            return

        if trainer.global_step % self.every_n_steps != 0:
            return

        with torch.no_grad():
            self._do_plotting(trainer, pl_module)


class PredictionWriter(BasePredictionWriter):
    """Save per-batch and per-epoch predictions plus target tensors to disk."""

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        prediction, batch = prediction
        torch.save(prediction, os.path.join(self.output_dir, f"pred-{batch_idx}.pt"))
        torch.save(
            batch["audio"],
            os.path.join(self.output_dir, f"target-audio-{batch_idx}.pt"),
        )

        if "params" in batch:
            torch.save(
                batch["params"],
                os.path.join(self.output_dir, f"target-params-{batch_idx}.pt"),
            )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions, batch = predictions
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        torch.save(batch["audio"], os.path.join(self.output_dir, "target-audio.pt"))

        if "params" in batch:
            torch.save(batch["params"], os.path.join(self.output_dir, "target-params.pt"))


class ValAudioProbe(Callback):
    """Render a fixed handful of validation predictions off-loop and log audio metrics.

    ``val/param_mse`` says nothing about how a prediction *sounds*, and rendering the
    whole split through the VST is far too slow to sit in the validation loop. This
    stages the first val batch's leading ``num_samples`` samples, hands them to
    ``probe_fn`` on a single worker thread, and logs the resulting metrics at the
    *next* validation — so the training step never waits on a render.

    At most one probe is in flight: when rendering is slower than the validation
    cadence, epochs are skipped rather than queued, which bounds the backlog to zero.
    A probe failure is logged and swallowed — a qualitative signal must never take a
    training run down with it.
    """

    def __init__(
        self,
        *,
        probe_root: str | Path,
        probe_fn: Callable[[Path, int], dict[str, float]],
        num_samples: int = 5,
    ) -> None:
        """Initialize the probe.

        :param probe_root: Directory each ``step-<global_step>/`` probe is staged under.
        :param probe_fn: Called on a worker thread with the staged probe dir and the
            originating ``global_step``; returns the fully-namespaced metrics to log
            verbatim (e.g. ``{"val_audio/mss_mean": ...}``). Exceptions it raises are
            logged as warnings at the next harvest and never crash the fit loop.
        :param num_samples: Upper bound on samples taken from the first val batch.
        """
        super().__init__()
        self.probe_root = Path(probe_root)
        self.num_samples = num_samples
        self._probe_fn = probe_fn
        # Created lazily: ddp_spawn pickles callbacks, and a live executor can't travel.
        self._pool: ThreadPoolExecutor | None = None
        self._future: Future[dict[str, float]] | None = None
        self._future_dir: Path | None = None
        self._future_step: int | None = None
        self._staged: tuple[Path, int] | None = None

    def __getstate__(self) -> dict[str, object]:
        """Drop process-local state so ddp_spawn can pickle the callback.

        :returns: The instance dict with the executor, in-flight future, and staged-slot fields
            reset; the restored copy starts with a clean slot.
        """
        state = self.__dict__.copy()
        state.update(_pool=None, _future=None, _future_dir=None, _future_step=None, _staged=None)
        return state

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor | None],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Stage the first val batch's leading samples in ``PredictionWriter`` layout.

        :param trainer: Supplies ``global_step`` and the rank/sanity gates.
        :param pl_module: Unused; present for the Lightning hook signature.
        :param outputs: ``validation_step`` return value; must carry ``preds``.
        :param batch: Val batch supplying the ``params`` targets.
        :param batch_idx: Only batch 0 is probed, so the same samples recur each epoch.
        :param dataloader_idx: Unused; present for the Lightning hook signature.
        :raises ValueError: when ``outputs`` has no ``preds`` key or ``batch`` has no
            ``params`` tensor.
        """
        if (
            batch_idx != 0
            or dataloader_idx != 0
            or trainer.sanity_checking
            or not trainer.is_global_zero
        ):
            return
        if not isinstance(outputs, dict) or "preds" not in outputs:
            got = sorted(outputs) if isinstance(outputs, dict) else type(outputs).__name__
            raise ValueError(
                "ValAudioProbe requires validation_step to return a 'preds' key; got "
                f"{got}. Wire the probe to a VST module (vst_ff, "
                "vst_flow_matching, vst_fake_oracle) that returns its predictions."
            )
        params = batch.get("params")
        if params is None:
            raise ValueError(
                "ValAudioProbe requires batch['params'] to re-render the target; got None."
            )

        probe_dir = self.probe_root / f"step-{trainer.global_step}"
        predictions_dir = probe_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        limit = self.num_samples
        # No target-audio tensor: training val batches carry none (read_audio is a
        # predict-only flag), so the probe re-renders the target from its params.
        for name, tensor in (
            ("pred", outputs["preds"]),
            ("target-params", params),
        ):
            torch.save(tensor[:limit].detach().cpu(), predictions_dir / f"{name}-0.pt")
        self._staged = (probe_dir, trainer.global_step)

    def on_validation_epoch_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        """Log a finished probe's metrics, then launch one for the samples just staged.

        :param trainer: Unused; present for the Lightning hook signature.
        :param pl_module: Receives the harvested metrics via ``log_dict``.
        """
        self._harvest(pl_module)
        self._launch()

    def teardown(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        """Stop accepting new probes; the in-flight one is left to finish.

        Deliberately no ``cancel_futures``: at most one probe exists, and its
        subprocesses block interpreter exit until they complete (bounded by their
        scaled timeouts), so the final snapshot still reaches R2 instead of dying
        with the process.

        :param trainer: Unused; present for the Lightning hook signature.
        :param pl_module: Unused; present for the Lightning hook signature.
        :param stage: Unused; present for the Lightning hook signature.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=False)

    def _harvest(self, pl_module: "LightningModule") -> None:
        """Log the in-flight probe's metrics if it has finished; no-op while it runs.

        A harvested probe's local directory is pruned (the R2 snapshot is the
        archive), bounding disk use across arbitrarily long runs; a failed
        probe's directory is kept for debugging.

        :param pl_module: Receives the harvested metrics via ``log_dict``.
        """
        if self._future is None or not self._future.done():
            return
        future, probe_dir, step = self._future, self._future_dir, self._future_step
        self._future, self._future_dir, self._future_step = None, None, None
        try:
            metrics = future.result()
        except Exception as exc:
            tail = _stderr_tail(exc)
            log.warning(
                "val audio probe at step %s failed: %s; skipping its metrics.%s",
                step,
                exc,
                f"\nstderr tail:\n{tail}" if tail else "",
            )
            return
        if probe_dir is not None:
            shutil.rmtree(probe_dir, ignore_errors=True)
        if not metrics:
            return
        pl_module.log_dict(
            {**metrics, "val_audio/probe_step": float(step)},
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
        )

    def _launch(self) -> None:
        """Submit the staged probe, or discard it while an earlier probe is still running."""
        if self._staged is None:
            return
        probe_dir, step = self._staged
        self._staged = None
        if self._future is not None:
            log.info(
                "val audio probe from step %s still running; skipping the step-%s probe.",
                self._future_step,
                step,
            )
            shutil.rmtree(probe_dir, ignore_errors=True)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="val-audio-probe")
        self._future = self._pool.submit(self._probe_fn, probe_dir, step)
        self._future_dir = probe_dir
        self._future_step = step


class LogPerParamMSE(Callback):
    """Log validation-set MSE broken down per parameter dimension of the ParamSpec."""

    def __init__(self, param_spec: str):
        """Select the ParamSpec whose dimension names label emitted metrics.

        :param param_spec: Registered ParamSpec name for validation outputs.
        """
        super().__init__()
        self.param_spec = param_specs[param_spec]

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.per_param_mse = 0.0
        self.count = 0

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        per_param_mse = outputs["per_param_mse"]
        self.per_param_mse += per_param_mse.detach().cpu().numpy()
        self.count += 1

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module,
    ) -> None:
        per_param_mse = self.per_param_mse / self.count
        names = self.param_spec.names
        pl_module.log_dict(
            {f"per_param_mse/{name}": mse for name, mse in zip(names, per_param_mse)},
        )
