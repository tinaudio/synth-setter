"""Tests for the ``ValAudioProbe`` callback's scheduling and harvest behavior.

The probe's render/score/upload step is injected via ``probe_fn`` so these tests
run without a VST: the callback's staging, single-slot throttling, harvest, and
rank/sanity gating all execute for real, and only the subprocess chain behind
``run_audio_probe`` is stood in for. The real chain is covered end-to-end by
``test_train.py::test_train_surge_xt_val_audio_probe``.
"""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer

from synth_setter.utils.callbacks import ValAudioProbe

_DRAIN_TIMEOUT_SECONDS = 30


def _drain(probe: ValAudioProbe) -> None:
    """Block until the probe's in-flight worker finishes, keeping tests sleep-free.

    Reaches for the private future deliberately: this is synchronization, not an
    assertion — production never waits on a probe, so there is no public API to use.

    :param probe: Callback whose worker future to wait on.
    """
    if probe._future is not None:
        concurrent.futures.wait([probe._future], timeout=_DRAIN_TIMEOUT_SECONDS)


class _FakeTrainer:
    """Minimal ``Trainer`` stand-in exposing the fields the probe reads."""

    def __init__(self, *, global_step: int = 0, is_global_zero: bool = True) -> None:
        """Initialize the fake with the two gate fields and a step.

        :param global_step: Step stamped into staged probe directory names.
        :param is_global_zero: Rank-0 gate; ``False`` makes every hook a no-op.
        """
        self.global_step = global_step
        self.is_global_zero = is_global_zero
        self.sanity_checking = False


def _trainer(*, global_step: int = 0, is_global_zero: bool = True) -> Trainer:
    """Build a ``_FakeTrainer`` narrowed to ``Trainer`` for the hook signatures.

    :param global_step: Step stamped into the staged probe directory name.
    :param is_global_zero: Rank-0 gate; ``False`` makes every hook a no-op.
    :returns: The fake cast to ``Trainer`` for the call site's type checker.
    """
    return cast("Trainer", _FakeTrainer(global_step=global_step, is_global_zero=is_global_zero))


class _RecordingModule:
    """``LightningModule`` stand-in recording ``log_dict`` payloads."""

    def __init__(self) -> None:
        self.logged: list[dict[str, float]] = []

    def log_dict(
        self,
        payload: dict[str, float],
        *,
        on_step: bool = False,
        on_epoch: bool = True,
        rank_zero_only: bool = True,
    ) -> None:
        """Record the metrics payload the callback harvested.

        :param payload: Metric name to value mapping the callback logged.
        :param on_step: Ignored; accepted to match the Lightning signature.
        :param on_epoch: Ignored; accepted to match the Lightning signature.
        :param rank_zero_only: Ignored; accepted to match the Lightning signature.
        """
        self.logged.append(dict(payload))


def _module() -> LightningModule:
    """Build a ``_RecordingModule`` narrowed to ``LightningModule``.

    :returns: The fake cast to ``LightningModule`` for the call site's type checker.
    """
    return cast("LightningModule", _RecordingModule())


def _batch(rows: int = 8) -> dict[str, torch.Tensor | None]:
    """Return a val batch shaped like ``VSTDataset``'s, with ``rows`` samples.

    ``audio`` is ``None`` as in real training val batches (``read_audio`` is a
    predict-only flag) — the probe must not touch it.

    :param rows: Batch dimension of the params tensor.
    :returns: Batch dict with ``audio`` and ``params`` keys.
    """
    return {
        "audio": None,
        "params": torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
    }


def _outputs(rows: int = 8) -> dict[str, torch.Tensor]:
    """Return a ``validation_step`` output dict carrying ``rows`` predictions.

    :param rows: Batch dimension of the predictions tensor.
    :returns: Output dict with the ``preds`` key the probe stages.
    """
    return {"preds": torch.ones(rows, 3) * 0.5}


def _probe(tmp_path: Path, **kwargs: Any) -> ValAudioProbe:
    r"""Build a ValAudioProbe rooted at ``tmp_path`` with a no-op probe_fn by default.

    :param tmp_path: Test directory the probe root is created under.
    :param \*\*kwargs: Overrides forwarded to ``ValAudioProbe`` (e.g. ``probe_fn``).
    :returns: Probe with ``num_samples=5``.
    """
    kwargs.setdefault("probe_fn", lambda probe_dir, step: {})
    return ValAudioProbe(probe_root=tmp_path / "val_audio_probe", num_samples=5, **kwargs)


def _run_validation(
    probe: ValAudioProbe,
    trainer: Trainer,
    module: _RecordingModule,
    *,
    rows: int = 8,
) -> None:
    """Drive one full validation epoch through the callback's hooks.

    :param probe: Callback under test.
    :param trainer: Trainer fake carrying step/rank/sanity state.
    :param module: Module fake recording harvested metrics.
    :param rows: Row count of the synthetic batch and predictions.
    """
    pl_module = cast("LightningModule", module)
    probe.on_validation_batch_end(trainer, pl_module, _outputs(rows), _batch(rows), 0)
    probe.on_validation_epoch_end(trainer, pl_module)


def test_val_audio_probe_stages_only_first_batch_up_to_num_samples(tmp_path: Path) -> None:
    """Only batch 0's leading num_samples rows are staged, and one probe launches.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    staged: list[Path] = []
    probe = _probe(tmp_path, probe_fn=lambda probe_dir, step: staged.append(probe_dir) or {})
    trainer = _trainer(global_step=5000)
    module = _RecordingModule()

    pl_module = cast("LightningModule", module)
    probe.on_validation_batch_end(trainer, pl_module, _outputs(), _batch(), 0)
    probe.on_validation_batch_end(trainer, pl_module, _outputs(), _batch(), 1)
    probe.on_validation_epoch_end(trainer, pl_module)
    _drain(probe)

    predictions = tmp_path / "val_audio_probe" / "step-5000" / "predictions"
    assert sorted(p.name for p in predictions.iterdir()) == [
        "pred-0.pt",
        "target-params-0.pt",
    ]
    assert torch.load(predictions / "pred-0.pt", weights_only=True).shape[0] == 5
    assert torch.load(predictions / "target-params-0.pt", weights_only=True).shape[0] == 5
    assert staged == [tmp_path / "val_audio_probe" / "step-5000"]


def test_val_audio_probe_stages_whole_batch_when_smaller_than_num_samples(tmp_path: Path) -> None:
    """A batch smaller than num_samples stages every row it has.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    probe = _probe(tmp_path)
    trainer = _trainer(global_step=10)

    _run_validation(probe, trainer, _RecordingModule(), rows=3)
    _drain(probe)

    pred = torch.load(
        tmp_path / "val_audio_probe" / "step-10" / "predictions" / "pred-0.pt",
        weights_only=True,
    )
    assert pred.shape[0] == 3


def test_val_audio_probe_harvests_metrics_from_finished_probe(tmp_path: Path) -> None:
    """A finished probe's metrics are logged at the next validation, tagged with its step.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    probe = _probe(tmp_path, probe_fn=lambda probe_dir, step: {"val_audio/mss_mean": 0.42})
    module = _RecordingModule()

    _run_validation(probe, _trainer(global_step=5000), module)
    assert module.logged == []  # nothing to harvest on the first validation

    _drain(probe)
    _run_validation(probe, _trainer(global_step=10_000), module)

    assert module.logged == [{"val_audio/mss_mean": 0.42, "val_audio/probe_step": 5000.0}]


def test_val_audio_probe_skips_launch_when_previous_probe_still_running(tmp_path: Path) -> None:
    """While a probe runs, newly staged epochs are dropped, not queued.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    launched: list[int] = []
    gate = threading.Event()

    def blocking_probe(probe_dir: Path, step: int) -> dict[str, float]:
        launched.append(step)
        assert gate.wait(timeout=_DRAIN_TIMEOUT_SECONDS), "gate was never released"
        return {}

    probe = _probe(tmp_path, probe_fn=blocking_probe)
    module = _RecordingModule()

    _run_validation(probe, _trainer(global_step=100), module)
    _run_validation(probe, _trainer(global_step=200), module)

    gate.set()
    _drain(probe)

    assert launched == [100]
    assert not (tmp_path / "val_audio_probe" / "step-200").exists()


def test_val_audio_probe_warns_and_continues_when_probe_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A probe exception is logged as a warning and never re-raised into the loop.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    :param caplog: Pytest fixture capturing the warning record.
    """

    def failing_probe(probe_dir: Path, step: int) -> dict[str, float]:
        raise RuntimeError("render exploded")

    probe = _probe(tmp_path, probe_fn=failing_probe)
    module = _RecordingModule()

    _run_validation(probe, _trainer(global_step=100), module)
    _drain(probe)
    with caplog.at_level("WARNING"):
        _run_validation(probe, _trainer(global_step=200), module)

    assert module.logged == []
    assert "render exploded" in caplog.text


def test_val_audio_probe_relaunches_after_a_failed_probe(tmp_path: Path) -> None:
    """A failed probe frees the slot, so the next epoch launches a fresh one.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    launched: list[int] = []

    def failing_probe(probe_dir: Path, step: int) -> dict[str, float]:
        launched.append(step)
        raise RuntimeError("render exploded")

    probe = _probe(tmp_path, probe_fn=failing_probe)
    module = _RecordingModule()

    _run_validation(probe, _trainer(global_step=100), module)
    _drain(probe)
    _run_validation(probe, _trainer(global_step=200), module)
    _drain(probe)

    assert launched == [100, 200]


def test_val_audio_probe_skips_during_sanity_checking(tmp_path: Path) -> None:
    """Sanity-check validation stages nothing.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    launched: list[int] = []
    probe = _probe(tmp_path, probe_fn=lambda probe_dir, step: launched.append(step) or {})
    trainer = _trainer(global_step=0)
    trainer.sanity_checking = True  # type: ignore[misc] — cast fake, attribute is real

    _run_validation(probe, trainer, _RecordingModule())
    _drain(probe)

    assert launched == []
    assert not (tmp_path / "val_audio_probe").exists()


def test_val_audio_probe_is_noop_on_non_zero_rank(tmp_path: Path) -> None:
    """Non-zero ranks stage nothing and launch nothing.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    launched: list[int] = []
    probe = _probe(tmp_path, probe_fn=lambda probe_dir, step: launched.append(step) or {})

    _run_validation(probe, _trainer(global_step=100, is_global_zero=False), _RecordingModule())
    _drain(probe)

    assert launched == []
    assert not (tmp_path / "val_audio_probe").exists()


def test_val_audio_probe_raises_when_outputs_lack_preds(tmp_path: Path) -> None:
    """Wiring the probe to a module without a preds key fails with a directed error.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    probe = _probe(tmp_path)

    with pytest.raises(ValueError, match="preds"):
        probe.on_validation_batch_end(
            _trainer(global_step=1),
            _module(),
            {"param_mse": torch.tensor(0.1)},
            _batch(),
            0,
        )


def test_val_audio_probe_prunes_probe_dir_after_successful_harvest(tmp_path: Path) -> None:
    """A harvested probe's local directory is deleted so disk use stays bounded.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    probe = _probe(tmp_path, probe_fn=lambda probe_dir, step: {"val_audio/mss_mean": 0.1})
    module = _RecordingModule()

    _run_validation(probe, _trainer(global_step=100), module)
    _drain(probe)
    assert (tmp_path / "val_audio_probe" / "step-100").exists()

    _run_validation(probe, _trainer(global_step=200), module)

    assert not (tmp_path / "val_audio_probe" / "step-100").exists()
    assert (tmp_path / "val_audio_probe" / "step-200").exists()


def test_val_audio_probe_keeps_probe_dir_after_failed_probe(tmp_path: Path) -> None:
    """A failed probe's directory is kept on disk for debugging.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """

    def failing_probe(probe_dir: Path, step: int) -> dict[str, float]:
        raise RuntimeError("render exploded")

    probe = _probe(tmp_path, probe_fn=failing_probe)
    module = _RecordingModule()

    _run_validation(probe, _trainer(global_step=100), module)
    _drain(probe)
    _run_validation(probe, _trainer(global_step=200), module)

    assert (tmp_path / "val_audio_probe" / "step-100").exists()


def _noop_probe_fn(probe_dir: Path, step: int) -> dict[str, float]:
    """Module-level probe_fn so the callback stays picklable, as production's partial is.

    :param probe_dir: Ignored staged probe directory.
    :param step: Ignored originating step.
    :returns: Empty metrics dict.
    """
    return {}


def test_val_audio_probe_is_picklable_after_launching_a_probe(tmp_path: Path) -> None:
    """ddp_spawn pickles callbacks; the live executor and future must not travel.

    :param tmp_path: Pytest fixture providing a fresh test directory.
    """
    import pickle

    probe = _probe(tmp_path, probe_fn=_noop_probe_fn)
    _run_validation(probe, _trainer(global_step=100), _RecordingModule())
    _drain(probe)

    # noqa rationale: same-process roundtrip of our own object — the ddp_spawn contract.
    restored = pickle.loads(pickle.dumps(probe))  # noqa: S301

    assert isinstance(restored, ValAudioProbe)
    assert restored.num_samples == probe.num_samples
    # The restored copy starts with a clean slot and can launch its own probe.
    _run_validation(restored, _trainer(global_step=200), _RecordingModule())
    _drain(restored)
    assert (tmp_path / "val_audio_probe" / "step-200").exists()
