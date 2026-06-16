"""Tests for the ``synth-setter-train`` CLI entrypoint.

Each test composes a Hydra ``cfg`` fixture and drives the in-process
``train(cfg)`` entrypoint (some chain ``evaluate``). Keep this module to
cfg-entrypoint tests; unit tests for helper functions belong in sibling
``test_*`` modules. ``tests/_meta/test_entrypoint_test_modules.py`` enforces
that no private ``synth_setter.cli`` helper is imported here.
"""

import os
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from synth_setter.cli.eval import evaluate
from synth_setter.cli.train import train
from synth_setter.data.vst import param_specs
from synth_setter.utils.utils import register_resolvers
from synth_setter.workspace import operator_workspace
from tests.conftest import (
    FAKE_VST_VARIANTS,
    NUM_FIXTURE_SAMPLES,
    REAL_VST_VARIANTS,
    _build_surge_xt_smoke_cfg,
    _SurgeSmokeVariant,
    build_fake_train_cfg,
)
from tests.evaluation._oracle_helpers import ORACLE_AUDIO_METRIC_BOUNDS
from tests.helpers.eval_fakes import (
    FAKE_AGGREGATED_METRICS_CSV,
    fake_metrics_csv,
    fake_postprocessing_subprocess,
)
from tests.helpers.run_if import RunIf
from tests.helpers.wandb_artifacts import publish_checkpoint_artifact

# Experiments cycled through the Surge XT VST smoke tests below. Single source of truth so
# the parametrize lists on the two ``test_train_*_surge_xt`` tests cannot drift apart.
_ORACLE_EXPERIMENT = "surge/fake_oracle"
_SURGE_SMOKE_EXPERIMENTS = (_ORACLE_EXPERIMENT, "surge/ffn_full")
_PREDICTION_PT_PREFIXES = ("pred", "target-audio", "target-params")
_FAKE_METRICS_CSV = fake_metrics_csv(NUM_FIXTURE_SAMPLES)

# TODO(#40): add @pytest.mark.ram gate for memory-intensive CPU tests test_train_fast_dev_run


def _smoke_eval_postprocessing_fake() -> Callable[[list[str]], None]:
    """Return fake eval postprocessing that materializes the fake-plugin smoke outputs.

    :returns: ``subprocess.run``-compatible callable.
    """
    return fake_postprocessing_subprocess(
        audio_metrics_csv=FAKE_AGGREGATED_METRICS_CSV,
        per_sample_metrics_csv=_FAKE_METRICS_CSV,
        render_sample_count=NUM_FIXTURE_SAMPLES,
    )


def test_train_fast_dev_run_tiny_model_tiny_data(cfg_train: DictConfig) -> None:
    """Run 1 train, val, and test step on CPU with `fast_dev_run`.

    Dataset/batch size constraints come from the shared `cfg_train` fixture
    (`batch_size=1`, `train_val_test_sizes=[2, 2, 2]`). This test only adds
    `fast_dev_run=True` to cap the loops at one batch each.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_fast_dev_run_gpu_compile(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU with torch.compile enabled.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.model.compile = True
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.check_val_every_n_epoch = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
        # Workaround for #709: ddp_spawn rank processes start with torch's
        # default `file_descriptor` sharing strategy, and their forked
        # dataloader workers inherit it. On the GitHub-hosted
        # `ubuntu-latest-4core` runner that strategy fails with
        # `RuntimeError: unable to resize file ... Invalid argument (22)`
        # because anonymous shm-backed fds can't be ftruncate'd in the
        # runner sandbox. Setting num_workers=0 keeps dataloading inline in
        # each rank process, sidestepping cross-process tensor shm entirely.
        # This test exercises ddp_spawn coordination, not dataloader
        # parallelism, so dropping workers does not weaken coverage.
        cfg_train.datamodule.num_workers = 0
    train(cfg_train)


@pytest.mark.gpu
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.accelerator = "gpu"
    HydraConfig().set_config(cfg_train)
    _, object_dict_1 = train(cfg_train)
    step_after_first = object_dict_1["trainer"].global_step
    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    _, object_dict_2 = train(cfg_train)
    step_after_resume = object_dict_2["trainer"].global_step

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    # The resume must actually train another epoch — `trainer.global_step` advancing
    # past the post-first-train value is the cheapest signal that the second `train()`
    # call did real work and didn't just load the checkpoint and exit. Replaces the
    # earlier `train/loss_epoch` decrease assertion, which broke when the metric_dict
    # keys changed.
    assert step_after_resume > step_after_first, (
        f"resume did not advance training: "
        f"global_step before={step_after_first}, after={step_after_resume}"
    )


@pytest.mark.parametrize("param_spec_name", ["surge_4", "surge_simple", "surge_xt"])
def test_cfg_surge_xt_global_wires_param_spec(param_spec_name: str) -> None:
    """Templated ``_build_surge_xt_smoke_cfg`` propagates the param spec for every supported spec.

    The propagation reaches both ``model.net.d_out`` and ``callbacks.log_per_param_mse.param_spec``
    — guards against the surge_4-only hardcodes the fixture used to carry.

    Calls the builder directly (not the ``cfg_surge_xt_global`` fixture) and pins
    ``accelerator="cpu"``: the cfg-shape contract is accelerator-independent and going
    through the fixture would drag in the parametrized ``accelerator`` hardware gate that
    hardfails on hosts without MPS/CUDA. ``experiment`` is pinned to the fixture default
    because param-spec propagation is itself experiment-independent — there's no need to
    cross-product with the experiment axis here.

    :param param_spec_name: Spec name driving the cfg builder.
    """
    cfg = _build_surge_xt_smoke_cfg(
        accelerator="cpu",
        param_spec_name=param_spec_name,
        experiment=_ORACLE_EXPERIMENT,
    )
    assert cfg.model.net.d_out == len(param_specs[param_spec_name])
    assert cfg.callbacks.log_per_param_mse.param_spec == param_spec_name


@pytest.mark.slow
def test_train_fake_mode_nondefault_spec_sizes_batches_from_registry(tmp_path: Path) -> None:
    """Fake-mode train through the entrypoint sizes batches from a non-default ``param_spec_name``.

    Drives the real ``train(cfg)`` entrypoint with ``datamodule.fake=true`` and the
    non-default ``surge_simple`` spec: no dataset on disk, so the run exercises the
    registry-derived fake width end-to-end. The width-agnostic ``surge/fake_oracle``
    experiment (oracle returns ``batch["params"]``) tolerates the registry-width batches,
    and the datamodule the entrypoint built carries that registry-derived width.

    :param tmp_path: Pinned as Hydra ``output_dir`` / ``log_dir``; no dataset is read.
    """
    expected_width = len(param_specs["surge_simple"])
    cfg = build_fake_train_cfg(tmp_path, param_spec_name="surge_simple")

    HydraConfig().set_config(cfg)
    _, object_dict = train(cfg)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"

    datamodule = object_dict["datamodule"]
    assert datamodule.train_dataset.num_params == expected_width
    sample = datamodule.train_dataset[0]
    assert sample["params"].shape == (2, expected_width)


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", REAL_VST_VARIANTS, indirect=True)
def test_train_surge_xt(cfg_surge_real_train: DictConfig, experiment_name: str) -> None:
    """Run training of the Surge XT model on the smoke test fixture, across both experiments and dataset formats.

    Asserts the trainer advanced and produced a finite ``train/loss`` — catches silent
    no-op trainers and NaN/Inf regressions that a bare ``train()`` call would not. The
    ``surge/fake_oracle`` leg additionally pins ``train/loss`` to exactly zero (the
    oracle constructs its loss as ``0.0 * net(mel_spec).sum()`` — any drift means the
    oracle stopped being an oracle); meaningful loss-progression coverage comes from
    the ``surge/ffn_full`` leg. Parametrized over h5 and Lance so both datamodules train
    through the real Surge XT render.

    :param cfg_surge_real_train: Surge XT training config (parametrized over experiment and
        dataset format).
    :param experiment_name: Hydra experiment override the cfg was built from — drives
        the oracle-specific tight bound below.
    """
    HydraConfig().set_config(cfg_surge_real_train)
    metric_dict, object_dict = train(cfg_surge_real_train)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"

    # `surge_ff_module` logs `train/loss` with `on_step=True, on_epoch=True`, which
    # populates `train/loss_step` (and `train/loss_epoch` if an epoch boundary was
    # crossed) in `trainer.callback_metrics`. With `TRAINING_STEPS=1` only the
    # step-level key is guaranteed; assert whichever is present is finite.
    loss_keys = [k for k in metric_dict if k.startswith("train/loss")]
    assert loss_keys, f"no train/loss* key in metric_dict: {sorted(metric_dict)}"
    for key in loss_keys:
        loss = metric_dict[key]
        assert torch.isfinite(loss).all(), f"{key} is not finite: {loss}"

    if experiment_name == _ORACLE_EXPERIMENT:
        for key in loss_keys:
            loss_value = metric_dict[key].item()
            assert loss_value == 0.0, f"oracle {key} not exactly zero: {loss_value}"


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", REAL_VST_VARIANTS, indirect=True)
def test_train_eval_surge_xt(
    tmp_path: Path,
    cfg_surge_real_train: DictConfig,
    cfg_surge_real_eval: DictConfig,
    param_spec_name: str,
    experiment_name: str,
) -> None:
    """End-to-end smoke test: train Surge XT briefly on a small fixture dataset, then run standalone eval on the saved checkpoint, for both dataset formats.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_real_train: Surge XT smoke-test training config (h5 or Lance arm).
    :param cfg_surge_real_eval: Matching smoke-test eval config (ckpt_path set by this test).
    :param param_spec_name: Param spec the fixtures (and therefore the trained model) are
        wired for — passed to ``predict_vst_audio.py`` so the script's decode layout matches
        the predicted tensor's encoding (mismatched specs go off-the-end and crash with
        ``can only convert an array of size 1 to a Python scalar``).
    :param experiment_name: Hydra experiment override the cfg was built from — drives
        the oracle-specific tight audio-metric bounds at the end of the test.
    """
    from pedalboard.io import AudioFile

    NUM_AUDIO_METRICS = 4  # mss, wmfcc, sot, rms
    METRICS_FILE_EXPECTATIONS = {
        "aggregated_metrics.csv": {
            "rows": NUM_AUDIO_METRICS,
            "columns": {"mean", "std"},
        },
        "metrics.csv": {
            "rows": NUM_FIXTURE_SAMPLES,
            "columns": {"mss", "wmfcc", "sot", "rms"},
        },
    }

    HydraConfig().set_config(cfg_surge_real_train)
    train(cfg_surge_real_train)

    # `cfg_surge_real_eval.ckpt_path` is pre-pointed at this same `tmp_path` by the
    # fixture; assert the train step actually produced the file before eval reads it.
    assert Path(cfg_surge_real_eval.ckpt_path).exists()

    HydraConfig().set_config(cfg_surge_real_eval)
    evaluate(cfg_surge_real_eval)

    # `PredictionWriter` (in `src/synth_setter/utils/callbacks.py`) with `write_interval=batch` saves three
    # tensors per predict batch: `pred-{i}.pt`, `target-audio-{i}.pt`, `target-params-{i}.pt`.
    predictions_dir = tmp_path / "predictions"
    assert predictions_dir.is_dir()
    expected_names = sorted(
        f"{prefix}-{i}.pt"
        for prefix in ("pred", "target-audio", "target-params")
        for i in range(NUM_FIXTURE_SAMPLES)
    )
    assert sorted(p.name for p in predictions_dir.iterdir()) == expected_names

    for i in range(NUM_FIXTURE_SAMPLES):
        pred = torch.load(predictions_dir / f"pred-{i}.pt", weights_only=True)
        assert torch.isfinite(pred).all(), f"pred-{i}.pt contains NaN/Inf"

        # The oracle's ``predict_step`` returns ``batch["params"]`` verbatim, so the
        # saved prediction tensor must be bit-identical to the saved target params.
        # This is the strongest oracle invariant — pinning it here isolates regressions
        # in ``predict_step`` from the noisier downstream audio metrics (which absorb
        # Surge XT's per-voice render jitter and would mask a small deviation).
        if experiment_name == _ORACLE_EXPERIMENT:
            target_params = torch.load(
                predictions_dir / f"target-params-{i}.pt", weights_only=True
            )
            assert torch.equal(pred, target_params), f"oracle pred-{i}.pt != target-params-{i}.pt"

    audio_dir = tmp_path / "audio"
    sample_dirs = sorted(d for d in audio_dir.iterdir() if d.is_dir())
    assert [d.name for d in sample_dirs] == [f"sample_{i}" for i in range(NUM_FIXTURE_SAMPLES)]
    # ``target.wav`` is rendered from fixture-truth params and must be audible —
    # silence there would be a real bug. ``pred.wav`` from a 1-step-trained model
    # can legitimately land in a silent region of Surge XT's param space (MPS
    # non-determinism); ``compute_rms`` clamps its denominator so silent pred
    # yields ``cosine_sim = 0`` rather than NaN, and the finite-metric assertion
    # at the end of this test is the real end check.
    for sample_dir in sample_dirs:
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "spec.png").is_file()
        assert (sample_dir / "params.csv").is_file()

        with AudioFile(str(sample_dir / "target.wav")) as f:
            target_audio = f.read(f.frames)
        target_peak = float(np.abs(target_audio).max())
        assert target_peak > 1e-6, (
            f"{sample_dir.name}/target.wav is silent (peak={target_peak:.2e})"
        )

    metrics_dir = tmp_path / "metrics"
    for metrics_file, expected in METRICS_FILE_EXPECTATIONS.items():
        assert (metrics_dir / metrics_file).is_file(), f"{metrics_file} not found in {metrics_dir}"
        metrics_df = pd.read_csv(metrics_dir / metrics_file)
        assert len(metrics_df) == expected["rows"]
        assert expected["columns"].issubset(metrics_df.columns)
        numeric = metrics_df[sorted(expected["columns"])].to_numpy()
        assert np.isfinite(numeric).all(), f"{metrics_file} contains NaN/Inf:\n{metrics_df}"

    if experiment_name == _ORACLE_EXPERIMENT:
        # Surge XT injects per-voice render jitter (oscillator phase, noise seed)
        # even with bit-identical params, so the audio metrics don't collapse to
        # zero — bounds absorb that jitter while still failing on a real regression.
        per_sample = pd.read_csv(metrics_dir / "metrics.csv")
        bounds = ORACLE_AUDIO_METRIC_BOUNDS
        assert per_sample["mss"].max() < bounds.mss_max, (
            f"oracle mss too high: {per_sample['mss'].tolist()}"
        )
        assert per_sample["wmfcc"].max() < bounds.wmfcc_max, (
            f"oracle wmfcc too high: {per_sample['wmfcc'].tolist()}"
        )
        assert per_sample["sot"].max() < bounds.sot_max, (
            f"oracle sot too high: {per_sample['sot'].tolist()}"
        )
        assert per_sample["rms"].min() > bounds.rms_min, (
            f"oracle rms too low: {per_sample['rms'].tolist()}"
        )


@pytest.mark.requires_vst
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("WANDB_API_KEY"),
    reason="real W&B round-trip needs WANDB_API_KEY (injected on trusted CI only)",
)
@pytest.mark.parametrize("experiment_name", ["surge/ffn_full"], indirect=True)
def test_train_resumes_from_wandb_resolved_checkpoint(
    tmp_path: Path,
    cfg_surge_xt: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
) -> None:
    """Training resumes from a ``ckpt_path`` pinned as ``${wandb:...}``, downloaded from the live registry.

    Exercises the exact train-side seam the wandb_checkpoint split protects: ``train.py`` reads
    ``cfg.get("ckpt_path")`` into ``trainer.fit(ckpt_path=...)``. A first one-step run produces a
    real Lightning checkpoint, published to ``tinaudio/synth-setter-citest``; a second run pins
    that artifact via ``${wandb:...}`` and must download + resume it, advancing ``global_step``.
    Proves the resolver works through the real W&B API on the train entrypoint, not just a fake.

    :param tmp_path: Shared output dir; also the workspace root the resolver caches under.
    :param cfg_surge_xt: Surge XT smoke-test training config (one step on the fixture dataset).
    :param monkeypatch: Pins ``SYNTH_SETTER_WORKSPACE`` so the download cache stays under tmp_path.
    :param experiment_name: Pinned to ``surge/ffn_full`` — the artifact id need only round-trip.
    """
    HydraConfig().set_config(cfg_surge_xt)
    _, first = train(cfg_surge_xt)
    step_after_first = first["trainer"].global_step
    ckpt = tmp_path / "checkpoints" / "last.ckpt"
    assert ckpt.is_file(), "first train step did not write last.ckpt"

    # Body runs inside the ``with`` so the resolver downloads before the artifact/run teardown.
    with publish_checkpoint_artifact(
        ckpt, "model-citest-ffn_full-resume", tmp_path / "wandb"
    ) as ref:
        # Contain the resolver's download cache under tmp_path so each run fetches fresh (a warm
        # self-hosted runner must not reuse a stale cached ckpt for the same :latest ref).
        monkeypatch.setenv("SYNTH_SETTER_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        operator_workspace.cache_clear()
        register_resolvers()
        with open_dict(cfg_surge_xt):
            cfg_surge_xt.ckpt_path = "${wandb:" + ref + "}"
            cfg_surge_xt.trainer.max_steps = step_after_first + 1

        HydraConfig().set_config(cfg_surge_xt)
        _, second = train(cfg_surge_xt)
        step_after_resume = second["trainer"].global_step

    assert (tmp_path / ".cache" / "checkpoints").is_dir(), "resolver did not download the artifact"
    assert step_after_resume > step_after_first, (
        f"resume did not advance training: before={step_after_first}, after={step_after_resume}"
    )


def test_train_fast_dev_run_lance_datamodule(cfg_train_lance: DictConfig) -> None:
    """Run 1 train, val, and test step on CPU reading batches from Lance shards.

    Exercises config wiring, ``LanceVSTDataModule`` setup, and real Lance batch
    reads end-to-end through the in-process ``train(cfg)`` entrypoint; the Hydra
    composition path lives on the ``cfg_train_lance`` fixture. Also pins the
    Dataset-API migration's two e2e-visible contracts on the live datamodule:
    splits open as directory datasets, and a column accepts unsorted fancy
    indices returning rows in the requested order.

    :param cfg_train_lance: Composed ``datamodule=surge_lance`` training config.
    """
    HydraConfig().set_config(cfg_train_lance)
    _, object_dict = train(cfg_train_lance)

    # Pin the Dataset-API migration e2e: the split the datamodule trained over
    # is a Lance dataset directory, not the legacy single ``.lance`` file.
    train_split = Path(object_dict["datamodule"].dataset_root) / "train.lance"
    assert train_split.is_dir()


@pytest.mark.fake_vst
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", FAKE_VST_VARIANTS, indirect=True)
def test_train_surge_fake(
    cfg_surge_fake_train: DictConfig,
    surge_smoke_variant: _SurgeSmokeVariant,
    experiment_name: str,
) -> None:
    """Run the Surge smoke training matrix over fake-plugin h5 and Lance splits.

    :param cfg_surge_fake_train: CPU training config for the dataset-format arm under test.
    :param surge_smoke_variant: Dataset-format arm (h5 or Lance) the cfg was built from.
    :param experiment_name: Hydra experiment override the cfg was built from.
    """
    HydraConfig().set_config(cfg_surge_fake_train)
    metric_dict, object_dict = train(cfg_surge_fake_train)

    trainer = object_dict["trainer"]
    assert trainer.global_step >= 1, f"trainer did not advance: global_step={trainer.global_step}"

    train_split = (
        Path(object_dict["datamodule"].dataset_root) / f"train{surge_smoke_variant.split_ext}"
    )
    assert train_split.exists()

    loss_keys = [key for key in metric_dict if key.startswith("train/loss")]
    assert loss_keys, f"no train/loss* key in metric_dict: {sorted(metric_dict)}"
    for key in loss_keys:
        loss = metric_dict[key]
        assert torch.isfinite(loss).all(), f"{key} is not finite: {loss}"

    if experiment_name == _ORACLE_EXPERIMENT:
        for key in loss_keys:
            loss_value = metric_dict[key].item()
            assert loss_value == 0.0, f"oracle {key} not exactly zero: {loss_value}"


@pytest.mark.fake_vst
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", FAKE_VST_VARIANTS, indirect=True)
def test_train_eval_surge_fake(
    tmp_path: Path,
    cfg_surge_fake_train: DictConfig,
    cfg_surge_fake_eval: DictConfig,
    surge_smoke_variant: _SurgeSmokeVariant,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
) -> None:
    """Train on a fake-plugin arm, then verify prediction tensors from checkpoint eval.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_fake_train: CPU training config for the dataset-format arm under test.
    :param cfg_surge_fake_eval: Matching eval config pinned to ``last.ckpt``.
    :param surge_smoke_variant: Dataset-format arm (h5 or Lance) under test.
    :param monkeypatch: Stubs render/metrics subprocesses so no real VST host launches.
    :param experiment_name: Hydra experiment override the cfg was built from.
    """
    metric_dict, object_dict = _evaluate_surge_fake_checkpoint(
        cfg_surge_fake_train, cfg_surge_fake_eval, monkeypatch
    )

    _assert_surge_fake_eval_basics(metric_dict, object_dict, surge_smoke_variant)

    predictions_dir = tmp_path / "predictions"
    assert predictions_dir.is_dir()
    assert sorted(path.name for path in predictions_dir.iterdir()) == _prediction_file_names()

    for sample_idx in range(NUM_FIXTURE_SAMPLES):
        pred = torch.load(predictions_dir / f"pred-{sample_idx}.pt", weights_only=True)
        assert torch.isfinite(pred).all(), f"pred-{sample_idx}.pt contains NaN/Inf"

        if experiment_name == _ORACLE_EXPERIMENT:
            target_params = torch.load(
                predictions_dir / f"target-params-{sample_idx}.pt", weights_only=True
            )
            assert torch.equal(pred, target_params), (
                f"oracle pred-{sample_idx}.pt != target-params-{sample_idx}.pt"
            )


@pytest.mark.fake_vst
@pytest.mark.parametrize("experiment_name", _SURGE_SMOKE_EXPERIMENTS, indirect=True)
@pytest.mark.parametrize("surge_smoke_variant", FAKE_VST_VARIANTS, indirect=True)
def test_train_eval_surge_fake_writes_audio_and_metrics_outputs(
    tmp_path: Path,
    cfg_surge_fake_train: DictConfig,
    cfg_surge_fake_eval: DictConfig,
    surge_smoke_variant: _SurgeSmokeVariant,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
) -> None:
    """Train on a fake-plugin arm, then verify fake render and metrics outputs.

    :param tmp_path: The temporary logging path.
    :param cfg_surge_fake_train: CPU training config for the dataset-format arm under test.
    :param cfg_surge_fake_eval: Matching eval config pinned to ``last.ckpt``.
    :param surge_smoke_variant: Dataset-format arm (h5 or Lance) under test.
    :param monkeypatch: Stubs render/metrics subprocesses so no real VST host launches.
    :param experiment_name: Hydra experiment override; parametrizes the train/eval run.
    """
    metric_dict, object_dict = _evaluate_surge_fake_checkpoint(
        cfg_surge_fake_train, cfg_surge_fake_eval, monkeypatch
    )

    _assert_surge_fake_eval_basics(metric_dict, object_dict, surge_smoke_variant)

    audio_dir = tmp_path / "audio"
    sample_dirs = sorted(path for path in audio_dir.iterdir() if path.is_dir())
    assert [path.name for path in sample_dirs] == [
        f"sample_{sample_idx}" for sample_idx in range(NUM_FIXTURE_SAMPLES)
    ]
    for sample_dir in sample_dirs:
        assert (sample_dir / "target.wav").is_file()
        assert (sample_dir / "pred.wav").is_file()
        assert (sample_dir / "spec.png").is_file()
        assert (sample_dir / "params.csv").is_file()

    metrics_dir = tmp_path / "metrics"
    for metrics_file, expected_rows in {
        "aggregated_metrics.csv": 4,
        "metrics.csv": NUM_FIXTURE_SAMPLES,
    }.items():
        assert (metrics_dir / metrics_file).is_file(), f"{metrics_file} not found"
        metrics_df = pd.read_csv(metrics_dir / metrics_file)
        assert len(metrics_df) == expected_rows
        numeric = metrics_df.select_dtypes(include=[np.number]).to_numpy()
        assert np.isfinite(numeric).all(), f"{metrics_file} contains NaN/Inf:\n{metrics_df}"


def _evaluate_surge_fake_checkpoint(
    cfg_train: DictConfig,
    cfg_eval: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the fake-plugin train-to-predict smoke path with faked postprocessing.

    :param cfg_train: CPU training config for the dataset-format arm under test.
    :param cfg_eval: Matching eval config pinned to ``last.ckpt``.
    :param monkeypatch: Stubs postprocessing subprocess dependencies.
    :returns: ``evaluate`` metric and object dictionaries.
    """
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    assert Path(cfg_eval.ckpt_path).exists()

    monkeypatch.setattr("synth_setter.cli.eval.subprocess.run", _smoke_eval_postprocessing_fake())
    monkeypatch.setattr("synth_setter.cli.eval.vst_headless_wrapper", lambda: object())
    monkeypatch.setattr(
        "synth_setter.cli.eval.as_file",
        lambda _traversable: nullcontext(Path("/fake/headless-wrapper")),
    )

    HydraConfig().set_config(cfg_eval)
    return evaluate(cfg_eval)


def _assert_surge_fake_eval_basics(
    metric_dict: dict[str, object],
    object_dict: dict[str, object],
    variant: _SurgeSmokeVariant,
) -> None:
    """Assert the shared fake-plugin predict-mode eval invariants.

    :param metric_dict: Metrics returned by ``evaluate``.
    :param object_dict: Objects returned by ``evaluate``.
    :param variant: Dataset-format arm selecting the predicted split's suffix.
    """
    assert metric_dict["audio/mss_mean"] == pytest.approx(0.5)
    assert metric_dict["audio/rms_std"] == pytest.approx(0.01)

    dataset_root = getattr(object_dict["datamodule"], "dataset_root")
    assert isinstance(dataset_root, str | os.PathLike)

    test_split = Path(dataset_root) / f"test{variant.split_ext}"
    assert test_split.exists()


def _prediction_file_names() -> list[str]:
    """Return the per-batch files ``PredictionWriter`` writes for the smoke split.

    :returns: Sorted expected prediction filenames.
    """
    return sorted(
        f"{prefix}-{sample_idx}.pt"
        for prefix in _PREDICTION_PT_PREFIXES
        for sample_idx in range(NUM_FIXTURE_SAMPLES)
    )
