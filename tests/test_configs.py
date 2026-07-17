"""Tests that Hydra config groups compose without errors."""

from collections.abc import Sequence
from typing import Any

import hydra
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationToMissingValueError

from synth_setter.resources import configs_dir
from tests.conftest import _build_surge_xt_smoke_cfg


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.datamodule
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.datamodule)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.datamodule
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.datamodule)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_cfg_train_trainer_keys_coherent_with_test_mode(cfg_train: DictConfig) -> None:
    """Guard: ``cfg_train`` fixture produces a coherent epoch-based trainer config.

    Regression guard for #625: the original mismatch was that
    ``configs/trainer/default.yaml`` shipped step-based keys (``min_steps``,
    ``max_steps``) which the fixture never unset, silently suppressing
    validation under ``limit_train_batches=0.01`` (#47, #619, #620, #624).

    The fix on this branch removes those keys from ``trainer/default.yaml``
    entirely and pins dataset shape via ``train_val_test_sizes`` instead of
    fractional ``limit_*_batches``. The guard now asserts the structural
    invariant: step-based keys must not be present in the composed trainer.
    """
    assert cfg_train.trainer.max_epochs == 1
    assert cfg_train.trainer.check_val_every_n_epoch == 1
    assert cfg_train.trainer.val_check_interval == 1
    assert "min_steps" not in cfg_train.trainer
    assert "max_steps" not in cfg_train.trainer


class TestWandbConfigResolvesFromEnv:
    """Verify wandb entity/project resolve from env vars (#265)."""

    def test_wandb_entity_resolves_from_env(self, monkeypatch):
        """OmegaConf resolves WANDB_ENTITY from environment."""
        monkeypatch.setenv("WANDB_ENTITY", "test-entity")
        cfg = OmegaConf.load(str(configs_dir() / "logger" / "wandb.yaml"))
        assert OmegaConf.select(cfg, "wandb.entity") == "test-entity"

    def test_wandb_project_resolves_from_env(self, monkeypatch):
        """OmegaConf resolves WANDB_PROJECT from environment."""
        monkeypatch.setenv("WANDB_PROJECT", "test-project")
        cfg = OmegaConf.load(str(configs_dir() / "logger" / "wandb.yaml"))
        assert OmegaConf.select(cfg, "wandb.project") == "test-project"

    def test_wandb_entity_defaults_to_none_when_env_unset(self, monkeypatch):
        """Entity falls back to None (user's default W&B entity) when env var unset."""
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        cfg = OmegaConf.load(str(configs_dir() / "logger" / "wandb.yaml"))
        assert OmegaConf.select(cfg, "wandb.entity") is None

    def test_wandb_project_defaults_to_synth_setter_when_env_unset(self, monkeypatch):
        """Project falls back to synth-setter when env var unset."""
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        cfg = OmegaConf.load(str(configs_dir() / "logger" / "wandb.yaml"))
        assert OmegaConf.select(cfg, "wandb.project") == "synth-setter"


# Volatile cfg branches that legitimately differ between the fixture builder and the YAML
# compose: ``paths`` is filesystem-anchored via ``operator_workspace()``, ``hydra`` carries
# the per-invocation runtime metadata, ``task_name`` is interpolated from the entry-point
# script name (test runner vs. ``train.py``), ``render`` is an eval-only group (null here,
# selected per-run in predict mode), and ``ckpt_path`` is an eval/deploy concern — the
# real surge experiment pins a ${wandb:...} ref while its test-mps smoke sibling trains
# from scratch (null) — none of which is part of the train cfg shape this contract pins.
_VOLATILE_TOP_KEYS = ("paths", "hydra", "task_name", "render", "ckpt_path")


def _strip_volatile(cfg_dict: dict[Any, Any]) -> dict[Any, Any]:
    """Drop top-level keys whose values legitimately differ across compose contexts."""
    return {k: v for k, v in cfg_dict.items() if k not in _VOLATILE_TOP_KEYS}


def _diff_dicts(a: dict[Any, Any], b: dict[Any, Any], prefix: str = "") -> list[str]:
    """Recursively diff two dicts and return human-readable difference lines."""
    diffs: list[str] = []
    for key in sorted(set(a) | set(b)):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in a:
            diffs.append(f"  + {path} (only in yaml): {b[key]!r}")
            continue
        if key not in b:
            diffs.append(f"  - {path} (only in fixture): {a[key]!r}")
            continue
        va, vb = a[key], b[key]
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs.extend(_diff_dicts(va, vb, path))
        elif va != vb:
            diffs.append(f"  ~ {path}: fixture={va!r}  yaml={vb!r}")
    return diffs


@pytest.mark.parametrize(
    ("experiment", "test_mps_yaml"),
    [
        ("surge/fake_oracle", "surge/test-mps-fake-oracle"),
        ("surge/ffn_full", "surge/test-mps-ffn"),
    ],
    ids=["fake_oracle", "ffn_full"],
)
def test_test_mps_yaml_matches_cfg_surge_xt_global(experiment: str, test_mps_yaml: str) -> None:
    """Each ``surge/test-mps-*.yaml`` matches the smoke fixture's MPS cfg for its experiment.

    Guard against silent drift: the fixture's open_dict bake-ins and each YAML's
    defaults list / overrides must stay in lockstep, otherwise a test that uses one
    and an ``experiment=surge/test-mps-*`` invocation that uses the other will produce
    different runs without anyone noticing. Builds both configs in-process (no MPS
    hardware needed — only the cfg shape is compared, not runtime behavior).

    :param experiment: Hydra ``experiment=...`` override the fixture is built against
        (``"surge/fake_oracle"`` or ``"surge/ffn_full"``).
    :param test_mps_yaml: Sibling smoke YAML the fixture is compared against
        (``"surge/test-mps-fake-oracle"`` or ``"surge/test-mps-ffn"``).
    """
    fixture_cfg = _build_surge_xt_smoke_cfg(
        accelerator="mps", param_spec_name="surge_4", experiment=experiment
    )
    fixture_d_out = fixture_cfg.model.net.d_out
    fixture_param_spec = fixture_cfg.callbacks.log_per_param_mse.param_spec
    GlobalHydra.instance().clear()

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        experiment_cfg = compose(
            config_name="train.yaml",
            return_hydra_config=False,
            overrides=[
                f"experiment={test_mps_yaml}",
                f"model.net.d_out={fixture_d_out}",
                f"callbacks.log_per_param_mse.param_spec={fixture_param_spec}",
            ],
        )
    GlobalHydra.instance().clear()

    # ``resolve=False`` keeps interpolation strings (``${paths.output_dir}``,
    # ``${hydra:...}``) verbatim on both sides so the comparison doesn't require an
    # active HydraConfig and doesn't trip on runtime-only resolvers.
    fixture_dict = _strip_volatile(OmegaConf.to_container(fixture_cfg, resolve=False))  # type: ignore[arg-type]
    experiment_dict = _strip_volatile(OmegaConf.to_container(experiment_cfg, resolve=False))  # type: ignore[arg-type]

    diffs = _diff_dicts(fixture_dict, experiment_dict)
    assert not diffs, (
        f"{test_mps_yaml}.yaml drifted from "
        f"cfg_surge_xt_global(mps, surge_4, {experiment!r}):\n" + "\n".join(diffs)
    )


def _compose(config_name: str, overrides: Sequence[str]) -> DictConfig:
    """Compose a top-level config with overrides, clearing GlobalHydra around it.

    :param config_name: Top-level config to compose (``train.yaml``, ``eval.yaml``, ...).
    :param overrides: Hydra CLI-style overrides.
    :returns: The composed config.
    """
    GlobalHydra.instance().clear()
    try:
        with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
            return compose(
                config_name=config_name, return_hydra_config=False, overrides=list(overrides)
            )
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    "model_name",
    ["vst_fake_oracle", "vst_ffn", "vst_flow", "vst_flowmlp", "vst_flowvae"],
)
def test_vst_model_group_composes(model_name: str) -> None:
    """Each synth-neutral VST model group composes successfully.

    :param model_name: Hydra model group selected for the composition.
    """
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_simple",
            f"model={model_name}",
            "trainer=cpu",
        ],
    )

    assert cfg.model._target_.startswith("synth_setter.models.vst_")


@pytest.mark.parametrize(
    ("experiment", "param_spec", "latent_dim"),
    [("vae_simple", "surge_simple", 92), ("vae_full", "surge_xt", 300)],
)
def test_vst_flowvae_experiment_couples_spec_and_output_width(
    experiment: str, param_spec: str, latent_dim: int
) -> None:
    """Concrete Flow-VAE experiments pair each ParamSpec with its encoded width.

    :param experiment: Surge experiment basename.
    :param param_spec: Expected concrete ParamSpec.
    :param latent_dim: Expected network output width.
    """
    cfg = _compose("train.yaml", [f"experiment=surge/{experiment}", "trainer=cpu"])

    assert cfg.model.param_spec == param_spec
    assert cfg.model.net.latent_dim == latent_dim


@pytest.mark.parametrize(
    (
        "legacy_name",
        "target_suffix",
        "expected_path",
        "expected_value",
        "expected_param_spec",
        "expected_compile",
        "expected_learning_rate",
    ),
    [
        (
            "surge_fake_oracle",
            "vst_fake_oracle_module.VSTFakeOracleModule",
            "net.d_out",
            92,
            None,
            False,
            1e-4,
        ),
        ("surge_ffn", "vst_ff_module.VSTFeedForwardModule", "net.d_out", 92, None, True, 1e-4),
        (
            "surge_flow",
            "vst_flow_matching_module.VSTFlowMatchingModule",
            "num_params",
            90,
            None,
            True,
            1e-4,
        ),
        (
            "surge_flowmlp",
            "vst_flow_matching_module.VSTFlowMatchingModule",
            "num_params",
            90,
            None,
            True,
            1e-4,
        ),
        (
            "surge_flowvae",
            "vst_flowvae_module.VSTFlowVAEModule",
            "net.latent_dim",
            92,
            "surge_xt",
            True,
            2e-4,
        ),
    ],
)
def test_legacy_surge_model_group_preserves_concrete_defaults(
    legacy_name: str,
    target_suffix: str,
    expected_path: str,
    expected_value: int,
    expected_param_spec: str | None,
    expected_compile: bool,
    expected_learning_rate: float,
) -> None:
    """Each archived model selection preserves its concrete defaults.

    :param legacy_name: Historical Hydra model-group name.
    :param target_suffix: Canonical VST target expected after composition.
    :param expected_path: Config field preserving the historical concrete default.
    :param expected_value: Historical concrete default.
    :param expected_param_spec: Historical ParamSpec default, when applicable.
    :param expected_compile: Historical torch.compile default.
    :param expected_learning_rate: Historical optimizer learning-rate default.
    """
    legacy_cfg = _compose(
        "train.yaml",
        ["datamodule=surge_simple", f"model={legacy_name}", "trainer=cpu"],
    )
    assert legacy_cfg.model._target_.endswith(target_suffix)
    assert OmegaConf.select(legacy_cfg.model, expected_path) == expected_value
    assert legacy_cfg.model.compile is expected_compile
    assert legacy_cfg.model.optimizer.lr == expected_learning_rate
    assert OmegaConf.select(legacy_cfg.model, "param_spec") == expected_param_spec


@pytest.mark.parametrize(
    ("callbacks_name", "expected_callback"),
    [("default_vst", "model_checkpoint"), ("eval_vst", "prediction_writer")],
)
def test_vst_callback_group_composes(callbacks_name: str, expected_callback: str) -> None:
    """Each synth-neutral VST callback group composes successfully.

    :param callbacks_name: Hydra callback group selected for the composition.
    :param expected_callback: Callback key expected in the composed group.
    """
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_simple",
            "model=vst_ffn",
            f"callbacks={callbacks_name}",
            "trainer=cpu",
        ],
    )

    assert expected_callback in cfg.callbacks


@pytest.mark.parametrize(
    ("callbacks_name", "expected_callback"),
    [("default_surge", "model_checkpoint"), ("eval_surge", "prediction_writer")],
)
def test_legacy_surge_callback_alias_composes_vst_callbacks(
    callbacks_name: str, expected_callback: str
) -> None:
    """Historical callback selections resolve canonical VST callbacks.

    :param callbacks_name: Historical Hydra callback-group name.
    :param expected_callback: Canonical callback expected after composition.
    """
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_simple",
            "model=vst_ffn",
            f"callbacks={callbacks_name}",
            "trainer=cpu",
        ],
    )

    assert expected_callback in cfg.callbacks


def test_log_per_param_mse_config_uses_active_datamodule_spec() -> None:
    """The VST per-parameter callback resolves the active datamodule spec."""
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_mini",
            "model=ffn",
            "callbacks=log_per_param_mse",
            "trainer=cpu",
        ],
    )

    assert cfg.callbacks.log_per_param_mse.param_spec == "surge_4"


def test_log_per_param_mse_config_requires_datamodule_spec() -> None:
    """The VST per-parameter callback rejects an unset ParamSpec."""
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=vst",
            "model=ffn",
            "callbacks=log_per_param_mse",
            "trainer=cpu",
        ],
    )

    with pytest.raises(InterpolationToMissingValueError, match="param_spec_name"):
        OmegaConf.to_container(cfg.callbacks, resolve=True, throw_on_missing=True)


def test_surge_training_defaults_enable_bounded_validation_and_auto_probe() -> None:
    """The surge family validates a bounded sample and enables the probe when usable."""
    cfg = _compose("train.yaml", ["experiment=surge/flow_simple"])

    assert cfg.trainer.limit_val_batches == 20
    assert cfg.training.val_audio_probe == "auto"


def test_surge_4_generate_dataset_experiment_composes_with_inline_finalize() -> None:
    """``generate_dataset/surge-4-lance-440k-20k-20k`` wires surge_4 and inline finalize.

    Pins the full-scale surge_4 Lance pipeline contract: the surge_4 render and
    datamodule groups, Lance output, the 440k/20k/20k split, and the inline
    finalize that writes ``dataset.complete`` in the same CLI process.
    """
    cfg = _compose("dataset.yaml", ["experiment=generate_dataset/surge-4-lance-440k-20k-20k"])

    assert cfg.render.param_spec_name == "surge_4"
    assert cfg.render.plugin_state_path == "presets/surge-mini.vstpreset"
    assert cfg.datamodule.param_spec_name == "surge_4"
    assert cfg.output_format == "lance"
    assert list(cfg.train_val_test_sizes) == [440000, 20000, 20000]
    assert cfg.finalize_inline is True


def test_surge_4_train_experiment_composes_with_surge_4_width() -> None:
    """``surge/ffn_4`` trains the FFN at the surge_4 encoded width on Lance data.

    Pins the surge_4 train contract: Lance datamodule keyed to the surge_4 spec,
    ``d_out`` equal to the spec's encoded width (7), and per-param MSE logging
    labeled with the surge_4 spec.
    """
    cfg = _compose("train.yaml", ["experiment=surge/ffn_4"])

    assert cfg.datamodule.param_spec_name == "surge_4"
    assert cfg.datamodule._target_ == "synth_setter.data.lance_datamodule.LanceVSTDataModule"
    assert cfg.model.net.d_out == 7
    assert cfg.callbacks.log_per_param_mse.param_spec == "surge_4"
    # plot_proj_ii's projection plots don't apply to the surge_4 spec; the
    # experiment disables it like its ffn_full/ffn_simple siblings.
    assert cfg.callbacks.plot_proj_ii is None


def test_surge_4_eval_experiment_composes_in_predict_mode() -> None:
    """``surge/eval_ffn_4`` evaluates a surge_4 FFN checkpoint in predict mode.

    Pins the surge_4 eval contract: predict mode with VST rendering and metrics,
    the surge_4 render group, and a mandatory ``ckpt_path``.
    """
    cfg = _compose("eval.yaml", ["experiment=surge/eval_ffn_4", "ckpt_path=dummy.ckpt"])

    assert cfg.mode == "predict"
    assert cfg.render.param_spec_name == "surge_4"
    assert cfg.datamodule.param_spec_name == "surge_4"
    assert cfg.model.net.d_out == 7
    assert cfg.evaluation.render_vst is True
    assert cfg.evaluation.compute_metrics is True
    assert cfg.evaluation.rerender_target is False
    assert cfg.ckpt_path == "dummy.ckpt"
    # eval.yaml defaults logger to null; the experiment must re-select the
    # wandb group or base.yaml's logger.wandb fragment dangles.
    assert cfg.logger.wandb._target_ == "lightning.pytorch.loggers.wandb.WandbLogger"
    # eval_vst callbacks: the prediction writer must be present.
    assert "prediction_writer" in cfg.callbacks


def test_ffn_smoke_experiment_wires_surge_xt_fixture_source() -> None:
    """``experiment=surge/ffn_smoke`` bakes in the R2 surge_xt fixture and smoke caps.

    Pins the contract that lets the experiment run end-to-end with no pre-staged
    local data: the opt-in R2 download URI; the batch size and single-process
    loading that the 20-sample train split forces; the 10-step cap with the
    surge-default 1M ``min_steps`` floor dropped; the surge_xt spec wiring
    (datamodule param spec + LogPerParamMSE callback) and the output width
    inherited from ``ffn_full``; and the disabled ``compile`` that keeps the
    fit + test setup from double-compiling.
    """
    cfg = _compose("train.yaml", ["experiment=surge/ffn_smoke"])

    assert cfg.datamodule.download_dataset_root_uri == (
        "r2://intermediate-data/fixtures/smoke-shard-surge-xt-v1/"
    )
    assert cfg.datamodule.batch_size == 4
    assert cfg.datamodule.num_workers == 0
    assert cfg.datamodule.param_spec_name == "surge_xt"
    assert cfg.callbacks.log_per_param_mse.param_spec == "surge_xt"
    assert cfg.trainer.max_steps == 10
    assert cfg.trainer.min_steps is None
    assert cfg.model.net.d_out == 300
    assert cfg.model.compile is False
