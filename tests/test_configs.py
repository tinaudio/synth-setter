"""Tests that Hydra config groups compose without errors."""

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError, MissingMandatoryValue

from synth_setter.clap import (
    DEFAULT_CLAP_TRAINING_CHECKPOINT,
    DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256,
)
from synth_setter.conditioning import NUM_SKETCH_CONTROLS
from synth_setter.data.vst.param_spec_registry import param_specs, resolve_param_spec_width
from synth_setter.models.vst_flowvae_module import VSTFlowVAEModule
from synth_setter.pipeline.data.matpac_plus import MATPAC_PLUS_FRONTEND
from synth_setter.pipeline.data.meanaudio import MEANAUDIO_EMBEDDING_DIM
from synth_setter.pipeline.data.t5gemma import T5GEMMA_EMBEDDING_DIM, T5GEMMA_MAX_LENGTH
from synth_setter.pupujepa import (
    DEFAULT_PUPUJEPA_TINY_CHECKPOINT,
    PUPUJEPA_CHECKPOINT_REVISION,
)
from synth_setter.resources import configs_dir
from synth_setter.utils import extras
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
        ("surge/flow_full", "surge/test-mps-flow"),
    ],
    ids=["fake_oracle", "ffn_full", "flow_full"],
)
def test_test_mps_yaml_matches_cfg_surge_xt_global(experiment: str, test_mps_yaml: str) -> None:
    """Each ``surge/test-mps-*.yaml`` matches the smoke fixture's MPS cfg for its experiment.

    Guard against silent drift: the fixture's open_dict bake-ins and each YAML's
    defaults list / overrides must stay in lockstep, otherwise a test that uses one
    and an ``experiment=surge/test-mps-*`` invocation that uses the other will produce
    different runs without anyone noticing. Builds both configs in-process (no MPS
    hardware needed — only the cfg shape is compared, not runtime behavior).

    :param experiment: Hydra ``experiment=...`` override the fixture is built against
        (for example, ``"surge/flow_full"``).
    :param test_mps_yaml: Sibling smoke YAML the fixture is compared against
        (for example, ``"surge/test-mps-flow"``).
    """
    fixture_cfg = _build_surge_xt_smoke_cfg(
        accelerator="mps", param_spec_name="surge_4", experiment=experiment
    )
    fixture_param_spec = fixture_cfg.datamodule.param_spec_name
    GlobalHydra.instance().clear()

    with initialize_config_module(version_base="1.3", config_module="synth_setter.configs"):
        experiment_cfg = compose(
            config_name="train.yaml",
            return_hydra_config=False,
            overrides=[
                f"experiment={test_mps_yaml}",
                f"synth={fixture_param_spec}",
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


@pytest.mark.parametrize(
    ("config_name", "overrides"),
    [
        pytest.param(
            "train.yaml",
            ["datamodule=pyfdn", "synth=pyfdn_n8_mono", "model=vst_flow"],
            id="datamodule",
        ),
        pytest.param("train.yaml", ["experiment=pyfdn/flow"], id="train"),
        pytest.param(
            "eval.yaml",
            ["experiment=pyfdn/flow", "ckpt_path=null"],
            id="eval",
        ),
    ],
)
def test_pyfdn_configs_compose_without_external_source(
    config_name: str,
    overrides: list[str],
) -> None:
    """PyFDN datamodule, train, and eval configs need no source path or digest.

    :param config_name: Top-level Hydra config under test.
    :param overrides: pyFDN selection for the composition case.
    """
    cfg = _compose(config_name, overrides)

    assert "source_audio_path" not in cfg.datamodule
    assert "source_audio_sha256" not in cfg.datamodule
    hydra.utils.instantiate(cfg.datamodule)


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
    ("profile", "input_shape"),
    [
        pytest.param("same_s", (256, 44), id="same-s"),
        pytest.param("same_l", (256, 44), id="same-l"),
        pytest.param("t5gemma", (T5GEMMA_EMBEDDING_DIM, T5GEMMA_MAX_LENGTH), id="t5gemma"),
        pytest.param("matpac_plus", (MATPAC_PLUS_FRONTEND.embedding_dim, 25), id="matpac_plus"),
        pytest.param("meanaudio_16k", (MEANAUDIO_EMBEDDING_DIM, 125), id="meanaudio_16k"),
    ],
)
def test_sequence_conditioning_profile_fake_batch_pools_through_encoder(
    profile: str, input_shape: tuple[int, int]
) -> None:
    """A sequence profile routes its declared fake batch through the encoder.

    :param profile: Conditioning profile under test.
    :param input_shape: Per-row shape expected from the profile.
    """
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            "synth=surge_xt",
            "model=vst_flow",
            f"conditioning={profile}",
            "trainer=cpu",
            "paths.output_dir=/tmp/synth-setter-test",
            "+datamodule.fake=true",
            "datamodule.batch_size=2",
            "datamodule.num_workers=0",
            "datamodule.persistent_workers=false",
        ],
    )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    encoder = hydra.utils.instantiate(cfg.model.encoder)

    assert datamodule.embedding_conditioning is not None
    assert datamodule.embedding_conditioning.column == profile
    assert datamodule.embedding_conditioning.input_shape == input_shape
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["conditioning"].shape == (2, *input_shape)
    pooled = encoder(batch["conditioning"])
    assert pooled.shape == (2, cfg.model.vector_field.d_model)
    assert cfg.model.conditioning.column == profile


def test_sketch_on_profile_composes_with_m2l_and_trains_one_step() -> None:
    """``sketch=on`` composes over ``conditioning=m2l`` and drives a train step."""
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            "synth=surge_4",
            "model=vst_flow",
            "conditioning=m2l",
            "sketch=on",
            "trainer=cpu",
            # The scheduler's T_max interpolates ${trainer.max_steps}, which
            # trainer/cpu.yaml leaves undefined.
            "+trainer.max_steps=1",
            "paths.output_dir=/tmp/synth-setter-test",
            "+datamodule.fake=true",
            "datamodule.batch_size=2",
            "datamodule.num_workers=0",
            "datamodule.persistent_workers=false",
            "model.compile=false",
            "model.vector_field.num_layers=1",
            "model.vector_field.d_model=32",
            "model.vector_field.d_ff=32",
            "model.vector_field.projection.num_tokens=8",
        ],
    )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)

    assert datamodule.sketch_controls is not None
    assert datamodule.sketch_controls.column == "sketch"
    assert model.sketch_tokens is not None
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["conditioning"].shape == (2, 128, 42)
    assert batch["sketch_ctrl"].shape == (2, NUM_SKETCH_CONTROLS, 32)
    loss = model._train_step(batch).loss  # noqa: SLF001
    assert torch.isfinite(loss)


def _compose_t5gemma_cached_train_cfg(
    model_name: str, model_overrides: Sequence[str]
) -> DictConfig:
    """Compose a one-step CPU train cfg reading synthetic T5Gemma batches.

    :param model_name: VST model config selected for the training step.
    :param model_overrides: Tiny-network overrides that keep the regression CPU-fast.
    :returns: The composed config.
    """
    return _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            "synth=surge_xt",
            f"model={model_name}",
            "conditioning=t5gemma",
            "trainer=cpu",
            "+trainer.max_steps=1",
            "paths.output_dir=/tmp/synth-setter-test",
            "+datamodule.fake=true",
            "datamodule.batch_size=2",
            "datamodule.num_workers=0",
            "datamodule.persistent_workers=false",
            "model.compile=false",
            *model_overrides,
        ],
    )


@pytest.mark.parametrize(
    ("model_name", "model_overrides", "expected_output_dim"),
    [
        (
            "vst_flow",
            [
                "model.vector_field.d_ff=16",
                "model.vector_field.d_model=16",
                "model.vector_field.num_heads=1",
                "model.vector_field.num_layers=1",
                "model.vector_field.projection.num_tokens=4",
            ],
            16,
        ),
        (
            "vst_ffn",
            [
                "model.net.d_model=16",
                "model.net.n_heads=1",
                "model.net.n_layers=1",
            ],
            300,
        ),
    ],
    ids=["flow", "feed_forward"],
)
def test_t5gemma_conditioning_profile_cached_batch_trains(
    model_name: str,
    model_overrides: list[str],
    expected_output_dim: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The T5Gemma profile trains either VST model from its cached Lance tensor.

    :param model_name: VST model config selected for the training step.
    :param model_overrides: Tiny-network overrides that keep the regression CPU-fast.
    :param expected_output_dim: Model-owned cached encoder output width.
    :param monkeypatch: Pytest fixture used to detach Lightning logging from a Trainer.
    """
    cfg = _compose_t5gemma_cached_train_cfg(model_name, model_overrides)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)

    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    loss = model.training_step(batch, batch_idx=0)

    assert datamodule.embedding_conditioning is not None
    assert datamodule.embedding_conditioning.column == "t5gemma"
    assert datamodule.embedding_conditioning.input_shape == (768, 256)
    assert batch["conditioning"].shape == (2, 768, 256)
    assert cfg.model.encoder_output_dim == expected_output_dim
    assert cfg.model.encoder.d_model == expected_output_dim
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_clap_conditioning_overrides_compose_and_instantiate() -> None:
    """A CLAP spec selects generic routing and the vector projection encoder."""
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            "synth=surge_xt",
            "model=vst_flow",
            "model/encoder=vector_projection",
            "trainer=cpu",
            "paths.output_dir=/tmp/synth-setter-test",
            "model.conditioning={column:clap,input_shape:[512]}",
            "datamodule.conditioning={column:clap,input_shape:[512]}",
        ],
    )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    encoder = hydra.utils.instantiate(cfg.model.encoder)

    assert datamodule.embedding_conditioning is not None
    assert datamodule.embedding_conditioning.column == "clap"
    assert encoder(torch.randn(2, 512)).shape == (2, 512)
    assert cfg.model.vector_field.conditioning_dim == 512


def test_ssondo_conditioning_profile_projects_960_vector() -> None:
    """The S-SONDO profile routes its global vector through projection."""
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            "synth=surge_xt",
            "model=vst_flow",
            "conditioning=ssondo",
            "trainer=cpu",
            "paths.output_dir=/tmp/synth-setter-test",
        ],
    )

    encoder = hydra.utils.instantiate(cfg.model.encoder)

    assert cfg.datamodule.conditioning.column == "ssondo"
    assert tuple(cfg.datamodule.conditioning.input_shape) == (960,)
    assert encoder(torch.randn(2, 960)).shape == (2, cfg.model.encoder_output_dim)


def _conditioning_profile_names() -> list[str]:
    """Enumerate every shipped conditioning profile.

    :returns: Sorted conditioning-profile names.
    """
    return sorted(
        entry.name.removesuffix(".yaml")
        for entry in (configs_dir() / "conditioning").iterdir()
        if entry.name.endswith(".yaml")
    )


# Waveform profiles interpolate datamodule geometry, which this bare composition lacks.
_WAVEFORM_CONDITIONING_PROFILES = frozenset(
    {
        "ast_online",
        "clap_online",
        "log_mel",
        "pupujepa_large_online",
        "pupujepa_tiny_online",
        "same_l_online",
        "same_s_online",
    }
)
_CACHED_CONDITIONING_PROFILES = [
    profile
    for profile in _conditioning_profile_names()
    if profile not in _WAVEFORM_CONDITIONING_PROFILES
]


@pytest.mark.parametrize("profile", _CACHED_CONDITIONING_PROFILES)
@pytest.mark.parametrize("model_name", ["vst_ffn", "vst_flow", "vst_flowmlp"])
def test_embedding_conditioning_profile_encoder_matches_model_output(
    profile: str, model_name: str
) -> None:
    """Every cached profile produces the output width its VST model owns.

    :param profile: Conditioning profile under test.
    :param model_name: VST architecture consuming the profile.
    """
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            "synth=surge_xt",
            f"model={model_name}",
            f"conditioning={profile}",
            "trainer=cpu",
            "paths.output_dir=/tmp/synth-setter-test",
        ],
    )
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    input_shape = tuple(cfg.model.conditioning.input_shape)

    encoded = encoder(torch.randn(2, *input_shape))

    assert encoded.shape == (2, cfg.model.encoder_output_dim)


@pytest.mark.parametrize("profile", _conditioning_profile_names())
def test_eval_config_conditioning_profile_composes(profile: str) -> None:
    """Regression guard for #2304: eval.yaml accepts every ``conditioning=`` profile.

    :param profile: Conditioning profile under test.
    """
    cfg = _compose(
        "eval.yaml",
        ["experiment=surge/flow_simple", f"conditioning={profile}", "trainer=cpu"],
    )

    # Raw modes are literals; cached profiles wire one shared column onto both sides.
    if isinstance(cfg.model.conditioning, str):
        assert cfg.datamodule.conditioning == cfg.model.conditioning
    else:
        column = cfg.model.conditioning.column
        assert column
        assert cfg.datamodule.conditioning.column == column


def test_clap_online_profile_matches_training_checkpoint_identity() -> None:
    """Online CLAP composition retains the shared production checkpoint identity."""
    cfg = _compose(
        "eval.yaml",
        ["experiment=surge/flow_simple", "conditioning=clap_online", "trainer=cpu"],
    )

    assert cfg.datamodule.conditioning == "audio"
    assert cfg.model.encoder.backbone.checkpoint == DEFAULT_CLAP_TRAINING_CHECKPOINT
    assert cfg.model.encoder.backbone.checkpoint_sha256 == DEFAULT_CLAP_TRAINING_CHECKPOINT_SHA256


def test_pupujepa_tiny_cached_profile_instantiates_100_patch_pool() -> None:
    """Four-second cached PupuJEPA sequences instantiate the generic pool."""
    cfg = _compose(
        "eval.yaml",
        ["experiment=surge/flow_simple", "conditioning=pupujepa_tiny", "trainer=cpu"],
    )

    encoder = hydra.utils.instantiate(cfg.model.encoder)

    assert tuple(cfg.model.conditioning.input_shape) == (1536, 100)
    assert encoder(torch.randn(2, 1536, 100)).shape == (2, cfg.model.encoder_output_dim)


def test_pupujepa_large_cached_profile_instantiates_100_patch_pool() -> None:
    """Four-second cached Large sequences instantiate the generic pool."""
    cfg = _compose(
        "eval.yaml",
        ["experiment=surge/flow_simple", "conditioning=pupujepa_large", "trainer=cpu"],
    )

    encoder = hydra.utils.instantiate(cfg.model.encoder)

    assert tuple(cfg.model.conditioning.input_shape) == (8192, 100)
    assert encoder(torch.randn(2, 8192, 100)).shape == (2, cfg.model.encoder_output_dim)


def test_pupujepa_large_online_profile_pins_variant_and_width() -> None:
    """Online Large composition selects the immutable checkpoint and teacher width."""
    cfg = _compose(
        "eval.yaml",
        [
            "experiment=surge/flow_simple",
            "conditioning=pupujepa_large_online",
            "trainer=cpu",
        ],
    )

    assert cfg.datamodule.conditioning == "audio"
    assert cfg.model.encoder.backbone.checkpoint == DEFAULT_PUPUJEPA_TINY_CHECKPOINT
    assert cfg.model.encoder.backbone.revision == PUPUJEPA_CHECKPOINT_REVISION
    assert cfg.model.encoder.backbone.variant == "large"
    assert cfg.model.encoder.head.embed_dim == 8192


def test_pupujepa_tiny_online_profile_pins_checkpoint_identity() -> None:
    """Online PupuJEPA composition retains the immutable HF revision."""
    cfg = _compose(
        "eval.yaml",
        [
            "experiment=surge/flow_simple",
            "conditioning=pupujepa_tiny_online",
            "trainer=cpu",
        ],
    )

    assert cfg.datamodule.conditioning == "audio"
    assert cfg.model.encoder.backbone.checkpoint == DEFAULT_PUPUJEPA_TINY_CHECKPOINT
    assert cfg.model.encoder.backbone.revision == PUPUJEPA_CHECKPOINT_REVISION
    assert cfg.model.encoder.backbone.variant == "tiny"
    assert cfg.model.encoder.head.embed_dim == 1536


def test_pupujepa_online_head_pools_the_same_span_as_the_cached_profile() -> None:
    """Online and cached PupuJEPA profiles describe one four-second teacher sequence.

    The online head builds a persistent positional buffer from ``max_seq_len``, so a
    span wider than the render emits leaves trained rows the forward pass never reads.
    """
    cached = _compose(
        "eval.yaml",
        ["experiment=surge/flow_simple", "conditioning=pupujepa_tiny", "trainer=cpu"],
    )
    online = _compose(
        "eval.yaml",
        ["experiment=surge/flow_simple", "conditioning=pupujepa_tiny_online", "trainer=cpu"],
    )

    assert online.model.encoder.head.max_seq_len == cached.model.conditioning.input_shape[1]


def test_eval_config_conditioning_unset_composes() -> None:
    """eval.yaml still composes when the ``conditioning`` group is left at null."""
    cfg = _compose("eval.yaml", ["experiment=surge/flow_simple", "trainer=cpu"])

    assert cfg.datamodule
    assert cfg.model


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
    ("model_name", "width_path"),
    [
        ("vst_fake_oracle", "net.d_out"),
        ("vst_ffn", "net.d_out"),
        ("vst_flow", "num_params"),
        ("vst_flow", "vector_field.projection.num_params"),
        ("vst_flowmlp", "num_params"),
        ("vst_flowmlp", "vector_field.n_params"),
        ("vst_flowvae", "net.decoder.latent_dim"),
        ("vst_flowvae", "net.encoder.latent_dim"),
        ("vst_flowvae", "net.latent_dim"),
    ],
)
# One non-default spec suffices: which spec is active doesn't change whether a config
# path interpolates the resolver. Exact per-spec widths are pinned in
# tests/data/vst/test_param_spec_registry.py.
def test_vst_model_width_derives_from_active_param_spec(
    model_name: str,
    width_path: str,
) -> None:
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            "synth=obxf",
            f"model={model_name}",
            "trainer=cpu",
        ],
    )

    assert OmegaConf.select(cfg.model, width_path) == resolve_param_spec_width("obxf")


# Parameterized over the live registry so newly registered synths are covered
# automatically; one representative width path suffices since path wiring is
# spec-independent (covered above).
@pytest.mark.parametrize("param_spec_name", sorted(param_specs))
def test_vst_model_width_resolves_for_every_registered_spec(param_spec_name: str) -> None:
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            f"synth={param_spec_name}",
            "model=vst_ffn",
            "trainer=cpu",
        ],
    )

    assert OmegaConf.select(cfg.model, "net.d_out") == resolve_param_spec_width(param_spec_name)


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


@pytest.fixture(scope="module")
def flowvae_module() -> VSTFlowVAEModule:
    """Build a small random-weight Flow-VAE module for batch-contract tests.

    :returns: Eval-mode module configured for the 187-wide OB-Xf parameter spec.
    """
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_lance",
            "synth=obxf",
            "model=vst_flowvae",
            "trainer=cpu",
            "+model.net.latent_flow_num_layers=2",
            "+model.net.latent_flow_hidden_dim=16",
            "+model.net.regression_flow_num_layers=2",
            "+model.net.regression_flow_hidden_dim=16",
        ],
    )
    module = VSTFlowVAEModule(
        net=hydra.utils.instantiate(cfg.model.net),
        optimizer=partial(torch.optim.Adam, lr=1e-3),  # pyright: ignore[reportArgumentType]
        scheduler=None,  # pyright: ignore[reportArgumentType]
        param_spec="obxf",
    )
    module.eval()
    return module


def test_flowvae_model_step_reads_mel_batch_key(flowvae_module: VSTFlowVAEModule) -> None:
    """The training path consumes ``mel`` and preserves its output contracts.

    :param flowvae_module: Small random-weight module under test.
    """
    mel = torch.randn(2, 2, 128, 401)
    params = torch.rand(2, 187) * 2 - 1

    losses, actual_mel, actual_params, out = flowvae_module.model_step(
        {"mel": mel, "params": params}
    )

    assert set(losses) == {"reconstruction_loss", "latent_loss", "param_loss"}
    assert actual_mel is mel
    assert actual_params is params
    assert out.x_hat.shape == (2, 187)
    assert out.y_hat.shape == (2, 2, 128, 401)


def test_flowvae_predict_step_reads_mel_batch_key(flowvae_module: VSTFlowVAEModule) -> None:
    """The prediction path consumes ``mel`` and returns model-width parameters.

    :param flowvae_module: Small random-weight module under test.
    """
    batch = {"mel": torch.randn(2, 2, 128, 401)}

    predictions, actual_batch = flowvae_module.predict_step(batch, batch_idx=0)

    assert predictions.shape == (2, 187)
    assert actual_batch is batch


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
            92,
            "surge_simple",
            True,
            1e-4,
        ),
        (
            "surge_flowmlp",
            "vst_flow_matching_module.VSTFlowMatchingModule",
            "num_params",
            92,
            "surge_simple",
            True,
            1e-4,
        ),
        (
            "surge_flowvae",
            "vst_flowvae_module.VSTFlowVAEModule",
            "net.latent_dim",
            92,
            "surge_simple",
            True,
            2e-4,
        ),
    ],
)
def test_legacy_surge_model_group_composes_canonical_defaults(
    legacy_name: str,
    target_suffix: str,
    expected_path: str,
    expected_value: int,
    expected_param_spec: str | None,
    expected_compile: bool,
    expected_learning_rate: float,
) -> None:
    """Each legacy model selection composes the canonical VST defaults.

    :param legacy_name: Legacy Hydra model-group name.
    :param target_suffix: Canonical VST target expected after composition.
    :param expected_path: Config field containing the canonical default.
    :param expected_value: Expected canonical default.
    :param expected_param_spec: Default ParamSpec, when applicable.
    :param expected_compile: Expected torch.compile default.
    :param expected_learning_rate: Expected optimizer learning-rate default.
    """
    legacy_cfg = _compose(
        "train.yaml",
        ["datamodule=surge_simple", "synth=surge_simple", f"model={legacy_name}", "trainer=cpu"],
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


def test_log_per_param_mse_config_uses_active_synth_spec() -> None:
    """The VST per-parameter callback resolves the selected synth's spec."""
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=surge_mini",
            "synth=surge_4",
            "model=ffn",
            "callbacks=log_per_param_mse",
            "trainer=cpu",
        ],
    )

    assert cfg.callbacks.log_per_param_mse.param_spec == "surge_4"


def test_log_per_param_mse_config_requires_synth_selection() -> None:
    """The VST per-parameter callback rejects a run with no synth selected."""
    cfg = _compose(
        "train.yaml",
        [
            "datamodule=vst",
            "model=ffn",
            "callbacks=log_per_param_mse",
            "trainer=cpu",
        ],
    )

    with pytest.raises(InterpolationKeyError, match="synth"):
        OmegaConf.to_container(cfg.callbacks, resolve=True, throw_on_missing=True)


@pytest.mark.parametrize("model_name", ["vst_flow", "vst_flowmlp"])
def test_vst_flow_config_uses_active_synth_spec_for_structured_metrics(model_name: str) -> None:
    """Every flow model receives the selected ParamSpec for number-group swaps.

    :param model_name: Hydra flow-model group under test.
    """
    cfg = _compose(
        "train.yaml",
        ["datamodule=surge_simple", "synth=surge_simple", f"model={model_name}", "trainer=cpu"],
    )

    assert cfg.model.param_spec == "surge_simple"


def test_surge_training_defaults_enable_bounded_validation_and_auto_probe() -> None:
    """The surge family validates a bounded sample and enables the probe when usable."""
    cfg = _compose("train.yaml", ["experiment=surge/flow_simple"])

    assert cfg.trainer.limit_val_batches == 20
    assert cfg.training.val_audio_probe == "auto"


def test_surge_eval_composition_retains_default_per_param_mse_callback() -> None:
    """The surge experiment default applies when composed through the eval entrypoint."""
    cfg = _compose(
        "eval.yaml",
        ["experiment=surge/ffn_simple", "ckpt_path=fake.ckpt", "trainer=cpu"],
    )

    assert cfg.callbacks.log_per_param_mse.param_spec == "surge_simple"


def test_surge_4_generate_dataset_experiment_composes_with_inline_finalize() -> None:
    """``generate_dataset/surge-4-lance-440k-20k-20k`` wires surge_4 and inline finalize.

    Pins the full-scale surge_4 Lance pipeline contract: the surge_4 render and
    datamodule groups, Lance output, the 440k/20k/20k split, and the inline
    finalize that writes ``dataset.complete`` in the same CLI process.
    """
    cfg = _compose("dataset.yaml", ["experiment=generate_dataset/surge-4-lance-440k-20k-20k"])

    assert cfg.synth.param_spec_name == "surge_4"
    assert cfg.synth.plugin_state_path == "presets/surge-mini.vstpreset"
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
    assert cfg.synth.param_spec_name == "surge_4"
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


def test_flow_simple_440k_experiment_owns_dataset_pin_and_training_cadence() -> None:
    """``surge/flow_simple_440k`` bakes the 440k contract previously spread into launch YAML.

    Pins the one-selector contract (#2196): the immutable 440k dataset root, the
    surge_simple spec on both datamodule and render, the enabled val audio probe,
    validation cadence and checkpoint monitor, and the disabled test stage — so
    ``experiment=surge/flow_simple_440k`` alone reproduces the full 440k run.
    """
    cfg = _compose("train.yaml", ["experiment=surge/flow_simple_440k"])

    assert cfg.datamodule.download_dataset_root_uri == (
        "r2://experiments/data/surge-simple-lance-440k-20k-20k/"
        "surge-simple-lance-440k-20k-20k-20260706T005448315Z/"
    )
    assert cfg.datamodule.param_spec_name == "surge_simple"
    assert cfg.synth.param_spec_name == "surge_simple"
    assert cfg.training.val_audio_probe is True
    assert cfg.trainer.val_check_interval == 2000
    assert cfg.trainer.limit_val_batches == 20
    assert cfg.callbacks.model_checkpoint.monitor == "val/param_mse"
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 1000
    assert cfg.test is False


def test_vst_flow_dropout_defaults_match_flash_foley_policy() -> None:
    """Content, sketch-group, and global CFG dropout share Flash Foley's rate."""
    cfg = _compose("train.yaml", ["experiment=surge/flow_sketch_prelim"])

    assert cfg.model.cfg_dropout_rate == 0.1
    assert cfg.model.sketch_dropout_rate == 0.1
    assert cfg.model.all_conditioning_dropout_rate == 0.1


def test_flow_sketch_prelim_experiments_differ_only_in_sketch_conditioning() -> None:
    """The preliminary A/B arms differ only in sketch conditioning."""
    base = _compose("train.yaml", ["experiment=surge/flow_sketch_prelim_base"])
    sketch = _compose("train.yaml", ["experiment=surge/flow_sketch_prelim"])

    for cfg in (base, sketch):
        assert cfg.datamodule.download_dataset_root_uri == (
            "r2://experiments/data/surge-simple-lance-1k-2k-2k/"
            "surge-simple-lance-1k-2k-2k-20260716T163226347Z/"
        )
        assert cfg.seed == 3407
        assert cfg.datamodule.param_spec_name == "surge_simple"
        assert cfg.trainer.max_steps == 10000
        assert cfg.trainer.min_steps == 10000
        assert cfg.trainer.val_check_interval == 1000
        assert cfg.training.val_audio_probe is True
        assert cfg.test is False
    assert base.run_name == "flow1k_prelim_base"
    assert base.model.sketch_controls is None
    assert base.datamodule.sketch is None
    assert sketch.run_name == "flow1k_prelim_sketch"
    assert sketch.model.sketch_controls.column == "sketch"
    assert sketch.model.sketch_controls.num_frames == 32
    assert sketch.datamodule.sketch == sketch.model.sketch_controls


def test_ffn_simple_smoke_experiment_pins_lance_fixture_and_smoke_caps() -> None:
    """``surge/ffn_simple_smoke`` bakes the RunPod smoke contract into one experiment.

    Pins the one-selector smoke contract (#2196): the small finalized R2 root a fresh pod can
    hydrate, single-process loading, the 10-step cap, and a checkpoint cadence that guarantees a
    file exists at step 10.
    """
    cfg = _compose("train.yaml", ["experiment=surge/ffn_simple_smoke"])

    assert cfg.datamodule.download_dataset_root_uri == (
        "r2://experiments/data/surge-simple-lance-1k-2k-2k/"
        "surge-simple-lance-1k-2k-2k-20260716T163226347Z/"
    )
    assert cfg.datamodule.num_workers == 0
    assert cfg.datamodule.param_spec_name == "surge_simple"
    assert cfg.trainer.max_steps == 10
    assert cfg.trainer.min_steps == 10
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 5


def test_ffn_smoke_experiment_wires_surge_xt_fixture_source() -> None:
    """``experiment=surge/ffn_smoke`` bakes in the R2 surge_xt root and smoke caps.

    Pins the contract that lets the experiment run end-to-end with no pre-staged
    local data: the opt-in R2 download URI and its row cap; the batch size and
    single-process loading the capped subset forces; the 10-step cap with the
    surge-default 1M ``min_steps`` floor dropped; the surge_xt spec wiring
    (datamodule param spec + LogPerParamMSE callback) and the output width
    inherited from ``ffn_full``; and the disabled ``compile`` that keeps the
    fit + test setup from double-compiling.
    """
    cfg = _compose("train.yaml", ["experiment=surge/ffn_smoke"])

    assert cfg.datamodule.download_dataset_root_uri == (
        "r2://experiments/data/surge-xt-lance-1k-2k-2k/"
        "surge-xt-lance-1k-2k-2k-20260717T042205119Z/"
    )
    assert cfg.datamodule.download_dataset_row_limit == 64
    assert cfg.datamodule.batch_size == 4
    assert cfg.datamodule.num_workers == 0
    assert cfg.datamodule.param_spec_name == "surge_xt"
    assert cfg.callbacks.log_per_param_mse.param_spec == "surge_xt"
    assert cfg.trainer.max_steps == 10
    assert cfg.trainer.min_steps is None
    assert cfg.model.net.d_out == 300
    assert cfg.model.compile is False


# Resolved-identity guard for the synth-identity hoist (#2565): each surge
# experiment must keep pairing this synth spec with these resolved widths and
# callback labels across the ownership migration to the root `synth` group.
SURGE_EXPERIMENT_IDENTITY_CASES: tuple[tuple[str, str, str, str], ...] = (
    ("surge/ffn_full", "surge_xt", "model.net.d_out", "surge_xt"),
    ("surge/ffn_4", "surge_4", "model.net.d_out", "surge_4"),
    ("surge/ffn_simple", "surge_simple", "model.net.d_out", "surge_simple"),
    ("surge/flow_full", "surge_xt", "model.num_params", "surge_xt"),
    ("surge/flow_simple", "surge_simple", "model.num_params", "surge_simple"),
    ("surge/flow_mlp_full", "surge_xt", "model.num_params", "surge_xt"),
    ("surge/fake_oracle", "surge_xt", "model.net.d_out", "surge_xt"),
    ("surge/vae_full", "surge_xt", "model.net.latent_dim", "surge_xt"),
)


@pytest.mark.parametrize(
    ("experiment", "param_spec", "width_path", "callback_spec"),
    SURGE_EXPERIMENT_IDENTITY_CASES,
    ids=[case[0].removeprefix("surge/") for case in SURGE_EXPERIMENT_IDENTITY_CASES],
)
def test_surge_experiment_resolves_consistent_synth_identity(
    experiment: str, param_spec: str, width_path: str, callback_spec: str
) -> None:
    """Each surge experiment resolves one synth spec across datamodule, model, and callback.

    :param experiment: Hydra ``experiment=...`` override under test.
    :param param_spec: Spec the experiment's synth selection must resolve to.
    :param width_path: Model key sized from the spec's parameter width.
    :param callback_spec: Expected ``log_per_param_mse.param_spec``.
    """
    cfg = _compose("train.yaml", [f"experiment={experiment}", "trainer=cpu"])

    assert cfg.datamodule.param_spec_name == param_spec
    assert OmegaConf.select(cfg, width_path) == resolve_param_spec_width(param_spec)
    assert OmegaConf.select(cfg, "callbacks.log_per_param_mse.param_spec") == callback_spec


# Compose-level guard that swapping in an audio datamodule cannot break model
# or callback identity, for each `jobs/predict/` experiment.
AUDIO_EXPERIMENT_SWAP_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "surge/ffn_full",
        "model.encoder_output_dim",
        "callbacks.log_per_param_mse.param_spec",
    ),
    (
        "surge/ffn_full",
        "model.net.d_out",
        "callbacks.log_per_param_mse.param_spec",
    ),
    (
        "surge/flow_full",
        "model.num_params",
        "callbacks.log_per_param_mse.param_spec",
    ),
    (
        "surge/flow_mlp_full",
        "model.num_params",
        "callbacks.log_per_param_mse.param_spec",
    ),
    ("surge/vae_full", "model.net.latent_dim", "model.param_spec"),
)


@pytest.mark.parametrize(
    ("experiment", "width_path", "spec_path"),
    AUDIO_EXPERIMENT_SWAP_CASES,
    ids=[
        f"{case[0].rsplit('/', 1)[-1]}-{case[1].rsplit('.', 1)[-1]}"
        for case in AUDIO_EXPERIMENT_SWAP_CASES
    ],
)
def test_surge_experiment_resolves_identity_with_audio_datamodule(
    experiment: str, width_path: str, spec_path: str
) -> None:
    """Swapping in an audio datamodule leaves model identity resolvable.

    :param experiment: Surge experiment composed with ``datamodule=fsd``.
    :param width_path: Model width key that must still resolve to the surge_xt width.
    :param spec_path: Param-spec label key that must still resolve.
    """
    cfg = _compose(
        "eval.yaml",
        [
            f"experiment={experiment}",
            "datamodule=fsd",
            "callbacks=[eval_surge,log_per_param_mse]",
            "mode=predict",
            "ckpt_path=fake.ckpt",
        ],
    )

    assert "param_spec_name" not in cfg.datamodule
    assert OmegaConf.select(cfg, width_path) == resolve_param_spec_width("surge_xt")
    assert OmegaConf.select(cfg, spec_path) == "surge_xt"


def test_extras_rejects_pyfdn_datamodule_spec_skewed_from_synth_selection() -> None:
    """The shipped pyFDN identity field exposes CLI-forced skew to ``extras``."""
    cfg = _compose(
        "train.yaml",
        [
            "experiment=pyfdn/flow",
            "trainer=cpu",
            "datamodule.param_spec_name=surge_4",
        ],
    )

    with pytest.raises(ValueError, match="surge_4"):
        extras(cfg)


def test_extras_rejects_datamodule_spec_skewed_from_synth_selection() -> None:
    """``extras`` fails fast when a forced datamodule spec contradicts ``synth``.

    Guards the one hole the rootward interpolation leaves open: a CLI override
    like ``datamodule.param_spec_name=surge_4`` would otherwise train a model
    sized for the synth selection on data described by a different spec.
    """
    cfg = _compose(
        "train.yaml",
        [
            "experiment=surge/ffn_full",
            "trainer=cpu",
            "datamodule.param_spec_name=surge_4",
        ],
    )

    with pytest.raises(ValueError, match="surge_4"):
        extras(cfg)


def test_extras_validates_synth_before_missing_extras_early_return() -> None:
    """A registry-skewed ``synth`` node fails even when no ``extras`` block is composed."""
    cfg = OmegaConf.create(
        {
            "synth": {
                "name": "surge_xt",
                "param_spec_name": "surge_4",
                "plugin_path": "plugins/Surge XT.vst3",
                "plugin_state_path": "presets/surge-base.vstpreset",
                "synth_version": "1.3.4",
            }
        }
    )

    with pytest.raises(ValueError, match="surge_4"):
        extras(cfg)


@pytest.mark.parametrize(
    ("experiment", "control_mode"),
    [
        ("flow_finetune", "gradient_spectral"),
        ("flow_finetune_learned", "learned_audio"),
        ("flow_finetune_null", "null"),
    ],
)
def test_torchsynth_finetune_arm_composes_to_its_control_mode(
    experiment: str, control_mode: str
) -> None:
    """Each simulator-feedback arm selects its own control without further overrides.

    :param experiment: ``experiment=torchsynth/...`` name under test.
    :param control_mode: Control arm the experiment must select.
    """
    cfg = _compose(
        "train.yaml",
        [f"experiment=torchsynth/{experiment}", "trainer=cpu", "model.base_checkpoint=base.ckpt"],
    )

    assert cfg.model.control_mode == control_mode
    # The differentiable render graph-breaks under compile and is single-device (#2585).
    assert cfg.model.compile is False
    assert (cfg.model.control_encoder is not None) == (control_mode == "learned_audio")
    assert (cfg.model.cost is not None) == (control_mode == "gradient_spectral")


@pytest.mark.parametrize(
    "experiment", ["flow_finetune", "flow_finetune_learned", "flow_finetune_null"]
)
def test_torchsynth_finetune_arm_instantiates_from_its_experiment(
    experiment: str, tmp_path: Path
) -> None:
    """Each arm's experiment builds a real module through the operator-facing config.

    Composition alone would still pass with a wrong ``_target_``, a broken interpolation, or
    instantiation-time wiring that only fails once Hydra builds the object.

    :param experiment: ``experiment=torchsynth/...`` name under test.
    :param tmp_path: Pytest-provided directory for the base checkpoint.
    """
    # Small enough that three arms instantiate quickly; the wiring under test is the
    # experiment's targets and interpolations, not the field's width.
    small = [
        "trainer=cpu",
        # vst_flow's cosine scheduler interpolates it. `++` because the cpu trainer carries
        # no step cap while the finetune arms pin their own.
        "++trainer.max_steps=10",
        "datamodule.sample_rate=16000",
        "datamodule.signal_length=16384",
        "model.vector_field.num_layers=1",
        "model.vector_field.d_model=32",
        "model.vector_field.d_ff=32",
        "model.vector_field.projection.num_tokens=4",
        "model.encoder.backbone.out_dim=32",
        "model.encoder.backbone.hidden_dim=4",
    ]
    pretrained = hydra.utils.instantiate(
        _compose("train.yaml", ["experiment=torchsynth/flow", *small]).model
    )
    checkpoint = tmp_path / "base.ckpt"
    torch.save({"state_dict": pretrained.state_dict()}, checkpoint)

    cfg = _compose(
        "train.yaml",
        [
            f"experiment=torchsynth/{experiment}",
            *small,
            f"model.base_checkpoint={checkpoint}",
        ],
    )
    module = hydra.utils.instantiate(cfg.model)

    assert module.control_mode == cfg.model.control_mode
    # The control is the only thing this run trains.
    assert not any(p.requires_grad for p in module.vector_field.flow.parameters())
    assert any(p.requires_grad for p in module.vector_field.control.parameters())


def test_torchsynth_finetune_without_base_checkpoint_raises() -> None:
    """The finetune arms refuse to run against an unnamed pretrained flow."""
    cfg = _compose("train.yaml", ["experiment=torchsynth/flow_finetune", "trainer=cpu"])

    with pytest.raises(MissingMandatoryValue):
        _ = cfg.model.base_checkpoint


@pytest.mark.parametrize(
    ("corpus", "audio_column"),
    [
        pytest.param("nsynth_test", "audio", id="nsynth"),
        pytest.param("esc50", "audio_wav", id="esc50"),
    ],
)
def test_third_party_eval_config_resolves_per_corpus(corpus: str, audio_column: str) -> None:
    """Each published corpus is servable through config alone, on the render contract.

    :param corpus: Corpus config under ``datamodule/third_party``.
    :param audio_column: Blob column that corpus stores its audio in.
    """
    cfg = _compose(
        "eval.yaml",
        [
            f"datamodule=third_party/{corpus}",
            "synth=surge_simple",
            "render=vst",
            "model=vst_flow",
            "trainer=cpu",
            "mode=predict",
            "callbacks=eval_vst",
            "ckpt_path=/tmp/none.ckpt",
            "datamodule.mel_stats_uri=/tmp/training-stats.npz",
        ],
    )

    assert cfg.datamodule.audio_column == audio_column
    assert cfg.datamodule.dataset_version == 1
    assert cfg.datamodule.sample_rate == cfg.render.sample_rate
    assert cfg.datamodule.signal_duration_seconds == cfg.render.signal_duration_seconds
    assert cfg.datamodule.conditioning == "mel"


def test_nsynth_sketch_eval_config_pins_corpus_controls_and_training_statistics() -> None:
    """The dedicated NSynth config composes the held-out corpus onto the sketch contract."""
    cfg = _compose(
        "eval.yaml",
        [
            "datamodule=third_party/nsynth_sketch",
            "sketch=on",
            "synth=surge_simple",
            "render=vst",
            "model=vst_flow",
            "trainer=cpu",
            "mode=predict",
            "callbacks=eval_vst",
            "ckpt_path=/tmp/none.ckpt",
            "paths.output_dir=/tmp/synth-setter-test",
        ],
    )

    assert cfg.datamodule.dataset_uri == "s3://experiments/third_party/NSynth/test.lance"
    assert cfg.datamodule.dataset_version == 1
    assert cfg.datamodule.audio_column == "audio"
    assert cfg.datamodule.row_limit is None
    assert cfg.datamodule.batch_size == 32
    assert cfg.datamodule.mel_stats_uri == (
        "r2://experiments/data/surge-simple-surgepy-lance-2m-40k-10k/"
        "surge-simple-surgepy-lance-2m-40k-10k-20260824T195308545Z/stats.npz"
    )
    assert cfg.datamodule.sketch == cfg.model.sketch_controls
    assert cfg.datamodule.sketch.num_frames == 32
    assert cfg.datamodule.sketch.num_control_tokens == 32
    assert "param_spec_name" not in cfg.datamodule
    assert "live_embeddings" not in cfg.datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    assert datamodule.sketch_controls is not None


def test_third_party_eval_config_requires_checkpoint_mel_statistics() -> None:
    """Normalized third-party evaluation requires checkpoint-training statistics."""
    cfg = _compose(
        "eval.yaml",
        [
            "datamodule=third_party/nsynth_test",
            "synth=surge_simple",
            "render=vst",
            "model=vst_flow",
            "trainer=cpu",
            "mode=predict",
            "callbacks=eval_vst",
            "ckpt_path=/tmp/none.ckpt",
        ],
    )

    assert OmegaConf.is_missing(cfg.datamodule, "mel_stats_uri")
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.datamodule.mel_stats_uri


def test_nsynth_sketch_eval_experiment_pins_full_production_run() -> None:
    """One experiment selector pins the complete NSynth sketch evaluation."""
    cfg = _compose("eval.yaml", ["experiment=surge/eval_flow_sketch_nsynth"])

    assert cfg.mode == "predict"
    assert cfg.ckpt_path == (
        "r2://intermediate-data/checkpoints/flow_sketch_prelim/"
        "flow_sketch_prelim-20260902T044048985Z-eed5063da1164b1e92ac62a55ffc17b3/"
        "last.ckpt"
    )
    assert cfg.ckpt_sha256 == "d20cd4c3c86ae062a206f05596072b230c8aa86334920c775c2b4fec04aefc9e"
    assert cfg.consumed_train_artifact_alias == "v0"
    assert cfg.datamodule.dataset_uri == "s3://experiments/third_party/NSynth/test.lance"
    assert cfg.datamodule.dataset_version == 1
    assert cfg.datamodule.row_limit is None
    assert cfg.datamodule.mel_stats_uri == (
        "r2://experiments/data/surge-simple-surgepy-lance-2m-40k-10k/"
        "surge-simple-surgepy-lance-2m-40k-10k-20260824T195308545Z/stats.npz"
    )
    assert (
        cfg.datamodule.mel_stats_sha256
        == "c0c45d75a8b77004b3802c761bc77b5b34e7709a08343b2cf70fee04b7f52a19"
    )
    assert cfg.model.sketch_controls.num_frames == 32
    assert cfg.model.test_cfg_strength == 8.0
    assert cfg.model.test_sketch_cfg_strength == 8.0
    assert cfg.model.test_sample_steps == 200
    assert cfg.evaluation.render_vst is True
    assert cfg.evaluation.compute_metrics is True
    assert cfg.evaluation.no_params is True
    assert cfg.evaluation.rerender_target is False
    assert cfg.render.renderer_backend == "surgepy"
    assert cfg.logger.wandb.offline is False
