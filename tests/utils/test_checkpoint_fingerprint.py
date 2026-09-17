"""Tests for the checkpoint fingerprint model and its cfg extraction."""

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig, OmegaConf

from synth_setter.utils.checkpoint_fingerprint import (
    CheckpointFingerprint,
    fingerprint_from_cfg,
    fingerprint_sidecar_uri,
)


def _train_cfg() -> DictConfig:
    """Build a cfg carrying the architecture-identity nodes the fingerprint reads.

    :returns: A cfg with ``model`` and ``synth`` nodes populated.
    """
    return cast(
        DictConfig,
        OmegaConf.create(
            {
                "model": {
                    "_target_": "synth_setter.models.vst_flow_matching_module.VSTFlowMatchingModule",
                    "conditioning": "mel",
                    "encoder": {"_target_": "synth_setter.models.components.ast.AST"},
                    "vector_field": {
                        "_target_": "synth_setter.models.components.transformer.ApproxEquivTransformer"
                    },
                },
                "synth": {"param_spec_name": "surge_simple"},
            }
        ),
    )


def test_fingerprint_from_cfg_extracts_architecture_identity_fields() -> None:
    """Every identity node in the cfg lands in its fingerprint field."""
    fingerprint = fingerprint_from_cfg(_train_cfg())

    assert fingerprint == CheckpointFingerprint(
        model_target="synth_setter.models.vst_flow_matching_module.VSTFlowMatchingModule",
        encoder_target="synth_setter.models.components.ast.AST",
        vector_field_target="synth_setter.models.components.transformer.ApproxEquivTransformer",
        conditioning="mel",
        param_spec_name="surge_simple",
    )


def test_fingerprint_from_cfg_missing_nodes_yield_none_fields() -> None:
    """An empty cfg fingerprints as all-None instead of raising."""
    fingerprint = fingerprint_from_cfg(cast(DictConfig, OmegaConf.create({})))

    assert fingerprint == CheckpointFingerprint()


def test_fingerprint_from_cfg_dict_conditioning_is_plain_container() -> None:
    """Dict conditioning is captured as a plain (JSON-serializable) container."""
    cfg = _train_cfg()
    cfg.model.conditioning = OmegaConf.create({"column": "m2l", "input_shape": [128, 42]})

    fingerprint = fingerprint_from_cfg(cfg)

    assert fingerprint.conditioning == {"column": "m2l", "input_shape": [128, 42]}


def test_fingerprint_json_round_trip_preserves_equality() -> None:
    """A fingerprint survives its JSON sidecar round trip unchanged."""
    fingerprint = fingerprint_from_cfg(_train_cfg())

    restored = CheckpointFingerprint.model_validate_json(fingerprint.model_dump_json())

    assert restored == fingerprint


def test_fingerprint_inequality_on_encoder_swap() -> None:
    """Swapping the encoder class changes the fingerprint (the #2588 A/B case)."""
    baseline = fingerprint_from_cfg(_train_cfg())
    fourier_cfg = _train_cfg()
    fourier_cfg.model.encoder._target_ = "synth_setter.models.components.fourier_number.Fourier"

    assert fingerprint_from_cfg(fourier_cfg) != baseline


def test_fingerprint_sidecar_uri_appends_suffix_to_checkpoint_uri() -> None:
    """The sidecar URI is the checkpoint URI plus the JSON suffix."""
    uri = fingerprint_sidecar_uri("r2://intermediate-data/checkpoints/flow-simple/model.ckpt")

    assert uri == ("r2://intermediate-data/checkpoints/flow-simple/model.ckpt.fingerprint.json")
