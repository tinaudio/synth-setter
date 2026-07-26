"""Tests for synth_setter.pipeline.ci.validate_spec validation functions."""

from __future__ import annotations

from synth_setter.pipeline.ci.validate_spec import (
    _REQUIRED_RENDER_FIELDS,
    _REQUIRED_TOP_LEVEL_FIELDS,
    validate_structure,
    validate_test_values,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig


def _make_valid_spec(*, output_format: str = "lance", **overrides: object) -> dict:
    """Build a minimal valid spec dict mirroring DatasetSpec.model_dump output."""
    ext = ".lance" if output_format == "lance" else f".{output_format}"
    spec: dict = {
        "task_name": "test",
        "run_id": "test-20260328T120000000Z",
        "created_at": "2026-03-28T12:00:00+00:00",
        "git_sha": "a" * 40,
        "is_repo_dirty": False,
        "output_format": output_format,
        "train_val_test_sizes": [32, 32, 32],
        "train_val_test_seeds": None,
        "base_seed": 42,
        "mask_degenerate_bins": False,
        "use_shard_queue": False,
        "num_params": 92,
        "num_shards": 3,
        "r2": {
            "bucket": "intermediate-data",
            "prefix_root": "data",
            "prefix": "data/test/test-20260328T120000000Z/",
        },
        "render": {
            "synth": {
                "name": "surge_simple",
                "param_spec_name": "surge_simple",
                "plugin_path": "plugins/Surge XT.vst3",
                "plugin_state_path": "presets/surge-base.vstpreset",
            },
            "renderer_version": "1.3.4",
            "renderer_backend": "pedalboard",
            "sample_rate": 44100,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "audio_dtype": "float16",
            "mel_spec_dtype": "float32",
            "samples_per_render_batch": 32,
            "samples_per_shard": 32,
            "max_retries": 0,
            "base_seed": 0,
            "sample_offset": 0,
            "attempts_per_sample": 100,
            "parallel": False,
            "plugin_reload_cadence": "render",
            "gui_toggle_cadence": "never",
            "param_sample_cadence": "sample",
        },
        "shards": [
            {
                "shard_id": i,
                "filename": f"shard-{i:06d}{ext}",
                "seed": 42 + i,
                "sample_offset": 0,
            }
            for i in range(3)
        ],
        # Computed field; mirrors DatasetSpec.split_shard_ranges output for
        # train_val_test_sizes=[32, 32, 32] and samples_per_shard=32.
        "split_shard_ranges": {
            "train": [0, 1],
            "val": [1, 2],
            "test": [2, 3],
        },
    }
    if "render" in overrides:
        # Merge nested render overrides instead of replacing the whole sub-dict.
        spec["render"] = {**spec["render"], **overrides.pop("render")}  # type: ignore[dict-item]
    spec.update(overrides)
    return spec


class TestValidateStructure:
    """Tests for validate_structure."""

    def test_valid_spec_returns_no_errors(self) -> None:
        """Valid spec with all required fields passes validation."""
        spec = _make_valid_spec()
        assert validate_structure(spec) == []

    def test_defaulted_storage_dtypes_may_be_omitted(self) -> None:
        """Specs may omit fields supplied by RenderConfig defaults."""
        spec = _make_valid_spec()
        del spec["render"]["audio_dtype"]
        del spec["render"]["mel_spec_dtype"]

        assert validate_structure(spec) == []

    def test_missing_field_returns_error(self) -> None:
        """Spec missing a required field returns a 'missing' error."""
        spec = _make_valid_spec()
        del spec["base_seed"]
        errors = validate_structure(spec)
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_invalid_git_sha_returns_error(self) -> None:
        """Non-hex git_sha returns a git_sha error."""
        spec = _make_valid_spec(git_sha="not-a-sha")
        assert any("git_sha" in e for e in validate_structure(spec))

    def test_empty_renderer_version_returns_error(self) -> None:
        """Empty render.renderer_version returns a renderer_version error."""
        spec = _make_valid_spec(render={"renderer_version": ""})
        assert any("renderer_version" in e for e in validate_structure(spec))

    def test_empty_shards_returns_error(self) -> None:
        """Empty shards list returns a shards error."""
        spec = _make_valid_spec(shards=[])
        assert any("shards" in e for e in validate_structure(spec))

    def test_unknown_output_format_returns_error(self) -> None:
        """An output_format outside the known mapping returns a structural error."""
        spec = _make_valid_spec(output_format="parquet")
        errors = validate_structure(spec)
        assert any("output_format" in e and "parquet" in e for e in errors)

    def test_missing_r2_block_returns_error(self) -> None:
        """Spec missing the top-level ``r2`` block (a DatasetSpec model field) is rejected.

        Pins the model-derived required set: ``r2`` is the nested
        ``R2Location`` field that replaced the flat ``r2_bucket`` /
        ``r2_prefix_root`` / ``r2_prefix`` keys. The structural validator
        derives the required set from the model, so adding/removing fields
        on ``DatasetSpec`` automatically tightens/loosens the check.
        """
        spec = _make_valid_spec()
        del spec["r2"]
        errors = validate_structure(spec)
        assert any("missing" in e and "r2" in e for e in errors)

    def test_required_top_level_fields_match_dataset_spec_model(self) -> None:
        """Required top-level set is derived from DatasetSpec, not hand-mirrored."""
        expected = set(DatasetSpec.model_fields) | set(DatasetSpec.model_computed_fields)
        assert set(_REQUIRED_TOP_LEVEL_FIELDS) == expected

    def test_required_render_fields_match_render_config_model(self) -> None:
        """Only backward-compatible storage fields may be omitted."""
        assert set(_REQUIRED_RENDER_FIELDS) == set(RenderConfig.model_fields) - {
            "audio_dtype",
            "mel_spec_dtype",
            # Checked shape-aware instead, so pre-nesting specs still validate.
            "synth",
        }

    def test_other_defaulted_render_field_remains_required(self) -> None:
        """Platform-dependent defaults must be materialized in persisted specs."""
        spec = _make_valid_spec()
        del spec["render"]["gui_toggle_cadence"]

        errors = validate_structure(spec)

        assert any("gui_toggle_cadence" in error for error in errors)


class TestValidateTestValues:
    """Tests for validate_test_values."""

    def test_valid_test_spec_returns_no_errors(self) -> None:
        """Spec matching generate_dataset/ci-materialize-test.yaml expectations passes."""
        spec = _make_valid_spec()
        assert validate_test_values(spec) == []
        assert all(s["filename"].endswith(".lance") for s in spec["shards"])

    def test_wrong_shard_count_returns_error(self) -> None:
        """Spec with 2 shards instead of 3 returns a shard count error."""
        spec = _make_valid_spec()
        spec["shards"] = spec["shards"][:2]
        errors = validate_test_values(spec)
        assert any("3 shards" in e for e in errors)

    def test_wrong_seeds_returns_error(self) -> None:
        """Spec with wrong seeds returns a seed error."""
        spec = _make_valid_spec()
        for shard, seed in zip(spec["shards"], (1, 2, 3), strict=True):
            shard["seed"] = seed
        errors = validate_test_values(spec)
        assert any("seed" in e for e in errors)

    def test_unknown_output_format_returns_error_not_keyerror(self) -> None:
        """Unknown output_format produces a graceful error rather than a KeyError crash."""
        spec = _make_valid_spec(output_format="parquet")
        errors = validate_test_values(spec)
        assert any("output_format" in e and "parquet" in e for e in errors)


class TestSynthIdentityShape:
    """Identity is checked shape-aware so archived specs stay validatable.

    ``validate-dataset-shards.yaml`` exposes a ``workflow_dispatch`` taking an
    arbitrary ``spec_uri``, so operators can point this validator at a spec written
    before identity was nested.
    """

    def _legacy_render(self) -> dict[str, object]:
        """Return a render mapping in the pre-nesting flat identity shape.

        :returns: A render dict whose identity keys are top-level.
        """
        spec = _make_valid_spec()
        render = dict(spec["render"])
        synth = render.pop("synth")
        assert isinstance(synth, dict)
        render["param_spec_name"] = synth["param_spec_name"]
        render["plugin_path"] = synth["plugin_path"]
        render["plugin_state_path"] = synth["plugin_state_path"]
        return render

    def test_a_pre_nesting_spec_still_validates(self) -> None:
        """A spec carrying the flat identity keys passes structural validation."""
        spec = _make_valid_spec()
        spec["render"] = self._legacy_render()

        assert validate_structure(spec) == []

    def test_a_spec_with_no_identity_at_all_is_rejected(self) -> None:
        """Neither shape present is an error, not a silently-accepted omission."""
        spec = _make_valid_spec()
        render = dict(spec["render"])
        del render["synth"]
        spec["render"] = render

        errors = validate_structure(spec)

        assert any("synth identity" in error for error in errors)

    def test_test_values_read_the_param_spec_from_the_legacy_shape(self) -> None:
        """The passthrough check reads identity from whichever shape the spec uses."""
        spec = _make_valid_spec()
        spec["render"] = self._legacy_render()

        assert not [e for e in validate_test_values(spec) if "param_spec_name" in e]

    def test_test_values_flag_a_wrong_param_spec_in_the_nested_shape(self) -> None:
        """A nested identity naming the wrong spec is still caught."""
        spec = _make_valid_spec()
        render = dict(spec["render"])
        render["synth"] = {**render["synth"], "param_spec_name": "surge_xt"}  # type: ignore[dict-item]
        spec["render"] = render

        assert any("param_spec_name" in e for e in validate_test_values(spec))
