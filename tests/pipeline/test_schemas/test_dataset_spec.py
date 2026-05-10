"""Behavioral tests for the unified DatasetSpec model."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.data.vst import param_specs
from src.pipeline.schemas.spec import (
    DatasetSpec,
    RenderConfig,
    ShardSpec,
    dataset_config_id_from_path,
)

FIXED_NOW = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)


def _valid_render_kwargs(plugin_path: str = "/fake/Plugin.vst3") -> dict[str, Any]:
    return {
        "plugin_path": plugin_path,
        "preset_path": "presets/surge-base.vstpreset",
        "param_spec_name": "surge_simple",
        "renderer_version": "1.3.4",
        "sample_rate": 16000,
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "sample_batch_size": 32,
        "batch_per_shard": 100,
    }


def _valid_spec_kwargs(plugin_path: str = "/fake/Plugin.vst3", **overrides: Any) -> dict[str, Any]:
    """Return DatasetSpec kwargs that build a 3-shard hdf5 spec by default."""
    kwargs: dict[str, Any] = {
        "task_name": "ci-smoke-test",
        "output_format": "hdf5",
        "train_val_test_sizes": [300, 0, 0],
        "base_seed": 42,
        "r2_bucket": "intermediate-data",
        "render": _valid_render_kwargs(plugin_path),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def patch_runtime_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub git/timestamp factories so DatasetSpec construction is deterministic."""
    monkeypatch.setattr("src.pipeline.schemas.spec._get_git_sha", lambda: "abc123def456")
    monkeypatch.setattr("src.pipeline.schemas.spec._is_repo_dirty", lambda: False)
    monkeypatch.setattr("src.pipeline.schemas.spec._utc_now", lambda: FIXED_NOW)


# ---------------------------------------------------------------------------
# ShardSpec
# ---------------------------------------------------------------------------


class TestShardSpec:
    """Pin ShardSpec immutability + extra-field rejection so workers can't mutate it."""

    def test_shard_spec_is_frozen(self) -> None:
        """Mutating a constructed ShardSpec raises — workers can't drift their assignment."""
        shard = ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42)
        with pytest.raises(ValidationError):
            shard.shard_id = 99  # type: ignore[misc]

    def test_shard_spec_rejects_extra_fields(self) -> None:
        """Unknown fields raise so a renamed key in the spec doesn't silently lose data."""
        with pytest.raises(ValidationError):
            ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42, extra="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RenderConfig
# ---------------------------------------------------------------------------


class TestRenderConfig:
    """Pin RenderConfig field-level validation so bad render args fail at construction."""

    def test_render_config_rejects_extra_fields(self) -> None:
        """Unknown render-config keys raise — guards against typos in render YAML."""
        kwargs = _valid_render_kwargs()
        kwargs["surprise"] = "value"
        with pytest.raises(ValidationError):
            RenderConfig(**kwargs)

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("sample_rate", 0, "sample_rate must be positive"),
            ("channels", 0, "channels must be >= 1"),
            ("velocity", 200, r"velocity must be in \[0, 127\]"),
            ("signal_duration_seconds", 0.0, "signal_duration_seconds must be positive"),
            ("sample_batch_size", 0, "sample_batch_size must be positive"),
            ("batch_per_shard", 0, "batch_per_shard must be positive"),
            ("param_spec_name", "   ", "param_spec_name.*not in registry"),
            ("renderer_version", "", "renderer_version must not be blank"),
        ],
    )
    def test_render_config_range_validators(self, field: str, bad_value: Any, match: str) -> None:
        """Each numeric/string field's range constraint surfaces a field-specific message."""
        kwargs = _valid_render_kwargs()
        kwargs[field] = bad_value
        with pytest.raises(ValidationError, match=match):
            RenderConfig(**kwargs)

    def test_render_config_velocity_bounds_are_inclusive(self) -> None:
        """MIDI velocity boundary values (0, 127) are accepted, not rejected as off-by-one."""
        for valid in (0, 127):
            cfg = RenderConfig(**{**_valid_render_kwargs(), "velocity": valid})
            assert cfg.velocity == valid


# ---------------------------------------------------------------------------
# DatasetSpec — construction & runtime-field auto-fill
# ---------------------------------------------------------------------------


class TestDatasetSpecConstruction:
    """Pin runtime-field auto-fill: launcher-supplied values win over default factories."""

    def test_fresh_construction_fills_runtime_fields(self, patch_runtime_io: None) -> None:
        """git_sha, created_at, run_id, and r2_prefix are derived from stubbed factories."""
        spec = DatasetSpec(**_valid_spec_kwargs())

        assert spec.git_sha == "abc123def456"
        assert spec.is_repo_dirty is False
        assert spec.created_at == FIXED_NOW
        assert spec.run_id == "ci-smoke-test-20260328T120000000Z"
        assert spec.r2_prefix == "data/ci-smoke-test/ci-smoke-test-20260328T120000000Z/"

    def test_run_id_uses_explicit_value_when_present(self, patch_runtime_io: None) -> None:
        """A caller-supplied run_id overrides the timestamp-derived default."""
        spec = DatasetSpec(**_valid_spec_kwargs(run_id="custom-run-id-001"))
        assert spec.run_id == "custom-run-id-001"

    def test_r2_prefix_uses_explicit_value_when_present(self, patch_runtime_io: None) -> None:
        """A caller-supplied r2_prefix overrides the {root}/{task}/{run_id}/ default."""
        spec = DatasetSpec(**_valid_spec_kwargs(r2_prefix="custom/prefix/here/"))
        assert spec.r2_prefix == "custom/prefix/here/"

    def test_r2_prefix_root_default_is_data(self, patch_runtime_io: None) -> None:
        """Without an override, the derived r2_prefix lives under the ``data/`` root."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.r2_prefix.startswith("data/")

    def test_r2_prefix_root_custom_threads_through(self, patch_runtime_io: None) -> None:
        """Custom r2_prefix_root replaces ``data/`` in the derived prefix."""
        spec = DatasetSpec(**_valid_spec_kwargs(r2_prefix_root="experiments"))
        assert spec.r2_prefix.startswith("experiments/")

    def test_dataset_spec_strict_rejects_extra_fields(self, patch_runtime_io: None) -> None:
        """Strict mode rejects unknown top-level fields so a renamed key fails loudly."""
        kwargs = _valid_spec_kwargs(unexpected_field="surprise")
        with pytest.raises(ValidationError):
            DatasetSpec(**kwargs)


class TestDatasetSpecValidators:
    """Pin DatasetSpec field validators: blank/zero/non-UTC inputs fail at construction."""

    def test_r2_bucket_blank_raises(self, patch_runtime_io: None) -> None:
        """Empty / whitespace-only bucket names are rejected — fail loud, not at upload time."""
        for blank in ("", "   ", "\t\n"):
            with pytest.raises(ValidationError, match="r2_bucket must not be blank"):
                DatasetSpec(**_valid_spec_kwargs(r2_bucket=blank))

    def test_task_name_blank_raises(self, patch_runtime_io: None) -> None:
        """Blank task_name is rejected — empty task names produce malformed run_ids."""
        with pytest.raises(ValidationError, match="task_name must not be blank"):
            DatasetSpec(**_valid_spec_kwargs(task_name="   "))

    def test_explicit_r2_prefix_missing_trailing_slash_raises(
        self, patch_runtime_io: None
    ) -> None:
        """Explicit r2_prefix must end with ``/`` — rclone treats the dest as a directory."""
        with pytest.raises(ValidationError, match="r2_prefix must end with"):
            DatasetSpec(**_valid_spec_kwargs(r2_prefix="data/no/slash"))

    def test_split_size_not_multiple_of_batch_per_shard_raises(
        self, patch_runtime_io: None
    ) -> None:
        """Every split size must be a multiple of batch_per_shard so shards aren't ragged."""
        with pytest.raises(ValidationError, match="not a multiple"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[150, 0, 0]))

    def test_negative_split_size_raises(self, patch_runtime_io: None) -> None:
        """Negative split sizes are rejected — they would produce ill-defined shard ranges."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[-100, 0, 0]))

    def test_zero_total_split_raises(self, patch_runtime_io: None) -> None:
        """All-zero splits are rejected — a spec must produce at least one shard."""
        with pytest.raises(ValidationError, match="must sum to a positive count"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[0, 0, 0]))

    def test_invalid_output_format_literal_raises(self, patch_runtime_io: None) -> None:
        """output_format is restricted to the supported literals — typos fail at parse time."""
        with pytest.raises(ValidationError):
            DatasetSpec(**_valid_spec_kwargs(output_format="parquet"))

    def test_naive_created_at_raises(self, patch_runtime_io: None) -> None:
        """A naive datetime on ``created_at`` is rejected so run_id stays UTC-derived."""
        naive = datetime(2026, 3, 28, 12, 0, 0)
        with pytest.raises(ValidationError, match="created_at must be timezone-aware UTC"):
            DatasetSpec(**_valid_spec_kwargs(created_at=naive))

    def test_non_utc_created_at_raises(self, patch_runtime_io: None) -> None:
        """A non-UTC tz-aware datetime is rejected so run_id stays UTC-derived."""
        from datetime import timedelta

        offset_tz = timezone(timedelta(hours=5))
        with pytest.raises(ValidationError, match="created_at must be timezone-aware UTC"):
            DatasetSpec(
                **_valid_spec_kwargs(created_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=offset_tz))
            )

    def test_naive_iso_string_created_at_raises(self, patch_runtime_io: None) -> None:
        """A JSON-style naive ISO string on ``created_at`` is rejected after parsing."""
        with pytest.raises(ValidationError, match="created_at must be timezone-aware UTC"):
            DatasetSpec(**_valid_spec_kwargs(created_at="2026-03-28T12:00:00"))

    def test_train_val_test_seeds_default_is_empty(self, patch_runtime_io: None) -> None:
        """Default ``train_val_test_seeds`` is empty — per-sample seeding lands later (see
        #884)."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.train_val_test_seeds == []

    def test_populated_train_val_test_seeds_raises_not_implemented(
        self, patch_runtime_io: None
    ) -> None:
        """A non-empty ``train_val_test_seeds`` raises NotImplementedError pointing at #884."""
        with pytest.raises(NotImplementedError, match=r"issues/884"):
            DatasetSpec(**_valid_spec_kwargs(train_val_test_seeds=[1, 2, 3]))


# ---------------------------------------------------------------------------
# DatasetSpec — computed fields
# ---------------------------------------------------------------------------


class TestDatasetSpecComputedFields:
    """Pin shards/num_params derivations so workers see the same plan the launcher built."""

    def test_shards_count_matches_total_size_div_batch(self, patch_runtime_io: None) -> None:
        """num_shards == sum(splits) / batch_per_shard, not derived from anything else."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[400, 100, 100]))
        assert spec.num_shards == 6
        assert len(spec.shards) == 6

    def test_shard_seeds_are_base_plus_shard_id(self, patch_runtime_io: None) -> None:
        """Each shard's seed = base_seed + shard_id — deterministic across workers."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[300, 0, 0]))
        assert [s.seed for s in spec.shards] == [42, 43, 44]

    def test_shard_filenames_zero_padded_six_digits(self, patch_runtime_io: None) -> None:
        """Shard filenames sort lexicographically — zero-pad to six digits or rclone reorders."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[300, 0, 0]))
        assert spec.shards[0].filename == "shard-000000.h5"
        assert spec.shards[-1].filename == "shard-000002.h5"

    @pytest.mark.parametrize(("output_format", "ext"), [("hdf5", ".h5"), ("wds", ".tar")])
    def test_shard_filename_extension_matches_output_format(
        self, patch_runtime_io: None, output_format: str, ext: str
    ) -> None:
        """Shard extension is selected by output_format — the writer dispatches on it."""
        spec = DatasetSpec(**_valid_spec_kwargs(output_format=output_format))
        assert all(s.filename.endswith(ext) for s in spec.shards)

    def test_num_params_resolved_from_registry(self, patch_runtime_io: None) -> None:
        """num_params is resolved by name from the param-spec registry, not hardcoded."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        assert spec.num_params == len(param_specs["surge_simple"])

    def test_unknown_param_spec_name_raises_at_validation(self, patch_runtime_io: None) -> None:
        """Unknown param spec name raises a clean ValidationError at construction.

        Lazy KeyError from ``num_params`` would surface as an opaque stack trace
        (e.g., during ``model_dump_json``); the field validator turns it into a
        Pydantic ``ValidationError`` listing the valid registry names.
        """
        kwargs = _valid_spec_kwargs()
        kwargs["render"] = {**kwargs["render"], "param_spec_name": "nonexistent_synth"}
        with pytest.raises(ValidationError, match="not in registry"):
            DatasetSpec(**kwargs)


# ---------------------------------------------------------------------------
# DatasetSpec — JSON round-trip
# ---------------------------------------------------------------------------


class TestDatasetSpecRoundTrip:
    """Pin JSON round-trip: launcher-built spec is reproduced byte-for-byte on the worker."""

    def test_json_round_trip_preserves_runtime_fields(self, patch_runtime_io: None) -> None:
        """git_sha / created_at / run_id / r2_prefix survive a JSON round-trip unchanged."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        json_str = spec.model_dump_json()
        restored = DatasetSpec.model_validate_json(json_str)

        assert restored.git_sha == spec.git_sha
        assert restored.is_repo_dirty == spec.is_repo_dirty
        assert restored.created_at == spec.created_at
        assert restored.run_id == spec.run_id
        assert restored.r2_prefix == spec.r2_prefix

    def test_json_round_trip_preserves_shards(self, patch_runtime_io: None) -> None:
        """Computed shard list and num_shards are stable across serialize → deserialize."""
        spec = DatasetSpec(**_valid_spec_kwargs(train_val_test_sizes=[400, 100, 100]))
        restored = DatasetSpec.model_validate_json(spec.model_dump_json())
        assert restored.shards == spec.shards
        assert restored.num_shards == spec.num_shards

    def test_json_round_trip_works_without_plugin_on_disk(self, patch_runtime_io: None) -> None:
        """plugin_path is opaque text, not a Path — round-trip succeeds when no file exists."""
        spec = DatasetSpec(**_valid_spec_kwargs(plugin_path="/nonexistent/path.vst3"))
        restored = DatasetSpec.model_validate_json(spec.model_dump_json())
        assert restored.render.plugin_path == "/nonexistent/path.vst3"

    def test_json_round_trip_rebuilds_with_no_runtime_drift(
        self, patch_runtime_io: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Worker reconstructing a spec from R2 sees the same git_sha/run_id the launcher used."""
        spec = DatasetSpec(**_valid_spec_kwargs())
        # Simulate worker on a different commit; default_factory must not run for JSON-loaded.
        json_str = spec.model_dump_json()

        # Re-patch _get_git_sha to return a drift value so a re-invocation of the
        # default_factory during model_validate_json would surface as a mismatch.
        monkeypatch.setattr("src.pipeline.schemas.spec._get_git_sha", lambda: "f" * 40)

        restored = DatasetSpec.model_validate_json(json_str)

        assert restored.git_sha == "abc123def456"


# ---------------------------------------------------------------------------
# dataset_config_id_from_path
# ---------------------------------------------------------------------------


class TestDatasetConfigIdFromPath:
    """dataset_config_id_from_path returns the stem, not the full path."""

    def test_extracts_stem(self) -> None:
        """Strips parent dirs and the ``.yaml`` extension to yield the bare config id."""
        assert (
            dataset_config_id_from_path(Path("configs/experiment/surge-simple-480k-10k.yaml"))
            == "surge-simple-480k-10k"
        )


# ---------------------------------------------------------------------------
# Interpreter-only contract: spec construction + serialization must not
# transitively pull pedalboard / VST3 native deps. The launcher (and CI
# validators) build/serialize specs in lightweight processes that lack
# pedalboard's runtime requirements.
# ---------------------------------------------------------------------------


class TestSpecConstructionStaysPedalboardFree:
    """Importing schemas + building/serializing a DatasetSpec must not load pedalboard.

    Run in a fresh subprocess so the parent test session — which loads pedalboard
    transitively via ``tests/conftest.py`` — does not poison the check.
    """

    def test_render_config_validation_does_not_import_pedalboard(self) -> None:
        """Constructing RenderConfig (which runs the param_spec_name validator) must stay
        interpreter-only."""
        script = (
            "import sys\n"
            "from src.pipeline.schemas.spec import RenderConfig\n"
            "RenderConfig(\n"
            "    plugin_path='/tmp/x.vst3', preset_path='/tmp/x.vstpreset',\n"
            "    param_spec_name='surge_simple', renderer_version='v1',\n"
            "    sample_rate=16000, channels=1, velocity=64,\n"
            "    signal_duration_seconds=1.0, min_loudness=-30.0,\n"
            "    sample_batch_size=1, batch_per_shard=1,\n"
            ")\n"
            "assert 'pedalboard' not in sys.modules, sorted(\n"
            "    m for m in sys.modules if m.startswith('pedalboard')\n"
            ")\n"
        )
        result = subprocess.run(  # noqa: S603 — sys.executable + literal script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"pedalboard leaked into spec construction:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_dataset_spec_num_params_does_not_import_pedalboard(self) -> None:
        """``num_params`` is emitted by ``model_dump_json`` so its lazy import path runs in every
        serialize.

        It must remain interpreter-only.
        """
        script = (
            "import sys\n"
            "from src.pipeline.schemas.spec import DatasetSpec\n"
            "spec = DatasetSpec(\n"
            "    task_name='ci', output_format='hdf5', train_val_test_sizes=[1, 0, 0],\n"
            "    base_seed=0, r2_bucket='b',\n"
            "    render={\n"
            "        'plugin_path': '/tmp/x.vst3', 'preset_path': '/tmp/x.vstpreset',\n"
            "        'param_spec_name': 'surge_simple', 'renderer_version': 'v1',\n"
            "        'sample_rate': 16000, 'channels': 1, 'velocity': 64,\n"
            "        'signal_duration_seconds': 1.0, 'min_loudness': -30.0,\n"
            "        'sample_batch_size': 1, 'batch_per_shard': 1,\n"
            "    },\n"
            ")\n"
            "_ = spec.num_params\n"
            "_ = spec.model_dump_json()\n"
            "assert 'pedalboard' not in sys.modules, sorted(\n"
            "    m for m in sys.modules if m.startswith('pedalboard')\n"
            ")\n"
        )
        result = subprocess.run(  # noqa: S603 — sys.executable + literal script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"pedalboard leaked into spec serialization:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_bare_spec_import_does_not_pull_data_vst_core(self) -> None:
        """`import src.pipeline.schemas.spec` alone must not transitively load
        ``src.data.vst.core``.

        spec.py's two lazy imports (``_param_spec_name_must_be_registered``,
        ``num_params``) point at ``src.data.vst.param_spec_registry``, NOT
        ``src.data.vst`` (whose package ``__init__`` previously pulled
        ``src.data.vst.core`` via re-exports). If either lazy import is
        re-promoted to module level — or repointed at ``src.data.vst`` — this
        test fails immediately, preserving the launcher's interpreter-only
        contract documented in ``DatasetSpec``'s docstring.
        """
        script = (
            "import sys\n"
            "import src.pipeline.schemas.spec  # noqa: F401\n"
            "for name in ('src.data.vst.core', 'src.data.vst', 'pedalboard'):\n"
            "    assert name not in sys.modules, (\n"
            "        f'{name!r} leaked into spec module import; '\n"
            "        f'this breaks the launcher-pure invariant'\n"
            "    )\n"
        )
        result = subprocess.run(  # noqa: S603 — sys.executable + literal script
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"bare spec import is no longer launcher-pure:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
