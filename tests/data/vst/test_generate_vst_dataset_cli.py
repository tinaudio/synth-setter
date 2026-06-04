"""CLI surface tests for ``src/synth_setter/data/vst/generate_vst_dataset.py``.

Pins the pydantic-settings parity guard: the CLI flag set is derived from
``RenderConfig.model_fields`` so adding/removing a field on the model
auto-extends/shrinks the CLI without a parallel code edit. These tests fail
if the CLI binding drifts from the model.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import CliApp

from synth_setter.cli.generate_dataset import build_generate_args
from synth_setter.data.vst.generate_vst_dataset import _GenerateCliArgs
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat, RenderConfig


def test_cli_args_class_inherits_every_render_config_field() -> None:
    """``_GenerateCliArgs`` carries every ``RenderConfig`` field — flag set follows.

    Subclassing ``RenderConfig`` makes the parity structural rather than convention-
    enforced: a new field on the model surfaces as a CLI flag automatically.
    """
    cli_fields = set(_GenerateCliArgs.model_fields.keys())
    render_fields = set(RenderConfig.model_fields.keys())

    assert render_fields <= cli_fields


def test_cli_args_class_adds_only_data_file_and_copy_dataset_root_beyond_render_config() -> None:
    """Beyond ``RenderConfig`` fields, the CLI's only extras are ``data_file`` +
    ``copy_dataset_root``.

    Guards against accidental CLI bloat — adding a flag here should be a deliberate decision,
    not silent drift. ``copy_dataset_root`` is the deliberate dataset-copy opt-in.
    """
    cli_fields = set(_GenerateCliArgs.model_fields.keys())
    render_fields = set(RenderConfig.model_fields.keys())

    extra = cli_fields - render_fields
    assert extra == {"data_file", "copy_dataset_root"}


def _smoke_spec() -> DatasetSpec:
    """A minimal ``DatasetSpec`` for round-trip tests — no I/O, no plugin required."""
    render_cfg = RenderConfig(
        plugin_path="plugins/Surge XT.vst3",
        preset_path="presets/surge-base.vstpreset",
        param_spec_name="surge_simple",
        renderer_version="1.3.4",
        sample_rate=44100,
        channels=2,
        velocity=100,
        signal_duration_seconds=4.0,
        min_loudness=-55.0,
        samples_per_render_batch=32,
        samples_per_shard=10000,
        # Darwin-portable (#714).
        gui_toggle_cadence="never",
    )
    return DatasetSpec(
        task_name="ci-smoke-test",
        output_format=OutputFormat.HDF5,
        train_val_test_sizes=(440000, 20000, 20000),
        base_seed=42,
        r2={"bucket": "intermediate-data"},  # type: ignore[arg-type]
        render=render_cfg,
    )


def test_build_generate_args_roundtrips_through_cli_parser() -> None:
    """Args emitted by ``build_generate_args`` parse back into the same ``RenderConfig``.

    Pins the full producer↔consumer contract: a divergence in flag spelling (kebab vs.
    underscore), value coercion (int vs. float), or ``extra="forbid"`` rejection would
    break this round-trip even when the field-set parity tests still pass.
    """
    spec = _smoke_spec()
    args = build_generate_args(spec, spec.shards[0], Path("/tmp"))

    parsed = CliApp.run(_GenerateCliArgs, cli_args=args[2:])
    reconstructed = RenderConfig(**parsed.model_dump(exclude={"data_file", "copy_dataset_root"}))

    assert reconstructed == spec.render
    assert parsed.data_file == "/tmp/shard-000000.h5"
    # No copy source on this spec, so the CLI flag is absent and parses to None.
    assert parsed.copy_dataset_root is None
