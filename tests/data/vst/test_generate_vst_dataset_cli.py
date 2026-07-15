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
from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline.schemas.spec import DatasetSpec, OutputFormat, RenderConfig


def test_cli_args_class_inherits_every_render_config_field() -> None:
    """``_GenerateCliArgs`` carries every ``RenderConfig`` field — flag set follows.

    Subclassing ``RenderConfig`` makes the parity structural rather than convention-
    enforced: a new field on the model surfaces as a CLI flag automatically.
    """
    cli_fields = set(_GenerateCliArgs.model_fields.keys())
    render_fields = set(RenderConfig.model_fields.keys())

    assert render_fields <= cli_fields


def test_cli_args_class_adds_only_data_file_beyond_render_config() -> None:
    """Beyond ``RenderConfig`` fields, the CLI's only extra is ``data_file``.

    Guards against accidental CLI bloat — adding a flag here should be a deliberate decision, not
    silent drift.
    """
    cli_fields = set(_GenerateCliArgs.model_fields.keys())
    render_fields = set(RenderConfig.model_fields.keys())

    extra = cli_fields - render_fields
    assert extra == {"data_file"}


def _smoke_spec() -> DatasetSpec:
    """Build a minimal ``DatasetSpec`` for round-trip tests — no I/O, no plugin required."""
    render_cfg = RenderConfig(
        plugin_path="plugins/Surge XT.vst3",
        plugin_state_path="presets/surge-base.vstpreset",
        param_spec_name=ParamSpecName("surge_simple"),
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
        output_format=OutputFormat.LANCE,
        train_val_test_sizes=(440000, 20000, 20000),
        base_seed=42,
        r2={"bucket": "intermediate-data"},  # type: ignore[arg-type]
        render=render_cfg,
    )


def test_build_generate_args_roundtrips_through_cli_parser(tmp_path: Path) -> None:
    """Args emitted by ``build_generate_args`` parse back into the same ``RenderConfig``.

    Pins the full producer↔consumer contract: a divergence in flag spelling (kebab vs.
    underscore), value coercion (int vs. float), or ``extra="forbid"`` rejection would
    break this round-trip even when the field-set parity tests still pass.

    :param tmp_path: Output directory the shard path is built under.
    """
    spec = _smoke_spec()
    args = build_generate_args(spec, spec.shards[0], tmp_path)

    parsed = CliApp.run(_GenerateCliArgs, cli_args=args[2:])
    reconstructed = RenderConfig(**parsed.model_dump(exclude={"data_file"}))

    # build_generate_args overrides base_seed with the shard's seed (#884), so the
    # round-tripped config matches spec.render with that one field substituted.
    expected = spec.render.model_copy(update={"base_seed": spec.shards[0].seed})
    assert reconstructed == expected
    assert parsed.data_file == str(tmp_path / "shard-000000.lance")
