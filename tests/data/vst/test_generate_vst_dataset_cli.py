"""CLI surface tests for ``src/data/vst/generate_vst_dataset.py``.

Pins the pydantic-settings parity guard: the CLI flag set is derived from
``RenderConfig.model_fields`` so adding/removing a field on the model
auto-extends/shrinks the CLI without a parallel code edit. These tests fail
if the CLI binding drifts from the model.
"""

from __future__ import annotations

from pipeline.schemas.spec import RenderConfig
from src.data.vst.generate_vst_dataset import _GenerateCliArgs


def test_cli_args_class_inherits_every_render_config_field() -> None:
    """``_GenerateCliArgs`` carries every ``RenderConfig`` field — flag set follows.

    Subclassing ``RenderConfig`` makes the parity structural rather than convention-
    enforced: a new field on the model surfaces as a CLI flag automatically.
    """
    cli_fields = set(_GenerateCliArgs.model_fields.keys())
    render_fields = set(RenderConfig.model_fields.keys())

    assert render_fields <= cli_fields


def test_cli_args_class_adds_only_data_file_beyond_render_config() -> None:
    """Beyond ``RenderConfig`` fields, the CLI's only extra binding is ``data_file``.

    Guards against accidental CLI bloat — if someone adds a flag here it should be a deliberate
    decision, not silent drift.
    """
    cli_fields = set(_GenerateCliArgs.model_fields.keys())
    render_fields = set(RenderConfig.model_fields.keys())

    extra = cli_fields - render_fields
    assert extra == {"data_file"}
