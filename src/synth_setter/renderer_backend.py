"""Shared backend selector consumed by the render config and runtime checks.

Interpreter-only (like ``param_spec_name``) so the launcher-pure
``pipeline.schemas.spec`` and the render-worker modules can share one
definition without pulling ``synth_setter.data.vst`` at import time.
"""

from typing import Literal

RendererBackend = Literal["pedalboard", "dawdreamer", "torchsynth"]
