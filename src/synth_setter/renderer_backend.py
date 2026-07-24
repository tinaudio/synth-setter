"""Shared renderer modes and plugin-path sentinel for config and runtime checks.

Interpreter-only (like ``param_spec_name``) so the launcher-pure
``pipeline.schemas.spec`` and the render-worker modules can share definitions
without pulling ``synth_setter.data.vst`` at import time.
"""

from typing import Literal

PluginProcessResetMode = Literal["reset", "preserve"]
RendererBackend = Literal["pedalboard", "dawdreamer", "torchsynth"]

# ``RenderConfig.plugin_path`` value that selects the in-process backend in
# place of a plugin-bundle path (see ``core.extract_renderer_version``).
TORCHSYNTH_PLUGIN_NAME = "torchsynth"
