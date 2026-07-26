"""Shared renderer modes and plugin-path sentinels for config and runtime checks.

Interpreter-only (like ``param_spec_name``) so the launcher-pure
``pipeline.schemas.spec`` and the render-worker modules can share definitions
without pulling ``synth_setter.data.vst`` at import time.
"""

from typing import Literal

type PluginProcessResetMode = Literal["reset", "preserve"]
type RendererBackend = Literal["pedalboard", "dawdreamer", "dawdreamer_faust", "torchsynth"]

# ``RenderConfig.plugin_path`` value that selects the in-process backend in
# place of a plugin-bundle path (see ``core.extract_renderer_version``).
TORCHSYNTH_PLUGIN_NAME = "torchsynth"
FAUST_PLUGIN_NAME = "faust"
