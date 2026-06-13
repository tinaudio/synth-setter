"""Duck-typed fakes for the *introspection* surface of ``pedalboard.VST3Plugin``.

Cover ``parameters[k].type / .valid_values / .get_raw_value_for`` and
``preset_data`` — the surface ``synth_setter.data.vst.introspect`` reads;
the render surface lives in ``tests.data.vst._fake_plugin``.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

# int covers the cardinality-classification cases the tests pin; the real
# pedalboard surface reports float / str / bool.
FakeValue: TypeAlias = float | str | bool | int


class IntrospectFakeParameter:
    """Duck-type of the pedalboard parameter wrapper's introspection surface."""

    def __init__(
        self,
        type_: type,
        valid_values: Sequence[FakeValue],
        raw_values: Sequence[float] | None = None,
        name: str = "",
        range_: tuple[float | None, float | None, float | None] = (None, None, None),
    ) -> None:
        """Pair each entry of ``valid_values`` with its raw [0, 1] host value.

        :param type_: Python type pedalboard inferred (``float``/``str``/``bool``).
        :param valid_values: Values the parameter can take, as pedalboard reports them.
        :param raw_values: Raw host value for each entry of ``valid_values``; evenly
            spaced on [0, 1] when omitted.
        :param name: Display name, as ``pedalboard``'s wrapper reports it.
        :param range_: ``(min, max, step)`` tuple, as the wrapper's ``range`` reports it.
        """
        self.type = type_
        self.valid_values = valid_values
        self.name = name
        self.range = range_
        if raw_values is None:
            n = len(valid_values)
            raw_values = [i / max(n - 1, 1) for i in range(n)]
        self._raw_by_value = dict(zip(valid_values, raw_values, strict=True))

    def get_raw_value_for(self, value: FakeValue) -> float:
        """Return the raw [0, 1] host value for ``value``.

        :param value: One of ``valid_values``.
        :returns: The raw host value paired with ``value``.
        """
        return self._raw_by_value[value]


class IntrospectFakePlugin:
    """Duck-type of the ``VST3Plugin`` introspection surface (no render path)."""

    def __init__(
        self,
        parameters: dict[str, IntrospectFakeParameter],
        preset_data: bytes = b"VST3-fake-preset",
        name: str = "Fake Synth",
    ) -> None:
        """Store the introspection surface verbatim.

        :param parameters: Mapping of python-name -> fake parameter wrapper.
        :param preset_data: Bytes returned by the ``preset_data`` property.
        :param name: Plugin display name.
        """
        self.parameters = parameters
        self.preset_data = preset_data
        self.name = name

    def load_preset(self, preset_path: str) -> None:
        """Adopt the preset file's bytes as state, mirroring the real plugin.

        :param preset_path: ``.vstpreset`` file whose bytes become ``preset_data``.
        """
        self.preset_data = Path(preset_path).read_bytes()


def exec_module(source: str) -> dict[str, Any]:
    """Execute emitted spec-module source and return its namespace.

    :param source: Python source produced by ``render_param_spec_module``.
    :returns: The executed module namespace.
    """
    namespace: dict[str, Any] = {}
    exec(compile(source, "<draft_spec>", "exec"), namespace)  # noqa: S102 — source is test-built
    return namespace


def assert_ruff_format_clean(source: str) -> None:
    """Assert emitted draft source survives ``ruff format --check`` unchanged.

    Runs the venv's own ruff (``sys.executable -m ruff``) so the check cannot
    resolve to a PATH-shadowing system binary.

    :param source: Python source produced by ``render_param_spec_module``.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "--stdin-filename", "draft.py", "-"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"ruff format would change the draft:\n{result.stdout}\n{result.stderr}"
    )
