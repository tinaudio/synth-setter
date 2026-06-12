"""Draft a ``ParamSpec`` scaffold from any loaded VST3 plugin (issue #1596).

Pedalboard exposes per-parameter metadata (``type``, ``valid_values``,
``get_raw_value_for``) that classifies most parameters automatically; the
draft is a starting point to hand-tune, not a finished spec.
"""

from __future__ import annotations

import csv
import io
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

from synth_setter.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    Parameter,
)

# The value types pedalboard commonly infers for VST3 parameters; anything
# else falls through to the continuous draft in ``_classify``.
ParameterValue: TypeAlias = float | str | bool


class IntrospectableParameter(Protocol):
    """Structural type for the pedalboard parameter surface this module reads.

    Pedalboard builds parameter wrappers dynamically, so its stubs carry no usable attribute types;
    this protocol pins the read contract instead.
    """

    @property
    def type(self) -> type:
        """Python type pedalboard inferred (``float`` / ``str`` / ``bool``)."""
        ...

    @property
    def name(self) -> str:
        """Display name, as the host reports it."""
        ...

    @property
    def range(self) -> tuple[float | None, float | None, float | None]:
        """``(min, max, step)`` in display units; ``None`` entries when unreported."""
        ...

    @property
    def valid_values(self) -> Sequence[ParameterValue]:
        """Values the parameter can take, as pedalboard reports them."""
        ...

    def get_raw_value_for(self, value: ParameterValue) -> float:
        """Return the raw [0, 1] host value for ``value``.

        :param value: One of ``valid_values``.
        :returns: The raw host value paired with ``value``.
        """
        ...


class IntrospectablePlugin(Protocol):
    """Structural type for the ``VST3Plugin`` surface this module reads."""

    @property
    def name(self) -> str:
        """Plugin display name."""
        ...

    @property
    def parameters(self) -> Mapping[str, IntrospectableParameter]:
        """Mapping of python-name -> parameter wrapper."""
        ...

    @property
    def preset_data(self) -> bytes:
        """Current plugin state in ``.vstpreset`` format."""
        ...


# Note-conditioning defaults copied from the Surge specs; every spec needs a
# pitch + note_start_and_end pair and these ranges are sane for most synths.
_DEFAULT_PITCH_MIN = 48
_DEFAULT_PITCH_MAX = 72
_DEFAULT_MAX_NOTE_DURATION_SECONDS = 4.0
# Mirrors [tool.ruff] line-length; drift is caught by the emitted-draft
# format-clean tests (test_rendered_module_is_ruff_format_clean).
_MAX_LINE_LENGTH = 99


@dataclass(frozen=True)
class SkippedParameter:
    """A plugin parameter left out of the draft, with the reason for the human tuner.

    .. attribute :: name

       Python-name key of the parameter in ``plugin.parameters``.

    .. attribute :: reason

       Why the parameter was skipped (degenerate range, metadata error, ...).
    """

    name: str
    reason: str


def draft_synth_params(
    plugin: IntrospectablePlugin,
) -> tuple[list[Parameter], list[SkippedParameter]]:
    """Classify every plugin parameter into a draft ``Parameter``, in plugin order.

    ``str``-typed parameters become onehot ``CategoricalParameter``s with raw
    values asked from the host; ``bool``-typed become two-value categoricals;
    every other type (pedalboard reports ``float``) becomes a full-range
    ``ContinuousParameter``. Parameters with fewer than two valid values carry
    no signal and are skipped, as is any parameter whose metadata lookup raises.

    :param plugin: A loaded plugin exposing the pedalboard parameter surface.
    :returns: ``(drafted, skipped)`` — drafted parameters in plugin order, and
        the skipped parameters with reasons.
    """
    drafted: list[Parameter] = []
    skipped: list[SkippedParameter] = []
    for name, param in plugin.parameters.items():
        # Broad catch: one bad parameter must not kill the whole draft.
        try:
            n_valid = len(param.valid_values)
            if n_valid < 2:
                skipped.append(SkippedParameter(name, f"degenerate: {n_valid} valid value(s)"))
                continue
            drafted.append(_classify(name, param))
        except Exception as exc:
            skipped.append(SkippedParameter(name, f"metadata error: {exc}"))
    return drafted, skipped


def _classify(name: str, param: IntrospectableParameter) -> Parameter:
    """Build the draft ``Parameter`` for one plugin parameter.

    :param name: Python-name key in ``plugin.parameters``.
    :param param: Pedalboard parameter wrapper (``type`` / ``valid_values`` /
        ``get_raw_value_for``).
    :returns: The drafted parameter.
    """
    if param.type in (str, bool):
        values = list(param.valid_values)
        raw_values = [float(param.get_raw_value_for(v)) for v in values]
        return CategoricalParameter(
            name=name, values=values, raw_values=raw_values, encoding="onehot"
        )
    return ContinuousParameter(name=name, min=0.0, max=1.0)


def render_param_spec_module(
    spec_name: str,
    *,
    plugin_name: str,
    params: Sequence[Parameter],
    skipped: Sequence[SkippedParameter],
    provenance: str | None = None,
) -> str:
    """Emit an editable, ruff-format-clean module defining the draft ``ParamSpec``.

    The module declares ``<SPEC_NAME>_PARAM_SPEC`` with the drafted synth
    parameters plus default note-conditioning parameters, and lists every
    skipped parameter as a comment so the human tuner can resurrect it. The
    output is emitted in the formatter's own style (double quotes, magic
    trailing commas, repo line length) so committing it does not trip pre-commit.

    :param spec_name: Registry key for the synth; uppercased for the module-level
        constant name (``my_synth`` -> ``MY_SYNTH_PARAM_SPEC``).
    :param plugin_name: Plugin display name, recorded in the module docstring.
    :param params: Drafted synth parameters, in plugin order.
    :param skipped: Parameters left out of the draft, echoed as comments.
    :param provenance: Optional one-line source description (plugin path,
        version, starting preset) recorded as a comment under the docstring.
    :returns: Python source for the draft spec module.
    """
    lines = _render_header(plugin_name, provenance)
    lines.extend(_render_imports(params))
    lines.append(f"{spec_name.upper()}_PARAM_SPEC = ParamSpec(")
    if params or skipped:
        lines.append("    [")
        for param in params:
            lines.extend(_render_parameter(param))
        for skip in skipped:
            lines.extend(_comment_lines(f"skipped {skip.name}: {skip.reason}", indent=" " * 8))
        lines.append("    ],")
    else:
        # ruff-format collapses empty brackets, so emit the collapsed form.
        lines.append("    [],")
    pitch = (
        f'DiscreteLiteralParameter(name="pitch", '
        f"min={_DEFAULT_PITCH_MIN}, max={_DEFAULT_PITCH_MAX})"
    )
    lines.extend(
        [
            "    [",
            f"        {pitch},",
            "        NoteDurationParameter(",
            '            name="note_start_and_end",',
            f"            max_note_duration_seconds={_DEFAULT_MAX_NOTE_DURATION_SECONDS},",
            "        ),",
            "    ],",
            ")",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_header(plugin_name: str, provenance: str | None) -> list[str]:
    """Render the emitted module's docstring and optional provenance comment.

    :param plugin_name: Plugin display name; sanitized so it cannot terminate the docstring or form
        an escape sequence.
    :param provenance: Optional source description, emitted as a comment.
    :returns: Source lines for the top of the draft module.
    """
    safe_name = _comment_safe(plugin_name).replace("\\", "/").replace('"', "'")
    lines = [
        f'"""Draft ParamSpec for {safe_name} — generated by synth-setter-introspect-plugin.',
        "",
        "Hand-tune before use: narrow continuous ranges, prune parameters that",
        "don't shape the sound, and weight categorical values. Then register the",
        "spec in synth_setter.data.vst.param_spec_registry.",
        '"""',
    ]
    if provenance is not None:
        lines.extend(["", *_comment_lines(f"Source: {provenance}", indent="")])
    return lines


def _render_imports(params: Sequence[Parameter]) -> list[str]:
    """Render the draft module's import block.

    Emits only the parameter-class imports the body uses — an unused import in a committed draft
    would fail the repo's ruff F401 check.

    :param params: Drafted synth parameters.
    :returns: Source lines for the import block, with surrounding blank lines.
    """
    lines = ["", "from synth_setter.data.vst.param_spec import ("]
    if any(isinstance(p, CategoricalParameter) for p in params):
        lines.append("    CategoricalParameter,")
    if any(isinstance(p, ContinuousParameter) for p in params):
        lines.append("    ContinuousParameter,")
    lines.extend(
        [
            "    DiscreteLiteralParameter,",
            "    NoteDurationParameter,",
            "    ParamSpec,",
            ")",
            "",
        ]
    )
    return lines


def _comment_safe(text: str) -> str:
    """Collapse whitespace so interpolated text cannot break out of one source line.

    Plugin exception messages may span lines; uncommented continuation lines
    would make the emitted module a syntax error.

    :param text: Free-form text destined for a single emitted source line.
    :returns: ``text`` with all whitespace runs collapsed to single spaces.
    """
    return " ".join(text.split())


def _comment_lines(text: str, indent: str) -> list[str]:
    """Render ``text`` as ``#`` comment lines wrapped to ``_MAX_LINE_LENGTH``.

    :param text: Free-form text; whitespace (including newlines) is collapsed.
    :param indent: Leading whitespace for each emitted line.
    :returns: One or more ``#``-prefixed source lines.
    """
    width = _MAX_LINE_LENGTH - len(indent) - len("# ")
    return [f"{indent}# {chunk}" for chunk in textwrap.wrap(_comment_safe(text), width)]


def _py_literal(value: object) -> str:
    """Render ``value`` as a Python literal in ruff-format's preferred style.

    ``repr`` alone is not format-stable: it prefers single quotes, while the
    formatter rewrites plain strings to double quotes.

    :param value: A parameter value (``str`` / ``bool`` / ``float``-like).
    :returns: Source text for the literal.
    """
    if not isinstance(value, str):
        return repr(value)
    quoted = repr(value)
    if quoted.startswith('"'):
        return quoted  # value holds single quotes only; repr already double-quotes
    if '"' in value:
        if "'" not in value:
            return quoted  # single-quoting avoids escapes, so the formatter keeps it
        # Both quote types: the formatter tie-breaks to double quotes. The rewrite
        # is safe — repr escaped every literal backslash, so each \' is a quote.
        body = quoted[1:-1].replace("\\'", "'").replace('"', '\\"')
        return f'"{body}"'
    return f'"{quoted[1:-1]}"'


def _render_parameter(param: Parameter) -> list[str]:
    """Render one drafted parameter as source lines for the emitted module.

    :param param: A parameter produced by ``draft_synth_params``.
    :returns: Source lines, indented for the ``ParamSpec`` synth-params list.
    :raises TypeError: ``param`` is not a draftable parameter type.
    """
    if isinstance(param, ContinuousParameter):
        single = (
            f"        ContinuousParameter(name={_py_literal(param.name)}, "
            f"min={param.min}, max={param.max}),"
        )
        if len(single) <= _MAX_LINE_LENGTH:
            return [single]
        return [
            "        ContinuousParameter(",
            f"            name={_py_literal(param.name)},",
            f"            min={param.min},",
            f"            max={param.max},",
            "        ),",
        ]
    if isinstance(param, CategoricalParameter):
        return [
            "        CategoricalParameter(",
            f"            name={_py_literal(param.name)},",
            "            values=[",
            *(f"                {_py_literal(v)}," for v in param.values),
            "            ],",
            "            raw_values=[",
            *(f"                {v!r}," for v in param.raw_values),
            "            ],",
            f"            encoding={_py_literal(param.encoding)},",
            "        ),",
        ]
    raise TypeError(f"cannot render draft parameter of type {type(param).__name__}")


def capture_preset(plugin: IntrospectablePlugin, out_path: Path) -> None:
    """Write the plugin's current state to ``out_path`` as a ``.vstpreset``.

    :param plugin: A loaded plugin; its ``preset_data`` is the captured state.
    :param out_path: Destination file; parent directories must exist.
    """
    out_path.write_bytes(plugin.preset_data)


def render_param_table_csv(
    plugin: IntrospectablePlugin,
    params: Sequence[Parameter],
    skipped: Sequence[SkippedParameter],
) -> str:
    """Render every plugin parameter as a CSV row with its draft outcome.

    Columns follow ``surge_params.csv`` (index, ``pyname``, display ``name``,
    ``range``) plus ``drafted_as`` (the drafted ``Parameter`` subclass, empty if
    skipped) and ``skipped_reason`` (empty if drafted) — a triage sheet for
    hand-tuning the draft spec.

    :param plugin: The introspected plugin; rows follow its parameter order.
    :param params: Drafted parameters from ``draft_synth_params``.
    :param skipped: Skipped parameters from ``draft_synth_params``.
    :returns: CSV text including the header row.
    """
    drafted_as = {p.name: type(p).__name__ for p in params}
    skip_reason = {s.name: s.reason for s in skipped}
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["", "pyname", "name", "range", "drafted_as", "skipped_reason"])
    for index, (pyname, param) in enumerate(plugin.parameters.items()):
        writer.writerow(
            [
                index,
                pyname,
                _read_or_blank(lambda p=param: p.name),
                _read_or_blank(lambda p=param: str(p.range)),
                drafted_as.get(pyname, ""),
                skip_reason.get(pyname, ""),
            ]
        )
    return buffer.getvalue()


def _read_or_blank(getter: Callable[[], str]) -> str:
    """Return ``getter()``, or ``""`` when the host metadata read raises.

    :param getter: Zero-arg metadata accessor on a parameter wrapper.
    :returns: The metadata value, or an empty cell on any error.
    """
    # Broad catch: one unreadable metadata field must not lose the whole table.
    try:
        return getter()
    except Exception:
        return ""
