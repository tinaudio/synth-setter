"""Wire a drafted ``ParamSpec`` into a synth-setter checkout (issue #1596).

Pure text transforms and path layout behind the introspect CLI's
``--register`` mode: insert the registry entries, emit the render config, and
compute where each artifact lands in the checkout. No plugin or pedalboard
dependency — everything here operates on source text and paths.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Marker file that identifies a synth-setter checkout root (and is the file
# the registry transform rewrites).
_REGISTRY_RELPATH = Path("src/synth_setter/data/vst/param_spec_registry.py")

_IMPORT_RE = re.compile(r"^from (synth_setter\.\S+) import ")


@dataclass(frozen=True)
class RegistrationPaths:
    """Checkout destinations for one registered synth's artifacts.

    .. attribute :: spec_module

       Draft ``ParamSpec`` module beside the hand-authored Surge specs.

    .. attribute :: preset

       Captured baseline ``.vstpreset`` under ``presets/``.

    .. attribute :: csv

       Per-parameter triage table at the checkout root (``surge_params.csv``
       convention).

    .. attribute :: render_config

       Hydra render group config selecting this synth.

    .. attribute :: registry

       The registry module the transform rewrites in place.
    """

    spec_module: Path
    preset: Path
    csv: Path
    render_config: Path
    registry: Path


def registration_paths(repo_root: Path, spec_name: str) -> RegistrationPaths:
    """Compute where each artifact for ``spec_name`` lands under ``repo_root``.

    :param repo_root: Synth-setter checkout root.
    :param spec_name: Registry key for the synth.
    :returns: The five destination paths.
    """
    return RegistrationPaths(
        spec_module=repo_root / "src/synth_setter/data/vst" / f"{spec_name}_param_spec.py",
        preset=repo_root / preset_repo_path(spec_name),
        csv=repo_root / f"{spec_name}_params.csv",
        render_config=repo_root / "src/synth_setter/configs/render" / f"{spec_name}.yaml",
        registry=repo_root / _REGISTRY_RELPATH,
    )


def preset_repo_path(spec_name: str) -> str:
    """Return the checkout-relative baseline-preset path for ``spec_name``.

    Single source for the path recorded in both ``preset_paths`` and the
    render config, which must agree.

    :param spec_name: Registry key for the synth.
    :returns: ``presets/<spec_name>-base.vstpreset``.
    """
    return f"presets/{spec_name}-base.vstpreset"


def is_checkout_root(path: Path) -> bool:
    """Report whether ``path`` is a synth-setter checkout root.

    :param path: Candidate directory.
    :returns: ``True`` when the registry module exists under ``path``.
    """
    return (path / _REGISTRY_RELPATH).is_file()


def find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` to the enclosing synth-setter checkout root.

    :param start: Directory to start from (typically the cwd).
    :returns: The first ancestor containing the registry module, or ``None``.
    """
    for candidate in (start, *start.parents):
        if is_checkout_root(candidate):
            return candidate
    return None


def registry_with_spec(source: str, spec_name: str) -> str:
    """Return ``source`` with ``spec_name`` registered in both registry dicts.

    Inserts the generated module's import (in sorted position, so ruff's I001
    stays clean) plus one entry each in ``param_specs`` and ``preset_paths``.
    Re-applying an identical registration is a no-op so ``--force`` re-runs
    converge instead of erroring.

    :param source: Current ``param_spec_registry.py`` source text.
    :param spec_name: Registry key; also derives the module/constant names.
    :returns: The modified registry source.
    :raises ValueError: ``spec_name`` is already registered with different
        wiring, or ``source`` lacks the registry's import/dict anchors.
    """
    module = f"synth_setter.data.vst.{spec_name}_param_spec"
    import_line = f"from {module} import {spec_name.upper()}_PARAM_SPEC"
    spec_entry = f'    "{spec_name}": {spec_name.upper()}_PARAM_SPEC,'
    preset_entry = f'    "{spec_name}": "{preset_repo_path(spec_name)}",'

    lines = source.splitlines()
    if f'"{spec_name}":' in source:
        if all(line in lines for line in (import_line, spec_entry, preset_entry)):
            return source
        raise ValueError(
            f"{spec_name!r} is already registered in param_spec_registry with different "
            "wiring; pick another --spec-name or remove the existing entries first."
        )

    lines.insert(_import_insert_index(lines, module), import_line)
    _insert_dict_entry(lines, "param_specs", spec_entry)
    _insert_dict_entry(lines, "preset_paths", preset_entry)
    return "\n".join(lines) + "\n"


def _import_insert_index(lines: list[str], module: str) -> int:
    """Find the sorted insertion index for ``from <module> import ...``.

    :param lines: Registry source lines.
    :param module: Dotted module path of the new import.
    :returns: Index at which to insert the new import line.
    :raises ValueError: No first-party import block exists to anchor on.
    """
    block: list[tuple[str, int, int]] = []  # (module, start, end-exclusive)
    for i, line in enumerate(lines):
        match = _IMPORT_RE.match(line)
        if not match:
            continue
        end = i + 1
        if line.rstrip().endswith("("):
            while end < len(lines) and lines[end].strip() != ")":
                end += 1
            end += 1
        block.append((match.group(1), i, end))
    if not block:
        raise ValueError(
            "param_spec_registry source has no 'from synth_setter…' import block to anchor on"
        )
    for existing_module, start, _ in block:
        if existing_module > module:
            return start
    return block[-1][2]


def _insert_dict_entry(lines: list[str], dict_name: str, entry: str) -> None:
    """Insert ``entry`` before the closing brace of module-level dict ``dict_name``.

    :param lines: Registry source lines, mutated in place.
    :param dict_name: Name of the dict assignment to extend.
    :param entry: Pre-indented ``"key": value,`` line.
    :raises ValueError: The dict's ``<name>… = {`` / ``}`` anchors are missing.
    """
    opener = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith(f"{dict_name}:") and line.rstrip().endswith("{")
        ),
        None,
    )
    if opener is not None:
        for i in range(opener + 1, len(lines)):
            if lines[i] == "}":
                lines.insert(i, entry)
                return
    raise ValueError(
        f"param_spec_registry source has no module-level dict {dict_name!r} to extend"
    )


def checkout_relative_path(plugin_path: str, root: Path) -> str:
    """Record ``plugin_path`` relative to the checkout when it lives inside it.

    Render workers resolve relative plugin paths against the checkout root, so
    an in-checkout plugin (the ``plugins/`` convention) stays portable; one
    outside it is pinned absolute.

    :param plugin_path: Plugin path as given on the CLI.
    :param root: Checkout root.
    :returns: Checkout-relative POSIX path, or the resolved absolute path.
    """
    given = Path(plugin_path)
    # Absolutize without dereferencing the final component: the plugins/
    # convention symlinks into the system VST3 dir outside the checkout, so
    # Path.resolve would escape the tree and force a host-specific path.
    absolute = given if given.is_absolute() else Path.cwd() / given
    for base in (root, root.resolve()):
        try:
            return absolute.relative_to(base).as_posix()
        except ValueError:
            continue
    return str(absolute.resolve())


def render_config_yaml(spec_name: str, *, plugin_path: str, renderer_version: str) -> str:
    """Emit the Hydra render group config selecting ``spec_name``.

    Generic render knobs (sample rate, shard sizing, cadences) inherit from
    the ``surge_xt`` group config — the same pattern as ``surge_simple.yaml``
    — while the synth's identity fields are pinned here. Paths and version are
    double-quoted via ``json.dumps`` (a subset of YAML's double-quote style)
    so arbitrary plugin paths cannot break the scalar.

    :param spec_name: Registry key; names the param spec and preset.
    :param plugin_path: ``.vst3`` path recorded for render workers.
    :param renderer_version: Plugin version pin checked before each render.
    :returns: YAML text for ``configs/render/<spec_name>.yaml``.
    """
    return "\n".join(
        [
            f"# Draft render config for {spec_name} — generated by",
            "# synth-setter-introspect-plugin. Generic render knobs inherit from the",
            "# surge_xt group config; the fields below pin this synth's identity.",
            "defaults:",
            "  - surge_xt",
            "",
            f"plugin_path: {json.dumps(plugin_path)}",
            f"preset_path: {json.dumps(preset_repo_path(spec_name))}",
            f"param_spec_name: {spec_name}",
            f"renderer_version: {json.dumps(renderer_version)}",
            "",
        ]
    )
