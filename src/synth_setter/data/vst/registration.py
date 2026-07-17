"""Wire a drafted ``ParamSpec`` into a synth-setter checkout (issue #1596).

Pure text transforms and path layout behind the introspect CLI's
``--register`` mode: insert the registry entries, emit the render config, and
compute where each artifact lands in the checkout. No plugin or pedalboard
dependency — everything here operates on source text and paths.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

# Marker file that identifies a synth-setter checkout root (and is the file
# the registry transform rewrites).
_REGISTRY_RELPATH = Path("src/synth_setter/data/vst/param_spec_registry.py")
# Registration writes configs/render; protect shipped groups from being overwritten.
_RESERVED_RENDER_CONFIG_NAMES = frozenset(
    {
        "obxf",
        "surge_4",
        "surge_simple",
        "surge_xt",
        "torchsynth_adsr",
        "torchsynth_full",
        "torchsynth_simple",
        "vst",
    }
)

_IMPORT_RE = re.compile(r"^from (synth_setter\.\S+) import ")


def _is_reserved_render_config_name(spec_name: str) -> bool:
    """Return whether ``spec_name`` is reserved for a shared render config.

    :param spec_name: Candidate synth registry key.
    :returns: ``True`` when the name collides with a shared render config.
    """
    return spec_name.casefold() in _RESERVED_RENDER_CONFIG_NAMES


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
    :raises ValueError: If the name is reserved for a shared render config.
    """
    if _is_reserved_render_config_name(spec_name):
        raise ValueError(f"{spec_name!r} is reserved for a render config")
    return RegistrationPaths(
        spec_module=repo_root / "src/synth_setter/data/vst" / f"{spec_name}_param_spec.py",
        preset=repo_root / preset_repo_path(spec_name),
        csv=repo_root / f"{spec_name}_params.csv",
        render_config=repo_root / "src/synth_setter/configs/render" / f"{spec_name}.yaml",
        registry=repo_root / _REGISTRY_RELPATH,
    )


def preset_repo_path(spec_name: str) -> str:
    """Return the checkout-relative baseline-preset path for ``spec_name``.

    Single source for the path recorded in both ``plugin_state_paths`` and the
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


def _registry_key_value(key: ast.expr | None) -> str | None:
    """Return the string identity of a registry dictionary key.

    :param key: Parsed dictionary key expression.
    :return: Plain or ``ParamSpecName``-wrapped string value, otherwise ``None``.
    """
    if (
        isinstance(key, ast.Call)
        and isinstance(key.func, ast.Name)
        and key.func.id == "ParamSpecName"
        and len(key.args) == 1
    ):
        key = key.args[0]
    return key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None


def _dict_key_lines(
    tree: ast.Module, *, lines: list[str], dict_name: str, key_value: str
) -> list[str]:
    """Return source lines defining one logical key in a module-level dict.

    :param tree: Parsed registry module.
    :param lines: Registry source split into lines.
    :param dict_name: Module-level dictionary variable to inspect.
    :param key_value: String key identity to find.
    :return: Source lines whose key evaluates to ``key_value``.
    """
    matches: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == dict_name for target in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if key is not None and _registry_key_value(key) == key_value:
                matches.append(lines[key.lineno - 1])
    return matches


def _module_scope_imports(node: ast.AST) -> list[ast.ImportFrom]:
    """Collect imports that can bind names in the module namespace.

    :param node: Parsed module or control-flow node sharing module scope.
    :return: Direct and conditional imports, excluding function and class namespaces.
    """
    if isinstance(node, ast.ImportFrom):
        return [node]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []
    imports: list[ast.ImportFrom] = []
    for child in ast.iter_child_nodes(node):
        imports.extend(_module_scope_imports(child))
    return imports


def registry_with_spec(source: str, spec_name: str) -> str:
    """Return ``source`` with ``spec_name`` registered in both registry dicts.

    Inserts the generated module's import (in sorted position, so ruff's I001
    stays clean) plus one entry each in ``_param_specs`` and ``plugin_state_paths``.
    Re-applying an identical registration is a no-op so ``--force`` re-runs
    converge instead of erroring.

    :param source: Current ``param_spec_registry.py`` source text.
    :param spec_name: Registry key; also derives the module/constant names.
    :returns: The modified registry source.
    :raises ValueError: ``spec_name`` is already registered with different
        wiring, or ``source`` lacks the registry's import/dict anchors.
    """
    module = f"synth_setter.data.vst.{spec_name}_param_spec"
    constant = f"{spec_name.upper()}_PARAM_SPEC"
    import_line = f"from {module} import {constant}"
    spec_entry = f'    ParamSpecName("{spec_name}"): {constant},'
    preset_entry = f'    "{spec_name}": "{preset_repo_path(spec_name)}",'

    lines = source.splitlines()
    tree = ast.parse(source)
    import_lines = [
        lines[node.lineno - 1]
        for node in _module_scope_imports(tree)
        if (
            node.module == module
            or any((alias.asname or alias.name) == constant for alias in node.names)
        )
    ]
    logical_wiring = (
        import_lines,
        _dict_key_lines(tree, lines=lines, dict_name="_param_specs", key_value=spec_name),
        _dict_key_lines(
            tree,
            lines=lines,
            dict_name="plugin_state_paths",
            key_value=spec_name,
        ),
    )
    if any(logical_wiring):
        if all(
            matches == [expected]
            for matches, expected in zip(
                logical_wiring, (import_line, spec_entry, preset_entry), strict=True
            )
        ):
            return source
        raise ValueError(
            f"{spec_name!r} is already registered in param_spec_registry with different "
            "wiring; pick another --spec-name or remove the existing entries first."
        )

    lines.insert(_import_insert_index(lines, module), import_line)
    _insert_dict_entry(lines, "_param_specs", spec_entry)
    _insert_dict_entry(lines, "plugin_state_paths", preset_entry)
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
    the ``vst`` group config, while this config pins the synth's identity.
    Every identity scalar is double-quoted via ``json.dumps`` (a subset of
    YAML's double-quote style) so arbitrary plugin paths cannot break the scalar.
    A spec name that is a YAML 1.1 literal (``on``, ``true``) stays a string.

    :param spec_name: Registry key; names the param spec and preset.
    :param plugin_path: ``.vst3`` path recorded for render workers.
    :param renderer_version: Plugin version pin checked before each render.
    :returns: YAML text for ``configs/render/<spec_name>.yaml``.
    :raises ValueError: If the name is reserved for a shared render config.
    """
    if _is_reserved_render_config_name(spec_name):
        raise ValueError(f"{spec_name!r} is reserved for a render config")
    return "\n".join(
        [
            "# Generated by synth-setter-introspect-plugin; generic VST knobs inherit from vst.",
            "defaults:",
            "  - vst",
            "",
            f"plugin_path: {json.dumps(plugin_path)}",
            f"plugin_state_path: {json.dumps(preset_repo_path(spec_name))}",
            f"param_spec_name: {json.dumps(spec_name)}",
            f"renderer_version: {json.dumps(renderer_version)}",
            "",
        ]
    )
