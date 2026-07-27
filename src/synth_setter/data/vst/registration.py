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
# The identity table ``--register`` extends alongside the param-spec registry.
_SYNTH_SPEC_RELPATH = Path("src/synth_setter/synth_spec.py")
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

    .. attribute :: synth_module

       The identity-table module the transform rewrites in place.

    .. attribute :: synth_config

       Hydra synth-identity group config generated from the table row.

    .. attribute :: identity_config

       Root ``configs/synth`` group config selecting this synth (#2565).
    """

    spec_module: Path
    preset: Path
    csv: Path
    render_config: Path
    registry: Path
    synth_module: Path
    synth_config: Path
    identity_config: Path


def registration_paths(repo_root: Path, spec_name: str) -> RegistrationPaths:
    """Compute where each artifact for ``spec_name`` lands under ``repo_root``.

    :param repo_root: Synth-setter checkout root.
    :param spec_name: Registry key for the synth.
    :returns: Each artifact's destination path.
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
        synth_module=repo_root / _SYNTH_SPEC_RELPATH,
        synth_config=(
            repo_root / "src/synth_setter/configs/render/synth" / f"{spec_name}.yaml"
        ),
        identity_config=repo_root / "src/synth_setter/configs/synth" / f"{spec_name}.yaml",
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


def _module_dict(tree: ast.Module, dict_name: str) -> ast.Dict | None:
    """Return one named module-level dictionary expression.

    :param tree: Parsed registry module.
    :param dict_name: Module-level dictionary variable to inspect.
    :returns: The dictionary expression, or ``None`` when absent.
    """
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names_dict = any(
            isinstance(target, ast.Name) and target.id == dict_name for target in targets
        )
        if names_dict and isinstance(node.value, ast.Dict):
            return node.value
    return None


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
    mapping = _module_dict(tree, dict_name)
    if mapping is None:
        return []
    return [
        lines[key.lineno - 1]
        for key in mapping.keys
        if key is not None and _registry_key_value(key) == key_value
    ]


def _dict_values(tree: ast.Module, *, dict_name: str, key_value: str) -> list[ast.expr]:
    """Return values assigned to one logical key in a module-level dict.

    :param tree: Parsed registry module.
    :param dict_name: Module-level dictionary variable to inspect.
    :param key_value: String key identity to find.
    :returns: Matching value expressions.
    """
    mapping = _module_dict(tree, dict_name)
    if mapping is None:
        return []
    return [
        value
        for key, value in zip(mapping.keys, mapping.values, strict=True)
        if key is not None and _registry_key_value(key) == key_value
    ]


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
    stays clean) plus one ``_param_specs`` entry. The preset mapping is derived
    from ``synth_spec.SYNTHS`` and needs no separate write.
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
    )
    if any(logical_wiring):
        if all(
            matches == [expected]
            for matches, expected in zip(
                logical_wiring, (import_line, spec_entry), strict=True
            )
        ):
            return source
        raise ValueError(
            f"{spec_name!r} is already registered in param_spec_registry with different "
            "wiring; pick another --spec-name or remove the existing entries first."
        )

    lines.insert(_import_insert_index(lines, module), import_line)
    _insert_dict_entry(lines, "_param_specs", spec_entry)
    return "\n".join(lines) + "\n"


def synths_with_spec(
    source: str, spec_name: str, *, plugin_path: str, synth_version: str
) -> str:
    """Return ``source`` with ``spec_name`` added to the synth identity table.

    Inserts one formatter-stable row into ``_synth_rows``. Re-applying an identical
    registration is a no-op so ``--force`` re-runs converge instead of erroring.

    :param source: Current ``synth_spec.py`` source text.
    :param spec_name: Registry key; also names the param spec and render group.
    :param plugin_path: ``.vst3`` path recorded for render workers.
    :param synth_version: Plugin version recorded in synth identity.
    :returns: The modified ``synth_spec.py`` source.
    :raises ValueError: ``spec_name`` is already present with different wiring,
        or ``source`` lacks the ``_synth_rows`` anchor.
    """
    expected = (spec_name, plugin_path, preset_repo_path(spec_name), synth_version)
    row = "\n".join(
        [
            f'    "{spec_name}": (',
            f"        {json.dumps(spec_name)},",
            f"        {json.dumps(plugin_path)},",
            f"        {json.dumps(preset_repo_path(spec_name))},",
            f"        {json.dumps(synth_version)},",
            "    ),",
        ]
    )

    lines = source.splitlines()
    existing = _dict_values(ast.parse(source), dict_name="_synth_rows", key_value=spec_name)
    if existing:
        try:
            matches_expected = len(existing) == 1 and ast.literal_eval(existing[0]) == expected
        except (TypeError, ValueError):
            matches_expected = False
        if matches_expected:
            return source
        raise ValueError(
            f"{spec_name!r} is already registered in synth_spec with different wiring; "
            "pick another --spec-name or remove the existing row first."
        )

    _insert_dict_entry(lines, "_synth_rows", row, source_name="synth_spec")
    return "\n".join(lines) + "\n"


def _identity_lines(spec_name: str) -> list[str]:
    """Shared header + identity lines of both generated synth group configs.

    Identity scalars are double-quoted via ``json.dumps`` so an arbitrary
    plugin path cannot break the YAML scalar.

    :param spec_name: Registry key; names the param spec.
    :returns: Provenance comment plus ``name`` / ``param_spec_name`` lines.
    """
    return [
        "# Generated artifact of ``synth_setter.synth_spec.SYNTHS``; "
        "edit the table, not this file.",
        f"name: {json.dumps(spec_name)}",
        f"param_spec_name: {json.dumps(spec_name)}",
    ]


def synth_group_yaml(spec_name: str, *, plugin_path: str, synth_version: str) -> str:
    """Emit the Hydra synth-identity group config for ``spec_name``.

    The generated projection of the ``SYNTHS`` row; ``tests/test_synth_spec.py``
    pins the two to agree.

    :param spec_name: Registry key; names the param spec and preset.
    :param plugin_path: ``.vst3`` path recorded for render workers.
    :param synth_version: Plugin version pinned in synth identity.
    :returns: YAML text for ``configs/render/synth/<spec_name>.yaml``.
    """
    return "\n".join(
        [
            *_identity_lines(spec_name),
            f"plugin_path: {json.dumps(plugin_path)}",
            f"plugin_state_path: {json.dumps(preset_repo_path(spec_name))}",
            f"synth_version: {json.dumps(synth_version)}",
            "",
        ]
    )


def identity_group_yaml(spec_name: str) -> str:
    """Emit the root ``configs/synth`` identity config for ``spec_name``.

    Identity-only projection of the ``SYNTHS`` row consumed via
    ``${synth.param_spec_name}``; ``tests/schemas/test_synth_config.py`` pins
    the group and the table to stay bijective.

    :param spec_name: Registry key; names the param spec and the group file.
    :returns: YAML text for ``configs/synth/<spec_name>.yaml``.
    """
    return "\n".join([*_identity_lines(spec_name), ""])


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


def _insert_dict_entry(
    lines: list[str], dict_name: str, entry: str, *, source_name: str = "param_spec_registry"
) -> None:
    """Insert ``entry`` before the closing brace of module-level dict ``dict_name``.

    :param lines: Source lines, mutated in place.
    :param dict_name: Name of the dict assignment to extend.
    :param entry: Pre-indented ``"key": value,`` line.
    :param source_name: Module named in the error, for a legible failure.
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
    raise ValueError(f"{source_name} source has no module-level dict {dict_name!r} to extend")


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


def render_config_yaml(spec_name: str) -> str:
    """Emit the Hydra render group config selecting ``spec_name``.

    Generic render knobs (sample rate, shard sizing, cadences) inherit from
    the ``vst`` group config, while this config pins the synth's identity.
    Every identity scalar is double-quoted via ``json.dumps`` (a subset of
    YAML's double-quote style) so arbitrary plugin paths cannot break the scalar.
    A spec name that is a YAML 1.1 literal (``on``, ``true``) stays a string.

    :param spec_name: Registry key; names the param spec and preset.
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
            f"  - synth: {spec_name}",
            "  - _self_",
            "",
        ]
    )
