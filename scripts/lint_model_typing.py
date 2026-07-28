#!/usr/bin/env python3
"""Require jaxtyping and beartype on new modeling callables."""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

GIT = shutil.which("git") or "git"


@dataclass(frozen=True, order=True)
class Diagnostic:
    """Pair a baseline-stable key with its remediation.

    .. attribute :: key

        Baseline identity for one callable and rule.

    .. attribute :: message

        Action required to satisfy the rule.
    """

    key: str
    message: str


@dataclass(frozen=True)
class ImportNames:
    """Track local names for the three typing libraries.

    .. attribute :: beartype_functions

        Local names imported from ``beartype.beartype``.

    .. attribute :: beartype_modules

        Local aliases for the ``beartype`` module.

    .. attribute :: jaxtyped_functions

        Local names imported from ``jaxtyping.jaxtyped``.

    .. attribute :: jaxtyping_modules

        Local aliases for the ``jaxtyping`` module.

    .. attribute :: jaxtyping_wrappers

        Local names imported from ``jaxtyping`` for use as annotation wrappers.

    .. attribute :: torch_modules

        Local aliases for the ``torch`` module.

    .. attribute :: torch_tensors

        Local names imported from ``torch.Tensor``.
    """

    beartype_functions: frozenset[str]
    beartype_modules: frozenset[str]
    jaxtyped_functions: frozenset[str]
    jaxtyping_modules: frozenset[str]
    jaxtyping_wrappers: frozenset[str]
    torch_modules: frozenset[str]
    torch_tensors: frozenset[str]


class ModelTypingVisitor(ast.NodeVisitor):
    """Collect modeling-typing diagnostics from one Python module."""

    def __init__(self, relative_path: Path, imports: ImportNames) -> None:
        """Initialize the visitor for a baseline-relative path.

        :param relative_path: Python file path relative to the models directory.
        :param imports: Local names that resolve to typing-library symbols.
        """
        self._relative_path = relative_path.as_posix()
        self._imports = imports
        self._scope: list[str] = []
        self.diagnostics: list[Diagnostic] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        """Include class names in baseline keys for nested callables.

        :param node: Class definition whose scope contains callables.
        """
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        """Apply JAX001 and JAX002 to a synchronous callable.

        :param node: Function definition to check.
        """
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        """Apply JAX001 and JAX002 to an asynchronous callable.

        :param node: Async function definition to check.
        """
        self._visit_function(node)

    def _visit_function(self, node: FunctionNode) -> None:
        """Record violations and recurse into nested definitions.

        :param node: Callable definition to check.
        """
        qualified_name = ".".join((*self._scope, node.name))
        key_prefix = f"{self._relative_path}:{qualified_name}"
        if not _has_runtime_typecheck(node, self._imports):
            self.diagnostics.append(
                Diagnostic(
                    f"{key_prefix}:JAX001",
                    "add @jaxtyped(typechecker=beartype)",
                )
            )
        if any(
            _contains_bare_torch_tensor(annotation, self._imports)
            for annotation in _annotations(node)
        ):
            self.diagnostics.append(
                Diagnostic(
                    f"{key_prefix}:JAX002",
                    "replace bare torch.Tensor with a jaxtyping annotation",
                )
            )

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def _collect_import_names(tree: ast.Module) -> ImportNames:
    """Resolve local names imported from beartype, jaxtyping, and torch.

    :param tree: Module syntax tree to inspect.
    :returns: Immutable local-name groups for lint matching.
    """
    beartype_functions: set[str] = set()
    beartype_modules: set[str] = set()
    jaxtyped_functions: set[str] = set()
    jaxtyping_modules: set[str] = set()
    jaxtyping_wrappers: set[str] = set()
    torch_modules: set[str] = set()
    torch_tensors: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name == "beartype":
                    beartype_modules.add(local_name)
                elif alias.name == "jaxtyping":
                    jaxtyping_modules.add(local_name)
                elif alias.name == "torch":
                    torch_modules.add(local_name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local_name = alias.asname or alias.name
                if node.module == "beartype" and alias.name == "beartype":
                    beartype_functions.add(local_name)
                elif node.module == "jaxtyping" and alias.name == "jaxtyped":
                    jaxtyped_functions.add(local_name)
                elif node.module == "jaxtyping":
                    jaxtyping_wrappers.add(local_name)
                elif node.module == "torch" and alias.name == "Tensor":
                    torch_tensors.add(local_name)

    return ImportNames(
        beartype_functions=frozenset(beartype_functions),
        beartype_modules=frozenset(beartype_modules),
        jaxtyped_functions=frozenset(jaxtyped_functions),
        jaxtyping_modules=frozenset(jaxtyping_modules),
        jaxtyping_wrappers=frozenset(jaxtyping_wrappers),
        torch_modules=frozenset(torch_modules),
        torch_tensors=frozenset(torch_tensors),
    )


def _annotations(node: FunctionNode) -> list[ast.expr]:
    """Return every parameter and return annotation on a callable.

    :param node: Callable definition containing annotations.
    :returns: Annotation expressions in signature order.
    """
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    annotations = [
        argument.annotation for argument in arguments if argument.annotation is not None
    ]
    if node.args.vararg is not None and node.args.vararg.annotation is not None:
        annotations.append(node.args.vararg.annotation)
    if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
        annotations.append(node.args.kwarg.annotation)
    if node.returns is not None:
        annotations.append(node.returns)
    return annotations


def _has_runtime_typecheck(node: FunctionNode, imports: ImportNames) -> bool:
    """Detect the required beartype-backed jaxtyped decorator.

    :param node: Callable definition whose decorators are checked.
    :param imports: Local names that resolve decorator symbols.
    :returns: Whether the required decorator is present.
    """
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or not _matches_symbol(
            decorator.func,
            direct_names=imports.jaxtyped_functions,
            module_names=imports.jaxtyping_modules,
            attribute="jaxtyped",
        ):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "typechecker" and _matches_symbol(
                keyword.value,
                direct_names=imports.beartype_functions,
                module_names=imports.beartype_modules,
                attribute="beartype",
            ):
                return True
    return False


def _contains_bare_torch_tensor(
    annotation: ast.expr,
    imports: ImportNames,
    *,
    in_jaxtyping: bool = False,
) -> bool:
    """Detect Tensor references outside a jaxtyping annotation.

    :param annotation: Annotation expression to inspect recursively.
    :param imports: Local names that resolve tensor and jaxtyping symbols.
    :param in_jaxtyping: Whether the expression is nested under a jaxtyping wrapper.
    :returns: Whether a bare Tensor reference is present.
    """
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            parsed = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False
        return _contains_bare_torch_tensor(
            parsed,
            imports,
            in_jaxtyping=in_jaxtyping,
        )
    if isinstance(annotation, ast.Subscript):
        wrapped_by_jaxtyping = _is_jaxtyping_wrapper(annotation.value, imports)
        return _contains_bare_torch_tensor(
            annotation.value,
            imports,
            in_jaxtyping=in_jaxtyping,
        ) or _contains_bare_torch_tensor(
            annotation.slice,
            imports,
            in_jaxtyping=in_jaxtyping or wrapped_by_jaxtyping,
        )
    if not in_jaxtyping and _matches_symbol(
        annotation,
        direct_names=imports.torch_tensors,
        module_names=imports.torch_modules,
        attribute="Tensor",
    ):
        return True
    return any(
        _contains_bare_torch_tensor(child, imports, in_jaxtyping=in_jaxtyping)
        for child in ast.iter_child_nodes(annotation)
        if isinstance(child, ast.expr)
    )


def _is_jaxtyping_wrapper(node: ast.AST, imports: ImportNames) -> bool:
    """Detect an annotation wrapper imported from jaxtyping.

    :param node: Expression naming a potential wrapper.
    :param imports: Local jaxtyping names.
    :returns: Whether the expression resolves to the jaxtyping module.
    """
    name = _name(node)
    if name is None:
        return False
    if name in imports.jaxtyping_wrappers:
        return True
    return any(name.startswith(f"{module}.") for module in imports.jaxtyping_modules)


def _matches_symbol(
    node: ast.AST,
    *,
    direct_names: frozenset[str],
    module_names: frozenset[str],
    attribute: str,
) -> bool:
    """Match a directly imported or module-qualified symbol.

    :param node: Expression naming the symbol.
    :param direct_names: Local direct-import aliases.
    :param module_names: Local module aliases.
    :param attribute: Attribute expected beneath a module alias.
    :returns: Whether the expression resolves to the requested symbol.
    """
    name = _name(node)
    if name in direct_names:
        return True
    return any(name == f"{module}.{attribute}" for module in module_names)


def _name(node: ast.AST) -> str | None:
    """Return the dotted name represented by an AST expression.

    :param node: Expression that may encode a name or attribute chain.
    :returns: Dotted name, or ``None`` for another expression type.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def collect_diagnostics(models_dir: Path) -> list[Diagnostic]:
    """Collect typing diagnostics beneath a models directory.

    :param models_dir: Modeling package to scan recursively.
    :returns: Stable, sorted diagnostics.
    """
    diagnostics: list[Diagnostic] = []
    for path in sorted(models_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        visitor = ModelTypingVisitor(path.relative_to(models_dir), _collect_import_names(tree))
        visitor.visit(tree)
        diagnostics.extend(visitor.diagnostics)
    return sorted(diagnostics)


def _baseline_keys(text: str) -> set[str]:
    """Parse active diagnostic keys from baseline text.

    :param text: Baseline contents.
    :returns: Non-empty, non-comment keys.
    """
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _read_base_baseline(path: Path) -> set[str] | None:
    """Read the frozen baseline from the pull request's git base.

    :param path: Current baseline path.
    :returns: Base keys, or ``None`` when the base does not contain the file.
    :raises RuntimeError: If an expected git base cannot be resolved.
    """
    root_result = subprocess.run(  # noqa: S603 - fixed git inspection command.
        [GIT, "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        check=False,
        text=True,
    )
    if root_result.returncode != 0:
        return None

    root = Path(root_result.stdout.strip())
    relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    github_base = os.environ.get("GITHUB_BASE_REF")
    base_ref = (
        f"origin/{github_base}"
        if github_base
        else os.environ.get("MODEL_TYPING_BASE_REF", "origin/main")
    )
    ref_result = subprocess.run(  # noqa: S603 - fixed git inspection command.
        [GIT, "-C", str(root), "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        capture_output=True,
        check=False,
        text=True,
    )
    if ref_result.returncode != 0:
        raise RuntimeError(f"cannot resolve model typing base ref {base_ref!r}")

    show_result = subprocess.run(  # noqa: S603 - fixed git inspection command.
        [GIT, "-C", str(root), "show", f"{base_ref}:{relative_path}"],
        capture_output=True,
        check=False,
        text=True,
    )
    if show_result.returncode != 0:
        return None
    return _baseline_keys(show_result.stdout)


def main() -> int:
    """Run the baseline-aware modeling typing lint.

    :returns: Zero when current diagnostics exactly match the baseline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("src/synth_setter/models"))
    parser.add_argument("--baseline", type=Path, default=Path(".model-typing-baseline.txt"))
    args = parser.parse_args()

    diagnostics = collect_diagnostics(args.models_dir)
    diagnostic_by_key = {diagnostic.key: diagnostic for diagnostic in diagnostics}
    baseline = _baseline_keys(args.baseline.read_text())
    try:
        base_baseline = _read_base_baseline(args.baseline)
    except RuntimeError as error:
        sys.stdout.write(f"{error}\n")
        return 1

    added_baseline_keys = sorted(baseline - base_baseline) if base_baseline is not None else []
    new_keys = sorted(diagnostic_by_key.keys() - baseline)
    stale_keys = sorted(baseline - diagnostic_by_key.keys())

    for key in added_baseline_keys:
        sys.stdout.write(f"baseline additions are forbidden: {key}\n")
    for key in new_keys:
        sys.stdout.write(f"{key}: {diagnostic_by_key[key].message}\n")
    for key in stale_keys:
        sys.stdout.write(f"stale baseline entry: {key}\n")
    return int(bool(added_baseline_keys or new_keys or stale_keys))


if __name__ == "__main__":
    raise SystemExit(main())
