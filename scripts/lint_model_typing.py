#!/usr/bin/env python3
"""Require jaxtyping and beartype on new modeling callables."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

JAXTYPING_ANNOTATIONS = frozenset(
    {
        "Array",
        "BFloat16",
        "Bool",
        "Complex",
        "Complex64",
        "Complex128",
        "Float",
        "Float16",
        "Float32",
        "Float64",
        "Inexact",
        "Int",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Integer",
        "Key",
        "Num",
        "Real",
        "Shaped",
        "UInt",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    }
)


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


class ModelTypingVisitor(ast.NodeVisitor):
    """Collect modeling-typing diagnostics from one Python module."""

    def __init__(self, relative_path: Path) -> None:
        """Initialize the visitor for a baseline-relative path.

        :param relative_path: Python file path relative to the models directory.
        """
        self._relative_path = relative_path.as_posix()
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
        if not _has_runtime_typecheck(node):
            self.diagnostics.append(
                Diagnostic(
                    f"{key_prefix}:JAX001",
                    "add @jaxtyped(typechecker=beartype)",
                )
            )
        if any(_contains_bare_torch_tensor(annotation) for annotation in _annotations(node)):
            self.diagnostics.append(
                Diagnostic(
                    f"{key_prefix}:JAX002",
                    "replace bare torch.Tensor with a jaxtyping annotation",
                )
            )

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


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


def _has_runtime_typecheck(node: FunctionNode) -> bool:
    """Detect the required beartype-backed jaxtyped decorator.

    :param node: Callable definition whose decorators are checked.
    :returns: Whether the required decorator is present.
    """
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or _name(decorator.func) != "jaxtyped":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "typechecker" and _name(keyword.value) == "beartype":
                return True
    return False


def _contains_bare_torch_tensor(annotation: ast.expr, *, in_jaxtyping: bool = False) -> bool:
    """Detect Tensor references outside a jaxtyping annotation.

    :param annotation: Annotation expression to inspect recursively.
    :param in_jaxtyping: Whether the expression is nested under a jaxtyping wrapper.
    :returns: Whether a bare Tensor reference is present.
    """
    if isinstance(annotation, ast.Subscript):
        wrapper_name = _name(annotation.value)
        wrapped_by_jaxtyping = (
            wrapper_name is not None
            and wrapper_name.rsplit(".", maxsplit=1)[-1] in JAXTYPING_ANNOTATIONS
        )
        return _contains_bare_torch_tensor(
            annotation.value, in_jaxtyping=in_jaxtyping
        ) or _contains_bare_torch_tensor(
            annotation.slice,
            in_jaxtyping=in_jaxtyping or wrapped_by_jaxtyping,
        )
    if isinstance(annotation, (ast.Tuple, ast.List)):
        return any(
            _contains_bare_torch_tensor(element, in_jaxtyping=in_jaxtyping)
            for element in annotation.elts
        )
    if isinstance(annotation, ast.BinOp):
        return _contains_bare_torch_tensor(
            annotation.left, in_jaxtyping=in_jaxtyping
        ) or _contains_bare_torch_tensor(annotation.right, in_jaxtyping=in_jaxtyping)
    return not in_jaxtyping and _name(annotation) in {"Tensor", "torch.Tensor"}


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
        visitor = ModelTypingVisitor(path.relative_to(models_dir))
        visitor.visit(tree)
        diagnostics.extend(visitor.diagnostics)
    return sorted(diagnostics)


def _read_baseline(path: Path) -> set[str]:
    """Read active diagnostic keys from a baseline file.

    :param path: Baseline file containing one key per line.
    :returns: Non-empty, non-comment keys.
    """
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


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
    baseline = _read_baseline(args.baseline)
    new_keys = sorted(diagnostic_by_key.keys() - baseline)
    stale_keys = sorted(baseline - diagnostic_by_key.keys())

    for key in new_keys:
        sys.stdout.write(f"{key}: {diagnostic_by_key[key].message}\n")
    for key in stale_keys:
        sys.stdout.write(f"stale baseline entry: {key}\n")
    return int(bool(new_keys or stale_keys))


if __name__ == "__main__":
    raise SystemExit(main())
