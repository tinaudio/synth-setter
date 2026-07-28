#!/usr/bin/env python3
"""Require jaxtyping and beartype on new modeling callables."""

from __future__ import annotations

import argparse
import ast
import os
import re
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

    def __init__(self, relative_path: Path, tree: ast.Module) -> None:
        """Initialize the visitor for a baseline-relative path.

        :param relative_path: Python file path relative to the models directory.
        :param tree: Module tree used for source-ordered import resolution.
        """
        self._relative_path = relative_path.as_posix()
        self._tree = tree
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
        imports = _collect_import_names(self._tree, before_line=node.lineno)
        if not _has_runtime_typecheck(node, imports):
            self.diagnostics.append(
                Diagnostic(
                    f"{key_prefix}:JAX001",
                    "add @jaxtyped(typechecker=beartype)",
                )
            )
        if any(
            _contains_bare_torch_tensor(annotation, imports) for annotation in _annotations(node)
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


def _collect_import_names(tree: ast.Module, *, before_line: int) -> ImportNames:
    """Resolve module bindings active before a callable definition.

    :param tree: Module syntax tree to inspect.
    :param before_line: Ignore statements at or after this source line.
    :returns: Immutable local-name groups for lint matching.
    """
    bindings: dict[str, str] = {}
    for statement in tree.body:
        if statement.lineno >= before_line:
            break
        _apply_module_binding(bindings, statement)
    return _import_names_from_bindings(bindings)


def _apply_module_binding(bindings: dict[str, str], statement: ast.stmt) -> None:
    """Apply one top-level statement to the import binding map.

    :param bindings: Mutable map from local names to canonical symbol kinds.
    :param statement: Top-level statement executed before a callable definition.
    """
    if isinstance(statement, ast.Import):
        for alias in statement.names:
            local_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            bindings[local_name] = _module_import_kind(alias)
        return
    if isinstance(statement, ast.ImportFrom):
        for alias in statement.names:
            bindings[alias.asname or alias.name] = _from_import_kind(statement, alias)
        return
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bindings[statement.name] = "other"
        return
    for name in _assignment_names(statement):
        bindings[name] = "other"


def _module_import_kind(alias: ast.alias) -> str:
    """Classify a module import for lint resolution.

    :param alias: Imported module and its local alias.
    :returns: Canonical symbol kind or ``other``.
    """
    if alias.name == "beartype":
        return "beartype_module"
    if alias.name == "jaxtyping":
        return "jaxtyping_module"
    if alias.name == "torch" or (alias.asname is None and alias.name.startswith("torch.")):
        return "torch_module"
    return "other"


def _from_import_kind(statement: ast.ImportFrom, alias: ast.alias) -> str:
    """Classify a direct import for lint resolution.

    :param statement: Import statement containing the source module.
    :param alias: Imported symbol and its local alias.
    :returns: Canonical symbol kind or ``other``.
    """
    if statement.module == "beartype" and alias.name == "beartype":
        return "beartype_function"
    if statement.module == "jaxtyping" and alias.name == "jaxtyped":
        return "jaxtyped_function"
    if statement.module == "jaxtyping":
        return "jaxtyping_wrapper"
    if statement.module == "torch" and alias.name == "Tensor":
        return "torch_tensor"
    return "other"


def _assignment_names(statement: ast.stmt) -> set[str]:
    """Return names assigned directly by a top-level statement.

    :param statement: Statement that may overwrite an imported name.
    :returns: Direct module-scope assignment targets.
    """
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return {
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
    return set()


def _import_names_from_bindings(bindings: dict[str, str]) -> ImportNames:
    """Group active module bindings by canonical symbol kind.

    :param bindings: Local names mapped to canonical symbol kinds.
    :returns: Immutable local-name groups for lint matching.
    """
    names_by_kind: dict[str, set[str]] = {}
    for name, kind in bindings.items():
        names_by_kind.setdefault(kind, set()).add(name)
    frozen_by_kind = {kind: frozenset(names) for kind, names in names_by_kind.items()}
    return ImportNames(
        beartype_functions=frozen_by_kind.get("beartype_function", frozenset()),
        beartype_modules=frozen_by_kind.get("beartype_module", frozenset()),
        jaxtyped_functions=frozen_by_kind.get("jaxtyped_function", frozenset()),
        jaxtyping_modules=frozen_by_kind.get("jaxtyping_module", frozenset()),
        jaxtyping_wrappers=frozen_by_kind.get("jaxtyping_wrapper", frozenset()),
        torch_modules=frozen_by_kind.get("torch_module", frozenset()),
        torch_tensors=frozen_by_kind.get("torch_tensor", frozenset()),
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
        return _contains_bare_tensor_forward_reference(
            annotation.value,
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
    children = (child for child in ast.iter_child_nodes(annotation) if isinstance(child, ast.expr))
    return any(
        _contains_bare_torch_tensor(child, imports, in_jaxtyping=in_jaxtyping)
        for child in children
    )


def _contains_bare_tensor_forward_reference(
    value: str,
    imports: ImportNames,
    *,
    in_jaxtyping: bool,
) -> bool:
    """Inspect a quoted annotation, including malformed expressions.

    :param value: Forward-reference annotation text.
    :param imports: Local tensor and jaxtyping names.
    :param in_jaxtyping: Whether the string is nested under a jaxtyping wrapper.
    :returns: Whether the text contains a bare Tensor reference.
    """
    try:
        parsed = ast.parse(value, mode="eval").body
    except SyntaxError:
        return not in_jaxtyping and _string_mentions_torch_tensor(value, imports)
    return _contains_bare_torch_tensor(parsed, imports, in_jaxtyping=in_jaxtyping)


def _string_mentions_torch_tensor(value: str, imports: ImportNames) -> bool:
    """Detect an imported Tensor spelling in malformed annotation text.

    :param value: Forward-reference text that failed expression parsing.
    :param imports: Local torch names.
    :returns: Whether the text contains a resolved Tensor spelling.
    """
    candidates = {*imports.torch_tensors}
    candidates.update(f"{module}.Tensor" for module in imports.torch_modules)
    return any(
        re.search(rf"(?<![\\w.]){re.escape(candidate)}(?!\\w)", value) is not None
        for candidate in candidates
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
        visitor = ModelTypingVisitor(path.relative_to(models_dir), tree)
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


def _run_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a read-only git query with captured output.

    :param cwd: Directory from which git resolves the repository.
    :param args: Git arguments after the executable.
    :returns: Completed process without exception-on-failure behavior.
    """
    return subprocess.run(  # noqa: S603 - fixed git executable and read-only callers.
        [GIT, "-C", str(cwd), *args],
        capture_output=True,
        check=False,
        text=True,
    )


def _read_base_baseline(path: Path, *, allow_missing_git_base: bool) -> set[str] | None:
    """Read the frozen baseline from the pull request's git base.

    :param path: Current baseline path.
    :param allow_missing_git_base: Permit isolated fixtures outside git.
    :returns: Base keys, or ``None`` when the base does not contain the file.
    :raises RuntimeError: If an expected git base cannot be resolved.
    """
    root_result = _run_git(path.parent, ["rev-parse", "--show-toplevel"])
    if root_result.returncode != 0:
        if allow_missing_git_base:
            return None
        raise RuntimeError("cannot locate git root for model typing baseline")

    root = Path(root_result.stdout.strip())
    relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    explicit_base = os.environ.get("MODEL_TYPING_BASE_REF")
    github_base = os.environ.get("GITHUB_BASE_REF")
    base_ref = explicit_base or (f"origin/{github_base}" if github_base else "origin/main")
    ref_result = _run_git(root, ["rev-parse", "--verify", f"{base_ref}^{{commit}}"])
    if ref_result.returncode != 0:
        raise RuntimeError(f"cannot resolve model typing base ref {base_ref!r}")

    show_result = _run_git(root, ["show", f"{base_ref}:{relative_path}"])
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
    parser.add_argument("--allow-missing-git-base", action="store_true")
    args = parser.parse_args()

    diagnostics = collect_diagnostics(args.models_dir)
    diagnostic_by_key = {diagnostic.key: diagnostic for diagnostic in diagnostics}
    baseline = _baseline_keys(args.baseline.read_text())
    try:
        base_baseline = _read_base_baseline(
            args.baseline,
            allow_missing_git_base=args.allow_missing_git_base,
        )
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
