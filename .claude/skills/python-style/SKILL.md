---
name: python-style
description: >-
  Google Python Style Guide adapted for this project. Covers imports, exceptions, naming,
  type annotations, docstrings, formatting, comprehensions, and all language rules.
  Used by the /review skill for Python code review.
---

# Python Style Guide

Adapted from the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Where this project's conventions (Ruff format line-length=99, Ruff lint) override Google defaults,
this guide notes the deviation.

**Project overrides:** Ruff format (line-length=99) handles formatting. Ruff handles import
sorting and lint. This guide covers everything Ruff doesn't enforce.

______________________________________________________________________

## 1. Language Rules

### 1.1 Imports

- Use `import x` for packages/modules, `from x import y` where x is the package prefix
- Never use relative imports
- Standard abbreviations OK: `import numpy as np`, `import pandas as pd`
- One import per line (except `typing` and `collections.abc` can combine)
- Group imports: `__future__` → stdlib → third-party → local. Sort lexicographically within groups
- Prefer abstract types in annotations: `Sequence` over `list`, `Mapping` over `dict`
- Use built-in generics: `list[int]` not `typing.List[int]`, `tuple[str, ...]` not `typing.Tuple`

### 1.2 Nested/Local/Inner Classes and Functions

- Nested functions and classes are fine when closing over a local variable
- Do not nest solely to hide a function — use a module-level function with `_` prefix instead
- Inner classes are fine when tightly coupled to the outer class

### 1.3 Default Iterators and Operators

- Use default iterators for types that support them:
  - `for key in adict:` not `for key in adict.keys()`
  - Iterate directly over file objects, not `readlines()`
  - `if key in adict:` not `if adict.has_key(key)`
- Do not mutate containers while iterating over them

### 1.4 Exceptions

- Use built-in exceptions: `ValueError` for precondition violations, `TypeError` for wrong types
- Custom exceptions must inherit from existing exception classes and end with `Error`
- Never use bare `except:` — catch specific exceptions
- Never catch `Exception` or `BaseException` without re-raising (exception: isolation boundaries like thread entry points or top-level handlers)
- Minimize code in `try`/`except` blocks — only wrap the operation that can fail
- Use `finally` for cleanup; prefer `with` statements for resource management
- Do NOT use `assert` for validation — assertions are stripped with `-O`

### 1.5 Mutable Global State

- Avoid mutable global state
- Module-level constants are fine: `MAX_RETRIES = 3` (all-caps)
- Mark module-level mutable values as internal with `_leading_underscore`

### 1.6 Comprehensions & Generators

- Single-clause comprehensions and simple filters are OK
- Multiple `for` clauses or complex filter expressions: use a regular loop instead
- Generator functions are fine; use `:yields:` in docstrings not `:returns:`
- Prefer `operator` module functions over lambdas for common operations

### 1.7 Default Arguments

- **Never use mutable objects as default values** (`list`, `dict`, `set`)
- Use `None` as default, initialize inside the function:
  ```python
  def foo(items: list[str] | None = None) -> list[str]:
      if items is None:
          items = []
  ```

### 1.8 Conditional Expressions

- OK for simple cases: `x = 1 if cond else 2`
- Each portion must fit on one line; otherwise use a full `if` statement

### 1.9 Properties

- Use `@property` only for computations or lazy evaluation
- Don't wrap simple attribute access in a property
- Don't create properties that subclasses might need to override

### 1.10 True/False Evaluations

- Use implicit false: `if foo:` not `if foo != []:`
- Use `if foo is None:` for None checks (never `if foo == None:`)
- For integers, explicit `if count == 0:` is OK to distinguish from None
- For NumPy arrays, use `.size` for emptiness (not implicit bool)
- `'0'` (string) evaluates to True

### 1.11 Decorators

- Use judiciously when there is a clear advantage
- Never use `staticmethod` unless forced by an API
- Use `classmethod` only for named constructors or class-level operations
- Decorators run at import time — avoid external dependencies in decorator code

### 1.12 Threading

- Do not rely on atomicity of built-in types
- Use `queue.Queue` for inter-thread communication
- Prefer `threading.Condition` over lower-level locks

### 1.13 Power Features

- Avoid: custom metaclasses, bytecode access, dynamic inheritance, `__del__` methods
- Standard library internals using these are fine

______________________________________________________________________

## 2. Style Rules

### 2.1 Formatting

**Handled by Ruff:** indentation (4 spaces), line length (99), whitespace, import
sorting. The following are NOT auto-enforced:

- No semicolons to terminate lines or combine statements
- Use implicit line joining with parentheses, not backslash continuation
- Two blank lines between top-level definitions, one between methods
- No trailing whitespace
- Trailing commas recommended when closing bracket is on a different line

### 2.2 Naming

| Type          | Convention           | Example           |
| ------------- | -------------------- | ----------------- |
| Packages      | `lower_with_under`   | `pipeline`        |
| Modules       | `lower_with_under`   | `storage_backend` |
| Classes       | `CapWords`           | `PipelineSpec`    |
| Exceptions    | `CapWords` + `Error` | `ValidationError` |
| Functions     | `lower_with_under`   | `validate_shard`  |
| Constants     | `CAPS_WITH_UNDER`    | `MAX_RETRIES`     |
| Variables     | `lower_with_under`   | `shard_count`     |
| Instance vars | `lower_with_under`   | `self.run_id`     |
| Parameters    | `lower_with_under`   | `num_shards`      |
| Type aliases  | `CapWords`           | `ShardMap`        |

- Choose descriptive, unambiguous names. No abbreviations ambiguous to outsiders.
- Never use single characters except: `i`, `j`, `k` (iterators), `e` (exception), `f` (file handle)
- No dashes in filenames; always use `.py` extension
- Single underscore `_` = internal/protected. Double underscore `__` = discouraged (name mangling)
- Use `_` prefix for module-private classes, functions, and constants
- Test methods: `test_<method>_<state>` pattern

### 2.3 Module Docstrings

- Every file should have a module-level docstring describing its contents and usage
- Include a typical usage example for modules with public APIs:
  ```python
  """Storage backend for R2 and local filesystem operations.

  Provides StorageBackend protocol and two implementations:
  LocalStorageBackend for development/testing and R2StorageBackend
  for production.

  Typical usage:
      storage = LocalStorageBackend(root=Path("/tmp/storage"))
      storage.upload_file(local_path, run_id, "metadata/spec.json")
  """
  ```
- Test module docstrings are optional unless they provide context about test execution
- Docstring summary lines must stay within the project line-length limit (99 chars)

### 2.4 Docstrings

Docstring compliance is enforced by pre-commit hooks.

### 2.5 Comments

- Comments start 2+ spaces from code, `#` followed by space
- Never describe code — assume readers know Python, explain intent
- Use `TODO:` with bug reference: `# TODO: b/12345 - handle edge case`
- Don't reference individuals in TODOs

### 2.6 Strings

- Use f-strings, `%`, or `.format()` for formatting — never `+` in loops
- Accumulate with `list.append()` + `''.join()` for linear complexity
- Be consistent with quote character (`'` or `"`)
- For logging: use pattern-parameters not f-strings: `logger.info('Version: %s', version)`
- Error messages: precisely match conditions, clearly identify interpolated values

### 2.7 Files and Resources

- Always use `with` statements for files, sockets, and similar resources
- Use `contextlib.closing()` for objects lacking `with` support
- Never rely on `__del__` for cleanup

### 2.8 Type Annotations

- Annotate all public APIs and complex code
- Don't annotate `self`/`cls` (except when using `Self` for proper type info)
- Don't annotate `__init__` return type (always `None`)
- Use `X | None` (modern) over `Optional[X]` for nullable types
- Always declare when arguments can be `None`: `a: str | None = None` not `a: str = None`
- Use `TypeAlias` for type aliases: `ShardMap: TypeAlias = dict[int, Path]`
- Specify type parameters for generics: `Sequence[int]` not `Sequence`
- Use `from __future__ import annotations` for forward references
- Use `if TYPE_CHECKING:` for import-only-for-types (exceptional cases only)
- Tuples: `tuple[int, ...]` for homogeneous, `tuple[int, str, float]` for fixed
- Annotated assignments for hard-to-infer types: `result: ShardResult = validate(shard)`
- `TypeVar` and `ParamSpec`: use descriptive names unless unconstrained and internal (`_T`)
- Constrain TypeVars when needed: `Addable = TypeVar("Addable", int, float, str)`
- Circular type dependencies: refactor to eliminate. Last resort: `some_mod = Any`
- Suppress type errors with `# type: ignore` only with specific error codes
- Use `str` for text, `bytes` for binary. Never `typing.Text`

### 2.9 Function Length

- Prefer small, focused functions
- No hard limit, but consider refactoring at ~40 lines
- Keep functions short for easier testing, modification, and debugging

### 2.10 Main Guard

- Use `if __name__ == '__main__':` for all executable modules
- Don't execute top-level operations that shouldn't run during import

______________________________________________________________________

## Review Checklist

| #    | Check                                                                      | Severity |
| ---- | -------------------------------------------------------------------------- | -------- |
| PY1  | Imports: one per line, grouped, sorted, no relative imports                | WARN     |
| PY2  | No bare `except:`, no catching `Exception` without re-raising              | BLOCK    |
| PY3  | No mutable default arguments (`list`, `dict`, `set` as defaults)           | BLOCK    |
| PY4  | No `assert` for validation (stripped with `-O`)                            | BLOCK    |
| PY5  | Naming follows convention table above                                      | WARN     |
| PY6  | Public functions have docstrings                                           | WARN     |
| PY7  | Module docstrings present with usage examples for public modules           | WARN     |
| PY8  | Type annotations on all public API function signatures                     | BLOCK    |
| PY9  | Nullable types explicit: `X \| None` not implicit via `= None`             | WARN     |
| PY10 | TypeVar/ParamSpec: descriptive names, constrained when needed              | WARN     |
| PY11 | No circular type deps; `if TYPE_CHECKING:` used sparingly                  | WARN     |
| PY12 | No mutable global state (module-level constants OK)                        | WARN     |
| PY13 | `with` statements for all file/socket/resource operations                  | BLOCK    |
| PY14 | Logging uses pattern-parameters: `logger.info('x: %s', val)` not f-strings | WARN     |
| PY15 | Comprehensions: single clause only, no nested `for`                        | WARN     |
| PY16 | No `staticmethod`; `classmethod` only for named constructors               | WARN     |
| PY17 | Functions \<= ~40 lines; refactor if longer                                | WARN     |
| PY18 | `if __name__ == '__main__':` guard on executable modules                   | WARN     |
| PY19 | Nested functions only when closing over locals; else use `_` module-level  | WARN     |
| PY20 | Default iterators used; no mutating containers while iterating             | WARN     |
| PY21 | Docstring summary lines \<= 99 chars                                       | WARN     |
