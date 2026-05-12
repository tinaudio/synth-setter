"""Tests for ``scripts/check_no_new_funcs_in_pydoclint_excluded.py``.

The guard closes the P6 exclude-list-bypass gap: new ``def`` / ``class``
declarations in files that ``[tool.pydoclint].exclude`` skips would
otherwise enter ``main`` without ever being linted. This module pins the
guard's behaviour on synthetic diffs so the CI check can be trusted as a
one-way ratchet.

See PR for the adversarial probe; the audit issue is #938.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

GUARD_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_no_new_funcs_in_pydoclint_excluded.py"
)


def _load_guard() -> ModuleType:
    """Import the guard via importlib without mutating ``sys.path``.

    :returns: The loaded module.
    :rtype: ModuleType
    """
    spec = importlib.util.spec_from_file_location(
        "check_no_new_funcs_in_pydoclint_excluded", GUARD_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()

PYDOCLINT_EXCLUDE_REGEX = r"""(?x)
    ^src/eval\.py$
  | ^src/pipeline/r2_io\.py$
"""


def test_extract_exclude_regex_reads_pydoclint_table(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """The regex is loaded from the project's pyproject.toml verbatim."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.pydoclint]\nexclude = '''" + PYDOCLINT_EXCLUDE_REGEX + "'''\n")
    assert guard.load_exclude_regex(pyproject).pattern == PYDOCLINT_EXCLUDE_REGEX


def test_extract_exclude_regex_errors_when_pyproject_lacks_section(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """A pyproject without the pydoclint section is a configuration error, not a silent pass."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'demo'\n")
    with pytest.raises(KeyError):
        guard.load_exclude_regex(pyproject)


def test_find_new_defs_flags_added_def_in_excluded_file() -> None:
    """A `+def` line in an excluded file is flagged with its function name."""
    diff = (
        "diff --git a/src/eval.py b/src/eval.py\n"
        "index abc..def 100644\n"
        "--- a/src/eval.py\n"
        "+++ b/src/eval.py\n"
        "@@ -10,0 +10,4 @@\n"
        "+def newly_added(x: int) -> int:\n"
        '+    """New."""\n'
        "+    return x\n"
        "+\n"
    )
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    findings = guard.find_new_defs_in_excluded(diff, regex)
    assert findings == [("src/eval.py", "newly_added", 10)]


def test_find_new_defs_flags_added_class_in_excluded_file() -> None:
    """A `+class` line is flagged just like `+def`."""
    diff = (
        "diff --git a/src/pipeline/r2_io.py b/src/pipeline/r2_io.py\n"
        "--- a/src/pipeline/r2_io.py\n"
        "+++ b/src/pipeline/r2_io.py\n"
        "@@ -5,0 +5,3 @@\n"
        "+class NewType:\n"
        '+    """Added."""\n'
        "+    pass\n"
    )
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    findings = guard.find_new_defs_in_excluded(diff, regex)
    assert findings == [("src/pipeline/r2_io.py", "NewType", 5)]


def test_find_new_defs_flags_async_def_in_excluded_file() -> None:
    """`+async def` is flagged like `+def`."""
    diff = (
        "diff --git a/src/eval.py b/src/eval.py\n"
        "--- a/src/eval.py\n"
        "+++ b/src/eval.py\n"
        "@@ -7,0 +7,2 @@\n"
        "+async def fetcher() -> None:\n"
        '+    """Async."""\n'
    )
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    findings = guard.find_new_defs_in_excluded(diff, regex)
    assert findings == [("src/eval.py", "fetcher", 7)]


def test_find_new_defs_ignores_non_excluded_file() -> None:
    """A `+def` in a non-excluded file is the ruff D-rules' problem, not this guard's."""
    diff = (
        "diff --git a/src/utils/logging_utils.py b/src/utils/logging_utils.py\n"
        "--- a/src/utils/logging_utils.py\n"
        "+++ b/src/utils/logging_utils.py\n"
        "@@ -1,0 +1,2 @@\n"
        "+def something() -> None:\n"
        "+    pass\n"
    )
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    assert guard.find_new_defs_in_excluded(diff, regex) == []


def test_find_new_defs_ignores_in_function_indented_def() -> None:
    """Nested `def` inside an existing function isn't a new top-level/method declaration to
    flag."""
    diff = (
        "diff --git a/src/eval.py b/src/eval.py\n"
        "--- a/src/eval.py\n"
        "+++ b/src/eval.py\n"
        "@@ -1,0 +1,2 @@\n"
        "+        def closure() -> int:\n"
        "+            return 1\n"
    )
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    assert guard.find_new_defs_in_excluded(diff, regex) == []


def test_find_new_defs_flags_method_at_class_indent_level() -> None:
    """A 4-space-indented `+def` is a class method addition — flag it."""
    diff = (
        "diff --git a/src/eval.py b/src/eval.py\n"
        "--- a/src/eval.py\n"
        "+++ b/src/eval.py\n"
        "@@ -10,0 +10,2 @@\n"
        "+    def added_method(self) -> int:\n"
        "+        return 1\n"
    )
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    findings = guard.find_new_defs_in_excluded(diff, regex)
    assert findings == [("src/eval.py", "added_method", 10)]


def test_find_new_defs_ignores_deleted_def() -> None:
    """A `-def` (removed function) is not a new declaration."""
    diff = (
        "diff --git a/src/eval.py b/src/eval.py\n"
        "--- a/src/eval.py\n"
        "+++ b/src/eval.py\n"
        "@@ -10,2 +10,0 @@\n"
        "-def gone(x: int) -> int:\n"
        "-    return x\n"
    )
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    assert guard.find_new_defs_in_excluded(diff, regex) == []


def test_find_new_defs_ignores_addition_in_test_diff_artifact() -> None:
    """`+++ b/file.py` and `+-+` headers must not look like additions."""
    diff = "diff --git a/src/eval.py b/src/eval.py\n--- a/src/eval.py\n+++ b/src/eval.py\n"
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    assert guard.find_new_defs_in_excluded(diff, regex) == []


def test_find_new_defs_does_not_count_no_newline_marker_toward_line_numbers() -> None:
    r"""``\ No newline at end of file`` is diff metadata, not a file line — line numbers after the
    marker must not be skewed."""
    diff = (
        "diff --git a/src/eval.py b/src/eval.py\n"
        "--- a/src/eval.py\n"
        "+++ b/src/eval.py\n"
        "@@ -10,1 +10,3 @@\n"
        " existing_line\n"
        "\\ No newline at end of file\n"
        "+def added_after_marker(x: int) -> int:\n"
        "+    return x\n"
    )
    import re

    regex = re.compile(PYDOCLINT_EXCLUDE_REGEX)
    findings = guard.find_new_defs_in_excluded(diff, regex)
    assert findings == [("src/eval.py", "added_after_marker", 11)]


def test_main_exits_zero_when_no_findings(tmp_path: Path) -> None:  # noqa: DOC101,DOC103
    """`main` exits 0 and prints nothing for a clean diff."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.pydoclint]\nexclude = '''" + PYDOCLINT_EXCLUDE_REGEX + "'''\n")
    clean_diff = (
        "diff --git a/src/utils/logging_utils.py b/src/utils/logging_utils.py\n"
        "--- a/src/utils/logging_utils.py\n"
        "+++ b/src/utils/logging_utils.py\n"
        "@@ -1,0 +1,2 @@\n"
        "+def something() -> None:\n"
        "+    pass\n"
    )
    exit_code = guard.run(diff_text=clean_diff, pyproject_path=pyproject)
    assert exit_code == 0


def test_main_exits_one_and_prints_findings_when_violation_present(  # noqa: DOC101,DOC103
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main` exits 1 and reports each finding with file:line: name."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.pydoclint]\nexclude = '''" + PYDOCLINT_EXCLUDE_REGEX + "'''\n")
    violation_diff = (
        "diff --git a/src/eval.py b/src/eval.py\n"
        "--- a/src/eval.py\n"
        "+++ b/src/eval.py\n"
        "@@ -10,0 +10,2 @@\n"
        "+def offender(x: int) -> int:\n"
        "+    return x\n"
    )
    exit_code = guard.run(diff_text=violation_diff, pyproject_path=pyproject)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "src/eval.py:10: offender" in captured.out
