"""The secret-dependent browser job accepts only trusted event contexts."""

import json
import shutil
from pathlib import Path

import pytest
import sh
from workflow_fixtures import load_workflow


@pytest.mark.parametrize(
    ("event", "actor", "head_repo", "expected"),
    [
        ("pull_request", "contributor", "owner/repo", True),
        ("pull_request", "dependabot[bot]", "owner/repo", False),
        ("pull_request", "contributor", "fork/repo", False),
        ("push", "contributor", "owner/repo", True),
        ("workflow_dispatch", "contributor", "owner/repo", True),
    ],
)
def test_browser_surge_job_authorizes_event_context(
    project_root: Path, event: str, actor: str, head_repo: str, expected: bool
) -> None:
    """The actual job predicate keeps secretless PRs out without disabling trusted runs.

    :param project_root: Repository containing the production workflow.
    :param event: GitHub trigger name.
    :param actor: Triggering account.
    :param head_repo: Pull-request source repository.
    :param expected: Whether this context can run the secret-dependent job.
    """
    web = project_root / "src/synth_setter/web"
    if shutil.which("node") is None or not (web / "node_modules/@actions/expressions").is_dir():
        pytest.skip("Install Node and run npm ci --prefix src/synth_setter/web")
    workflow = load_workflow(project_root, "browser-surge-wasm-e2e.yml")
    request = {
        "expression": workflow["jobs"]["browser-surge-wasm"]["if"],
        "context": {
            "github": {
                "event_name": event,
                "actor": actor,
                "repository": "owner/repo",
                "event": {"pull_request": {"head": {"repo": {"full_name": head_repo}}}},
            }
        },
    }
    result = sh.Command("node")(
        "--input-type=module",
        "-e",
        """
import { readFileSync } from 'node:fs';
import { Lexer, Parser, Evaluator, data } from '@actions/expressions';
const request = JSON.parse(readFileSync(0, 'utf8'));
const tokens = new Lexer(request.expression).lex().tokens;
const expression = new Parser(tokens, ['github'], []).parse();
const context = JSON.parse(JSON.stringify(request.context), data.reviver);
console.log(new Evaluator(expression, context).evaluate().coerceString());
""",
        _in=json.dumps(request),
        _cwd=web,
        _timeout=10,
    )
    assert json.loads(str(result)) is expected
