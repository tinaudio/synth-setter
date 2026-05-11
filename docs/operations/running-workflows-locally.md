# Running GitHub Workflows Locally (act)

## When to use this

Use [`act`](https://github.com/nektos/act) to run the workflows under
`.github/workflows/` on a local Docker host. The two main reasons:

- **Debug a failing CI run** without `git push`-and-wait. `act` reproduces the
  step output, exit codes, and env exactly enough that most non-cloud failures
  surface in the same step locally.
- **Iterate on workflow YAML changes** (a new step, a different action version,
  a fix to a shell snippet) and exercise them against real PRs / events before
  opening a PR.

This guide is operational — what to install, how to wire `.env` in, where the
gotchas live. For the **purpose** of each workflow, what triggers it, and the
secret/artifact graph, read [`../reference/github-actions.md`](../reference/github-actions.md).

## Prerequisites

- Docker daemon reachable at `/var/run/docker.sock`.
- `act` ≥ 0.2.88 (`brew install act` / `go install github.com/nektos/act@latest`).
- `gh` on the host, authenticated (`gh auth status`) — useful for borrowing a
  live `GITHUB_TOKEN` and for inspecting real PR events to feed `act`.

## First-time setup

`act` prompts on first run for a default image size and dies on EOF in a
non-TTY (CI containers, dev container `bash -c`, ssh-without-tty). Avoid the
prompt by writing the mappings once:

```bash
mkdir -p ~/.config/act
cat > ~/.config/act/actrc <<'EOF'
-P ubuntu-latest=catthehacker/ubuntu:act-latest
-P ubuntu-22.04=catthehacker/ubuntu:act-22.04
-P ubuntu-20.04=catthehacker/ubuntu:act-20.04
EOF
```

The medium [`catthehacker/ubuntu:act-*`](https://github.com/catthehacker/docker_images)
images are ~2.3 GB and cover the bulk of our workflows. The full GitHub-runner
clone (`full-latest`) is ~60 GB and rarely necessary.

## Listing what's available

```bash
act -l
```

Prints every job × event combination across all workflow files, with the
runner image label and the trigger events. Job IDs are not globally unique
(several workflows have `run_tests`, `code-quality`, etc.) — narrow with
`-W <path>` when the listing flags duplicates.

## Secrets: feed `.env` directly

`.env` is already the local source of truth for credentials. Pass it as
`act`'s secrets file — no preprocessing required, including for multi-line
PEM values (`OCI_API_KEY_PEM`):

```bash
act <event> --secret-file .env -W .github/workflows/<file>
```

### `GITHUB_TOKEN` rename

Workflows reference `secrets.GITHUB_TOKEN`. The matching value in `.env` is
named `GIT_PAT`. Map it once into a derived secrets file (the rename keeps the
PAT out of your shell history and the new file readable only by you):

```bash
# Strip the host's stale tokens out of the .env passthrough, then append a
# fresh GITHUB_TOKEN from the local gh auth.
{
  grep -vE '^(GITHUB_TOKEN|GIT_PAT)=' .env
  printf 'GITHUB_TOKEN=%s\n' "$(gh auth token)"
} > /tmp/act.secrets
chmod 600 /tmp/act.secrets

act <event> --secret-file /tmp/act.secrets -W .github/workflows/<file>
```

The `.env`'s `GIT_PAT` may be stale (rotated, expired, or set for the wrong
account); `gh auth token` is always live and matches the identity in
`gh auth status`. Use either, but expect a `gh: Bad credentials (HTTP 401)`
from the workflow if the token is expired.

### Gotcha: `-s GITHUB_TOKEN=…` does *not* always propagate

`act -s KEY=value` and `act --secret-file …` look interchangeable but aren't
for `GITHUB_TOKEN` specifically. The reliable pattern is to put the line in
the `--secret-file`. A `-s GITHUB_TOKEN=…` alone has been observed to leave
`secrets.GITHUB_TOKEN` empty inside the step's env, so steps like
`gh: To use GitHub CLI in a GitHub Actions workflow, set the GH_TOKEN environment variable` fire even though the flag was passed.

## Working tree: copy vs `--bind`

`act` resolves `actions/checkout` by copying or bind-mounting the host's
working directory into the runner container. Two consequences:

- **Default (`docker cp`).** The container sees the file tree but no `.git`
  directory of its own. Anything that calls `git rev-parse HEAD`,
  `git log`, or [`tj-actions/changed-files`](https://github.com/tj-actions/changed-files)
  fails inside the runner.
- **`--bind`.** The current working directory is bind-mounted, including
  `.git`. Git commands inside the runner read host state. Use this whenever
  the workflow needs real git history.

### Worktree caveat

In a `git worktree`, the `.git` entry is a *file* pointing at
`/path/to/main/.git/worktrees/<name>`. Plain `--bind` only mounts the worktree
directory — the gitdir target outside it doesn't resolve, and `git` inside the
container still says "not a git repository".

Two ways out:

1. **Run from the canonical clone**, not a worktree. `.git` is a real dir,
   `--bind` mounts it as part of the workdir, and everything works:

   ```bash
   cd /path/to/synth-setter           # canonical clone, not a worktree
   act pull_request --bind -W .github/workflows/code-quality-pr.yaml \
     --eventpath /tmp/pr-event.json --secret-file /tmp/act.secrets
   ```

2. **Bind-mount the gitdir target explicitly** if you must stay in a worktree:

   ```bash
   act --bind \
     --container-options "-v /path/to/main/.git:/path/to/main/.git:ro" \
     <event> -W <file>
   ```

## Synthetic PR events

`pull_request` and `workflow_run` workflows read `github.event.*` fields.
`act` doesn't synthesize them; supply an event payload yourself.

Minimum viable `pull_request` payload (real SHAs from your branch so
`changed-files` and diff-driven steps work):

```bash
cat > /tmp/pr-event.json <<EOF
{
  "action": "synchronize",
  "pull_request": {
    "number": 907,
    "base": { "sha": "$(git rev-parse origin/main)", "ref": "main" },
    "head": { "sha": "$(git rev-parse HEAD)",         "ref": "$(git branch --show-current)" }
  },
  "repository": {
    "name": "synth-setter",
    "full_name": "tinaudio/synth-setter",
    "owner": { "login": "tinaudio" }
  }
}
EOF

act pull_request --bind --eventpath /tmp/pr-event.json \
  --secret-file /tmp/act.secrets -W .github/workflows/code-quality-pr.yaml
```

For workflows that hit `gh api` against a *real* PR (e.g., `pr-metadata-gate`),
set `pull_request.number` to the live PR number — the workflow will fetch the
rest from the API.

## Custom runner image for `gh`-using workflows

The medium `catthehacker/ubuntu:act-latest` does **not** ship `gh`. Real
GitHub-hosted runners do. Workflows that call `gh api`, `gh pr view`, etc.
fail with `gh: command not found` out of the box.

A thin Dockerfile that layers `gh` onto the medium image is checked in at
[`act-runner.Dockerfile`](./act-runner.Dockerfile). Build it once:

```bash
docker build -f docs/operations/act-runner.Dockerfile -t act-runner-gh:latest docs/operations
```

Then reference it for workflows that need `gh`:

```bash
act pull_request --bind \
  -W .github/workflows/pr-metadata-gate.yaml \
  --eventpath /tmp/pr-event.json \
  --secret-file /tmp/act.secrets \
  -P ubuntu-latest=act-runner-gh:latest \
  --pull=false
```

`--pull=false` keeps `act` from trying to `docker pull` the locally-built tag.
You can also hard-code the override in `~/.config/act/actrc` if `gh`-using
workflows are the common case for you.

If the build fails with `containerd.sock: timeout` against the BuildKit
backend, fall back to the classic builder: `DOCKER_BUILDKIT=0 docker build …`.

## Examples

### `code-quality-pr.yaml`

Pre-commit on changed files. Heavy with action installs but reusable thanks to
`actions/cache`:

```bash
cd /path/to/synth-setter
act pull_request --bind \
  -W .github/workflows/code-quality-pr.yaml \
  --eventpath /tmp/pr-event.json \
  --secret-file /tmp/act.secrets
```

`tj-actions/changed-files` needs `--bind` and a canonical clone (not a
worktree) — otherwise the `.git` resolution problem above bites and the
"changed files" list comes back empty.

### `pr-metadata-gate.yaml`

Needs `gh` + `GITHUB_TOKEN` + a live PR number. Use the custom image and a
`pull_request` event whose `pull_request.number` is real:

```bash
act pull_request --bind \
  -W .github/workflows/pr-metadata-gate.yaml \
  --eventpath /tmp/pr-event.json \
  --secret-file /tmp/act.secrets \
  -P ubuntu-latest=act-runner-gh:latest \
  --pull=false
```

A successful local run prints `✓ Linked issue found`, taxonomy checks for
each linked issue, and `✓ All linked issues trace to an Epic`.

### `check-auth.yml`

Validates that each provider's credentials work. The workflow short-circuits
when `SKYPILOT_API_SERVER_ENDPOINT` is set (the remote API server owns provider
creds), so the local run primarily exercises the `pip install`, `apt install`,
and skip-path. Narrow to one matrix entry to keep run time bounded:

```bash
act workflow_dispatch \
  -W .github/workflows/check-auth.yml \
  --matrix provider:local \
  --secret-file /tmp/act.secrets
```

### Dry-run any workflow

`-n` validates the workflow without creating containers. Cheap parse check
when iterating on YAML:

```bash
act <event> -W .github/workflows/<file> -n
```

## Known limitations

| Limitation                                                                                                                                                                        | Workaround                                                                                                                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `runs-on: macos-*` (used by `test.yml` `run_tests_macos` and `test-mps.yml`). `act` has no macOS images.                                                                          | Skip locally; rely on CI for the Apple-Silicon paths. Or remap to Linux: `-P macos-latest=-self-hosted` (jobs that compile against macOS-only deps will fail at the next step). |
| `workflow_call` reusable workflows (`generate-dataset-shards.yaml`, `validate-dataset-shards.yaml`, `spec-materialization.yml`). `act` does not trigger `workflow_call` directly. | Run the *caller* (e.g., `test-spec-materialization.yml`) so the reusable is invoked through the normal job graph.                                                               |
| `gh` missing from default image.                                                                                                                                                  | Use the [`act-runner-gh`](#custom-runner-image-for-gh-using-workflows) image.                                                                                                   |
| Worktree `.git` is a gitfile, not a directory.                                                                                                                                    | Run from the canonical clone, or bind-mount the gitdir target via `--container-options`.                                                                                        |
| Stale `act` containers from a previous Ctrl-C accumulate and can deadlock new runs.                                                                                               | `docker ps -aq --filter "name=^act-" \| xargs -r docker rm -f`                                                                                                                  |
| Heavy first-run image pull (`act-latest` ≈ 2.3 GB; `forcePull=true` is the default).                                                                                              | After the first run, pass `--pull=false` to skip re-pulls.                                                                                                                      |
| `act` redacts substrings of any loaded secret in log output (`/home/build` → `/home/build/***` if `build` appears in a secret).                                                   | Cosmetic only — execution is unaffected. To make logs readable, narrow the secrets file to just what the workflow uses.                                                         |

## Where to look when something breaks

1. Re-run with `-v` (verbose) — surfaces the docker commands `act` issues and
   the per-step env it builds.
2. Re-run with `--reuse` — keeps the container around after the job ends so
   you can `docker exec -it <act-…> bash` and poke at filesystem / installed
   tools.
3. Compare to the real CI run on github.com. If a step passes there and fails
   under `act`, the diff is usually one of: image contents (gh, gcc, etc.),
   `.git` visibility, secrets reaching the step, or `github.event.*` shape.
4. For credential failures, validate the token outside `act` first:
   `GH_TOKEN=$(gh auth token) gh api repos/tinaudio/synth-setter`. Stale PATs
   in `.env` are the most common cause.

## Related docs

- [`docs/reference/github-actions.md`](../reference/github-actions.md) — what
  each workflow does, its triggers, and its cross-workflow dependencies.
- [`docs/operations/credential-rotation-guide.md`](./credential-rotation-guide.md) —
  what every secret is for and how to rotate it.
