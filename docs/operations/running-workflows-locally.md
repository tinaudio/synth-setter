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

The repo ships an [`.actrc`](../../.actrc) at the root that maps the runner
labels used by our workflows to the medium [`catthehacker/ubuntu:act-*`](https://github.com/catthehacker/docker_images)
images. `act` picks it up automatically when run from inside the repo, so you
don't need to create a per-user `~/.config/act/actrc`. The committed mappings
cover `ubuntu-latest`, `ubuntu-latest-4core` (used by `cpu-slow`,
`docker-build-validation`, `test-vst-slow`), `ubuntu-22.04`, and
`ubuntu-20.04`. Without that mapping, `act` prompts on first run for a
default image size and dies on EOF in a non-TTY (CI containers, dev
container `bash -c`, ssh-without-tty).

The full GitHub-runner clone (`full-latest`, ~60 GB) is rarely necessary;
the medium images (~2.3 GB) cover everything except a handful of
GitHub-only tools. The [Custom runner image](#custom-runner-image-for-gh-using-workflows)
section below shows how to layer `gh` on top, which is the only common gap.

The [`test-act.yaml`](../../.github/workflows/test-act.yaml) workflow guards
this setup: any PR that touches `.actrc`, the runner Dockerfile, or the
test-act workflow itself runs `act -l`, an `act -n` dry-run, and a real
`act` run of `test-dataset-generation.yml` with `--input provider=local`
on a GitHub runner — catching config rot, dynamic-matrix breakage, and
docker-in-runner failures at PR time instead of on someone's laptop.
Fork PRs skip the real-run step (R2 secrets unavailable to forks).

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

### Supplying `GITHUB_TOKEN`

Workflows reference `secrets.GITHUB_TOKEN`. `.env.example` doesn't define that
name — the closest tokens it documents are `RESTRICTED_AGENT_GIT_PAT` (a
narrow-scope PAT for the dev container's agent) and any personal PAT a
developer keeps locally.

For local `act` runs, prefer the host's live `gh` auth — always current,
no `.env` rename, no copy/paste:

```bash
# Build a derived secrets file with whatever the .env contributes,
# plus a live GITHUB_TOKEN from the host's gh auth.
{
  grep -vE '^GITHUB_TOKEN=' .env
  printf 'GITHUB_TOKEN=%s\n' "$(gh auth token)"
} > /tmp/act.secrets
chmod 600 /tmp/act.secrets

act <event> --secret-file /tmp/act.secrets -W .github/workflows/<file>
```

If you prefer a PAT from `.env` (for example, `RESTRICTED_AGENT_GIT_PAT`), map
it by name into the derived file instead:

```bash
{
  grep -vE '^GITHUB_TOKEN=' .env
  sed -n 's/^RESTRICTED_AGENT_GIT_PAT=/GITHUB_TOKEN=/p' .env
} > /tmp/act.secrets
chmod 600 /tmp/act.secrets
```

Either way, expect `gh: Bad credentials (HTTP 401)` from the workflow if the
token is expired or doesn't grant access to the repo you're querying.

### Gotcha: bash evaluation order with `-s KEY=$VAR`

`act -s KEY=value` and `act --secret-file ...` work equivalently — *as long as
the shell evaluates `value` correctly at the call site*. The trap is the bash
assignment-on-command-line form:

```bash
# WRONG — $TOKEN expands to empty in the parent shell, so act gets -s GITHUB_TOKEN=
TOKEN="$(gh auth token)" act pull_request -W .github/workflows/pr-metadata-gate.yaml -s "GITHUB_TOKEN=$TOKEN"

# OK — assign first, then reference
TOKEN="$(gh auth token)"
act pull_request -W .github/workflows/pr-metadata-gate.yaml -s "GITHUB_TOKEN=$TOKEN"

# OK — substitute inline so the value is captured before act sees the flag
act pull_request -W .github/workflows/pr-metadata-gate.yaml -s "GITHUB_TOKEN=$(gh auth token)"
```

If `secrets.GITHUB_TOKEN` ends up empty inside the workflow, `gh: To use GitHub CLI in a GitHub Actions workflow, set the GH_TOKEN environment variable` fires from any step that calls `gh`. Check your shell invocation
before blaming `act`.

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
backend, fall back to the classic builder: `DOCKER_BUILDKIT=0 docker build ...`.

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
| `runs-on: gpu-x64` (used by `test-gpu.yml`). `act` can't reproduce the self-hosted GPU runner — no CUDA, no driver, no Docker GPU passthrough by default.                         | Skip locally; rely on CI. If you need to exercise the workflow shape (steps, env), remap to Linux: `-P gpu-x64=catthehacker/ubuntu:act-latest` (GPU-dependent steps will fail). |
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
   you can `docker exec -it <act-...> bash` and poke at filesystem / installed
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
