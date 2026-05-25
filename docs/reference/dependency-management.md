# Dependency management

Single source of truth for how dependencies install in this project, why the
command differs by hardware, and how to keep the committed `uv.lock` honest.

## Command surface

| Target                               | Command                                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| macOS (Apple Silicon, MPS)           | `uv sync --frozen`                                             |
| Linux GPU box (CUDA 12.8)            | `uv sync --frozen --extra cu128`                               |
| Linux CPU-only — laptop / CI runner  | `uv sync --frozen --extra cpu --no-default-groups --group dev` |
| Lint / type-check only (no torch)    | `uv sync --frozen --only-group dev`                            |
| Verify the lock is in sync           | `uv lock --check`                                              |
| Regenerate the lock after a dep edit | `uv lock` (then commit the diff)                               |

`--frozen` errors instead of silently re-resolving when the lock and
`pyproject.toml` disagree. CI uses it for the main project install everywhere
it can.

`.github/workflows/uv-lock-check.yml` runs `uv lock --check` on macOS + Linux
for every PR that touches `pyproject.toml` or `uv.lock`; a stale lock is
what it catches.

## Mac: no backend flag

A bare `uv sync --frozen` on Apple Silicon resolves torch from PyPI's
MPS-capable wheel. **Do not pass `--extra cpu` or `--extra cu128` on a
Mac** — the marker in `[tool.uv.sources]` excludes both on macOS, so the
flag is a silent no-op that confuses readers. The `cpu`/`cu128` extras
are Linux/Windows backend selectors.

Intel Macs resolve to a CPU-only torch from the same wheel (no MPS hardware).

## CUDA is the source of truth for reported numbers

MPS results need not match CPU/CUDA bit-for-bit. Some ops fall back to CPU
under MPS; others raise "operator not currently implemented for the MPS
device" unless `PYTORCH_ENABLE_MPS_FALLBACK=1` is set. When a Mac
collaborator sees a small numerical divergence from a Linux-CUDA run on the
same input, that's MPS divergence — not a reproducibility bug to chase.

Gate any test that hits an MPS-incompatible op behind the `mps` pytest
marker so CI can route it to the macOS runner.

## The resync flip (Linux footgun)

Two `uv sync` behaviours combine into a Linux footgun:

1. **Exact sync**: every invocation removes/replaces packages not matching
   the current resolution.
2. **Non-sticky extras**: `--extra cpu` applies only to the command it's
   passed to; the next `uv` invocation has no memory of it.

So any later `uv` command without `--extra cpu` re-resolves torch to CUDA
and exact-syncs the CPU wheel out, pulling multi-GB CUDA wheels onto a
GPU-less box. The wrong env often still imports (no GPU is not an import
error), so tests can pass silently on the bloated install.

The trap is **Linux-only**. macOS is immune (markers exclude both extras).

How to avoid it on Linux:

- CI runners are ephemeral per-run — they survive the flip because there is
  no "later command" to flip them. This is why CI doesn't set `UV_EXTRA`.
- **Never run `uv add` or `uv lock` inside a `--extra cpu` environment.** Do
  all dependency edits and relocks on a Mac or GPU box where the default
  resolution is the one we want.
- If you are running interactively on a GPU-less Linux box, export
  `UV_EXTRA=cpu` in the shell so every `uv` invocation inherits the flag.

## Relock cadence

`uv.lock` is committed and reviewed like code. Run `uv lock` (or
`uv lock --upgrade`) when you:

- Add, remove, or rebound a dependency in `pyproject.toml`.
- Want to pick up a security fix or bug fix from upstream.
- See `uv lock --check` fail in CI on a PR you didn't expect to bump deps.

**Regenerate on a Mac or GPU box, not on a `--extra cpu` Linux env.** A lock
made under `--extra cpu` will pin the CPU torch as the default-resolution
branch and the next bare `uv sync` from a GPU box will fight it.

**Review the lockfile diff in PRs** — especially torch, CUDA, and native
wheels (`pedalboard`, `librosa`/`soundfile`, `h5py`/`hdf5plugin`). A
lockfile change can move a result. The `uv-lock-check.yml` matrix can only
prove the lock is consistent with `pyproject.toml`, not that the new pins
are the ones you wanted.

## Backward-compat shims

`pyproject.toml` retains `[project.optional-dependencies]` entries for
`torch`, `dev`, `docs`, and `all`. These exist so `pip install -e ".[torch,dev]"`
keeps working for callsites that cannot honor `[tool.uv.sources]`, including:

- `Makefile`'s `make install` target.
- `environment.yaml` (Conda envs).
- `scripts/sync_worker_checkout.sh` (SkyPilot worker setup).
- `.github/workflows/docs.yml` (mkdocs build).
- `.github/workflows/test-dataset-finalization.yml` (`pip install -e ".[torch,dev]"` smoke).

The shims do not produce a lockfile-validated install; they re-resolve every
time. Prefer `uv sync --frozen` for any new install path. Removal of the
shims tracks alongside migration of the callsites above.

## Adding a new extra or dependency group

If a workflow needs a slim install closure (e.g. lint-only, validator-only),
add a `[dependency-groups]` group rather than a `[project.optional-dependencies]`
extra. Groups are PEP 735 — private to `uv sync`, not published in the
package's metadata.

If the closure is publicly-installable (e.g. `synth-setter[cloud]` for a
hypothetical cloud-only optional dep set), use `[project.optional-dependencies]`.

After either edit: run `uv lock`, review the diff, and commit `pyproject.toml`
and `uv.lock` in the same commit.

## Open questions

- **GPU fleet CUDA generations.** The lock pins `cu128` wheels via
  `pytorch-cu128`. CUDA wheels are forward-compatible with newer drivers;
  the failure mode is a box whose driver is *older* than cu128. If the
  fleet ever spans multiple toolkit generations, the fix is a second routed
  extra (e.g. `cu118`), not a revert to `uv pip install --torch-backend`.
  Tracked under #1243's open questions.

## Related

- [`pyproject.toml`](../../pyproject.toml) — the canonical `[tool.uv.sources]`,
  `[tool.uv].conflicts`, and `[dependency-groups]` definitions.
- [`uv.lock`](../../uv.lock) — the universal lockfile (macOS + Linux + Windows).
- [`.github/workflows/uv-lock-check.yml`](../../.github/workflows/uv-lock-check.yml) —
  the per-OS lock-sync gate.
- [`docs/getting-started.md`](../getting-started.md) — first-install
  walkthrough.
