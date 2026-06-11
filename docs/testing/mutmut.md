# Mutation Testing (mutmut)

`make mutmut` runs [mutmut](https://github.com/boxed/mutmut) v3 against the
modules listed under `[tool.mutmut].paths_to_mutate` in `pyproject.toml`.
This is the authoritative entry point — the CI workflow
`.github/workflows/mutmut.yml` runs it on Linux (`workflow_dispatch` + weekly
cron).

## Requires Python 3.11+

mutmut parses the whole `pyproject.toml`. On Python 3.10 it falls back to the
legacy `toml` library, which crashes (`IndexError` in `load_array`) on the
PEP 735 `dev` group's mixed string/inline-table arrays; 3.11+ uses stdlib
`tomllib`, which parses it cleanly ([#1414](https://github.com/tinaudio/synth-setter/issues/1414)).
Run `make mutmut` under a 3.11+ interpreter — the CI workflow pins 3.11 for
this reason.

## `also_copy` covers the full package

mutmut copies the paths under `paths_to_mutate` into a `mutants/` sandbox and
strips the real `src/` off `sys.path`. Any module a test imports transitively
must come along via `also_copy`. `also_copy = ["src/synth_setter/"]` therefore
includes the *whole* package, not just the mutated subdirs.

Only revisit the `[tool.mutmut]` config when adding a *new* top-level mutate
path under `src/`. All other `synth_setter.*` imports are covered
automatically — `also_copy` already pulls in any module added outside
`paths_to_mutate`.

## Keep mutmut-target tests in-process

mutmut drives subprocesses via `os.fork()` and runs tests under
`MUTANT_UNDER_TEST=<paths_to_mutate-entry>`. Tests that shell out to
`python -m …` inherit that env var and crash mutmut's trampoline
(`mutmut.config is None` in any fresh interpreter).

Use the parser directly with `capsys` / `CliRunner` (argparse / click) rather
than `subprocess.run([sys.executable, "-m", …])`. The reference test is
`test_cli_help_advertises_mask_degenerate_bins_flag` in
`tests/pipeline/data/test_stats.py`.

## macOS gotcha

On macOS, the parent process imports `torch` / `h5py` / `hydra` from
`tests/conftest.py` during stats collection. Apple's fork-safety check then
SIGSEGVs every forked child, so local `make mutmut` runs report mass segfaults
that don't reflect real test outcomes.

The Makefile target sets `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` to soften
this. The authoritative end-to-end run is the Linux CI job.
