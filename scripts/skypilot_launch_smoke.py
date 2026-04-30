"""Launch the smoke `generate_dataset` run on RunPod via SkyPilot.

Materializes a `DatasetPipelineSpec` locally from the smoke config, ships the
frozen spec into the worker via `task.update_file_mounts`, forwards the
worker-side env from a `.env.cloud` file via `task.update_envs`, and submits
a SkyPilot managed job that runs the existing container CLI.

This is the first scaffolding under the SkyPilot integration epic (#534).
No `compute_config` schema, no `pipeline generate --backend skypilot` CLI —
those land in the Phase A–C PRs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import click
import sky
import sky.jobs
from dotenv import dotenv_values

from pipeline.schemas.config import dataset_config_id_from_path, load_dataset_config
from pipeline.schemas.spec import materialize_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset" / "ci-smoke-test.yaml"
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "compute" / "runpod-template.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env.cloud"

# Worker-side mount destination. The image's WORKDIR is /home/build/synth-setter (Dockerfile),
# so the spec lands under <repo_root>/data/ on the worker — gitignored, no clash with repo
# files, and portable if the container layout changes later.
WORKER_REPO_ROOT = "/home/build/synth-setter"
WORKER_SPEC_PATH = f"{WORKER_REPO_ROOT}/data/skypilot-launch-smoke-spec.json"

# Local source path. Lives under the system tempdir (not the repo) so it shares a filesystem
# with SkyPilot's staging dir (also under /tmp). Inside the dev-snapshot container the repo
# workspace is bind-mounted from the runner host while /tmp is on the container's overlay —
# crossing those two with `os.rename` raises EXDEV (Errno 18) when SkyPilot stages mounts.
LOCAL_SPEC_PATH = Path(tempfile.gettempdir()) / "skypilot-launch-smoke-spec.json"


def load_worker_env(path: Path) -> dict[str, str]:
    """Read worker-side env from a dotenv file using python-dotenv.

    `dotenv_values` returns a dict whose values are `Optional[str]` (a key with no `=` becomes
    `None`); coerce to a plain `dict[str, str]` for `task.update_envs(...)` and skip None entries.
    """
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Path to a DatasetConfig YAML.",
)
@click.option(
    "--template",
    "template_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_TEMPLATE,
    show_default=True,
    help="Path to the SkyPilot task YAML template.",
)
@click.option(
    "--env-file",
    "env_file_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_ENV_FILE,
    show_default=True,
    help="Path to a KEY=VALUE env file forwarded to the worker.",
)
@click.option(
    "--job-name",
    type=str,
    default=None,
    help="SkyPilot job name (default: synth-setter-smoke-<config_id[:8]>).",
)
def main(
    config_path: Path,
    template_path: Path,
    env_file_path: Path,
    job_name: str | None,
) -> None:
    """Launch the smoke `generate_dataset` run on RunPod via SkyPilot."""
    if not env_file_path.is_file():
        raise click.ClickException(
            f"Worker env file not found: {env_file_path}. "
            f"Copy .env.cloud.example to .env.cloud and fill in values."
        )

    worker_env = load_worker_env(env_file_path)
    if not worker_env:
        raise click.ClickException(f"No env vars parsed from {env_file_path}.")

    config = load_dataset_config(config_path)
    config_id = dataset_config_id_from_path(config_path)
    spec = materialize_spec(config, config_id)

    LOCAL_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_SPEC_PATH.write_text(spec.model_dump_json(indent=2))
    click.echo(f"Materialized spec to {LOCAL_SPEC_PATH}")

    resolved_job_name = job_name or f"synth-setter-smoke-{config_id[:8]}"

    task = sky.Task.from_yaml(str(template_path))
    task.update_envs(worker_env)
    task.update_file_mounts({WORKER_SPEC_PATH: str(LOCAL_SPEC_PATH)})

    click.echo(f"Submitting SkyPilot job: name={resolved_job_name}")
    sky.jobs.launch(task, name=resolved_job_name)


if __name__ == "__main__":
    main()
