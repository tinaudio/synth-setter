"""Stop RunPod pods from a shard generation run.

Emergency kill switch for when something goes wrong with many pods.
Filters pods by the shardgen-<run_id> naming convention used by
runpod_launch.py.

Requires:
    pip install runpod click
    export RUNPOD_API_KEY=<your-key>

Usage:
    # Stop all pods from a specific run
    python scripts/runpod_stop.py --run-id 20260310-143022-a3f2b1

    # Stop ALL shardgen pods
    python scripts/runpod_stop.py --all
"""

import os
import sys

import click

try:
    import runpod
except ImportError:
    runpod = None  # type: ignore[assignment]


def _stop_pods(name_prefix: str) -> int:
    """Terminate all pods whose name starts with name_prefix.

    Returns count.
    """
    pods = runpod.get_pods()
    matched = [p for p in pods if p.get("name", "").startswith(name_prefix)]

    if not matched:
        print(f"No pods found matching prefix '{name_prefix}'.")
        return 0

    print(f"Found {len(matched)} pod(s) matching '{name_prefix}':")
    for pod in matched:
        pid = pod["id"]
        name = pod.get("name", "")
        status = pod.get("desiredStatus", "unknown")
        print(f"  {pid} ({name}) — status: {status}")

    print()
    for pod in matched:
        pid = pod["id"]
        try:
            runpod.terminate_pod(pid)
            print(f"  Terminated: {pid}")
        except Exception as e:
            print(f"  ERROR terminating {pid}: {e}")

    return len(matched)


@click.command()
@click.option("--run-id", default=None, help="Run ID to filter pods by.")
@click.option("--all", "stop_all", is_flag=True, default=False, help="Stop ALL shardgen pods.")
def main(run_id: str | None, stop_all: bool) -> None:
    """Stop RunPod pods from a shard generation run."""
    if runpod is None:
        raise click.UsageError("runpod package not installed. Install it with: pip install runpod")
    if not os.environ.get("RUNPOD_API_KEY"):
        raise click.UsageError("RUNPOD_API_KEY environment variable is required.")

    if not run_id and not stop_all:
        raise click.UsageError("Provide --run-id <id> or --all.")

    if stop_all:
        prefix = "shardgen-"
    else:
        prefix = f"shardgen-{run_id[:12]}"

    count = _stop_pods(prefix)
    print(f"\nTerminated {count} pod(s).")


if __name__ == "__main__":
    main()
