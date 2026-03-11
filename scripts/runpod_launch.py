"""Launch RunPod pods for massively parallel shard generation.

Launches N pods on RunPod, each running MODE=generate-shards to produce M
shards. All pods upload to a shared R2 prefix, resulting in N*M total shards
under runs/<run_id>/.

Each pod auto-derives its instance_id from RUNPOD_POD_ID, so shard filenames
are globally unique across all pods.

Requires:
    pip install runpod click
    export RUNPOD_API_KEY=<your-key>

Usage:
    python scripts/runpod_launch.py \\
        --num-pods 50 --shards-per-pod 10 --shard-size 10000 \\
        --image tinaudio/perm:dev-snapshot-abc1234

    # Dry run (print what would be launched)
    python scripts/runpod_launch.py --dry-run \\
        --num-pods 50 --shards-per-pod 10 --shard-size 10000 \\
        --image tinaudio/perm:dev-snapshot-abc1234
"""

import os
import sys
import time
import uuid
from datetime import datetime, timezone

import click

try:
    import runpod
except ImportError:
    runpod = None  # type: ignore[assignment]


def _make_run_id() -> str:
    """Generate a unique run ID: YYYYMMDD-HHMMSS-<6 hex chars>."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _make_pod_env(
    run_id: str,
    shards_per_pod: int,
    shard_size: int,
    param_spec: str,
    max_workers: int | None,
) -> dict[str, str]:
    """Build the env dict passed to each RunPod pod."""
    env = {
        "IDLE_AFTER": "0",
        "MODE": "generate-shards",
        "NUM_SHARDS": str(shards_per_pod),
        "PARALLEL": "1",
        "PARAM_SPEC": param_spec,
        "R2_PREFIX": f"runs/{run_id}",
        "SHARD_SIZE": str(shard_size),
    }
    if max_workers is not None:
        env["MAX_WORKERS"] = str(max_workers)
    return env


def _launch_pods(
    num_pods: int,
    env: dict[str, str],
    image: str,
    gpu_type: str,
    cloud_type: str,
    run_id: str,
    volume_size: int,
) -> list[str]:
    """Launch num_pods RunPod pods and return their IDs."""
    pod_ids = []
    for i in range(num_pods):
        pod = runpod.create_pod(
            name=f"shardgen-{run_id[:12]}-{i:03d}",
            image_name=image,
            gpu_type_id=gpu_type,
            cloud_type=cloud_type,
            volume_in_gb=volume_size,
            env=env,
        )
        pod_id = pod["id"]
        pod_ids.append(pod_id)
        print(f"  Launched pod {i + 1}/{num_pods}: {pod_id}")
    return pod_ids


def _poll_pods(pod_ids: list[str], poll_interval: int = 30) -> tuple[set[str], set[str]]:
    """Poll pods until all have exited.

    Returns (completed, failed) sets.
    """
    completed: set[str] = set()
    failed: set[str] = set()
    total = len(pod_ids)

    while len(completed) + len(failed) < total:
        time.sleep(poll_interval)
        for pid in pod_ids:
            if pid in completed or pid in failed:
                continue
            try:
                pod = runpod.get_pod(pid)
            except Exception as e:
                print(f"  WARNING: Could not fetch status for {pid}: {e}")
                continue

            desired = pod.get("desiredStatus")
            if desired == "EXITED":
                completed.add(pid)
                print(f"  Pod {pid}: COMPLETED")
            elif desired in ("TERMINATED", "ERROR"):
                failed.add(pid)
                print(f"  Pod {pid}: FAILED ({desired})")

        running = total - len(completed) - len(failed)
        print(f"  Progress: {len(completed)} done, {len(failed)} failed, " f"{running} running")

    return completed, failed


@click.command()
@click.option("--num-pods", "-n", type=int, required=True, help="Number of RunPod pods to launch.")
@click.option(
    "--shards-per-pod",
    "-s",
    type=int,
    required=True,
    help="Number of shards each pod generates.",
)
@click.option(
    "--shard-size", type=int, default=10000, show_default=True, help="Samples per shard."
)
@click.option(
    "--image",
    required=True,
    help="Docker image to run (e.g. tinaudio/perm:dev-snapshot-abc1234).",
)
@click.option(
    "--gpu-type",
    default="NVIDIA RTX A5000",
    show_default=True,
    help="RunPod GPU type ID.",
)
@click.option(
    "--run-id",
    default=None,
    help="Run ID for grouping pods (default: auto-generated timestamp+uuid).",
)
@click.option(
    "--param-spec",
    default="surge_simple",
    show_default=True,
    help="Param spec name.",
)
@click.option(
    "--cloud-type",
    default="COMMUNITY",
    type=click.Choice(["COMMUNITY", "SECURE"]),
    show_default=True,
    help="RunPod cloud type.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Cap on concurrent workers per pod (default: auto from CPU count).",
)
@click.option(
    "--volume-size",
    type=int,
    default=50,
    show_default=True,
    help="Temporary volume size in GB per pod.",
)
@click.option(
    "--poll-interval",
    type=int,
    default=30,
    show_default=True,
    help="Seconds between status polls.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be launched without creating pods.",
)
def main(
    num_pods: int,
    shards_per_pod: int,
    shard_size: int,
    image: str,
    gpu_type: str,
    run_id: str | None,
    param_spec: str,
    cloud_type: str,
    max_workers: int | None,
    volume_size: int,
    poll_interval: int,
    dry_run: bool,
) -> None:
    """Launch RunPod pods for massively parallel shard generation."""
    if runpod is None:
        raise click.UsageError("runpod package not installed. Install it with: pip install runpod")
    if not os.environ.get("RUNPOD_API_KEY"):
        raise click.UsageError(
            "RUNPOD_API_KEY environment variable is required.\n"
            "Get your API key from https://www.runpod.io/console/user/settings"
        )

    run_id = run_id or _make_run_id()
    env = _make_pod_env(run_id, shards_per_pod, shard_size, param_spec, max_workers)
    total_shards = num_pods * shards_per_pod
    total_samples = total_shards * shard_size

    print("=== RunPod Shard Generation ===")
    print(f"  Run ID        : {run_id}")
    print(f"  Image         : {image}")
    print(f"  GPU type      : {gpu_type}")
    print(f"  Cloud type    : {cloud_type}")
    print(f"  Pods          : {num_pods}")
    print(f"  Shards/pod    : {shards_per_pod}")
    print(f"  Shard size    : {shard_size:,} samples")
    print(f"  Total shards  : {total_shards:,}")
    print(f"  Total samples : {total_samples:,}")
    print(f"  R2 prefix     : runs/{run_id}")
    print(f"  Volume size   : {volume_size} GB")
    print()

    if dry_run:
        print("[DRY RUN] Would launch pods with env:")
        for k, v in sorted(env.items()):
            print(f"  {k}={v}")
        print()
        print("[DRY RUN] No pods created.")
        return

    print("Launching pods...")
    pod_ids = _launch_pods(num_pods, env, image, gpu_type, cloud_type, run_id, volume_size)

    print(f"\nWaiting for {len(pod_ids)} pods to complete...")
    completed, failed = _poll_pods(pod_ids, poll_interval)

    print(f"\n=== Run Complete: {run_id} ===")
    print(f"  Completed : {len(completed)}/{len(pod_ids)}")
    print(f"  Failed    : {len(failed)}/{len(pod_ids)}")
    print(f"  R2 path   : runs/{run_id}/")
    if failed:
        print(f"  Failed pod IDs: {sorted(failed)}")
        print(
            f"\n  To retry failed pods, re-run with --num-pods {len(failed)} " f"--run-id {run_id}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
