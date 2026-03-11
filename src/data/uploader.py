"""Dataset uploader interface and implementations.

The DatasetUploader protocol defines the interface for uploading a local
directory to a remote location. Two concrete implementations are provided:

- RcloneUploader: uploads to Cloudflare R2 (or any rclone remote) via the
  rclone CLI. Used in production Docker images.
- LocalFakeUploader: copies to a local directory. Used in tests and CI where
  real R2 credentials are not available.
"""

import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DatasetUploader(Protocol):
    """Uploads a local directory to a remote destination."""

    def upload(self, local_dir: Path, remote_path: str) -> None:
        """Upload all files in local_dir to the remote path.

        Args:
            local_dir: Local directory containing dataset files to upload.
            remote_path: Destination path on the remote (e.g.
                "runs/surge_xt/abc1234"). The concrete implementation decides
                how to prefix this (e.g. rclone remote name + bucket).
        """
        ...


class RcloneUploader:
    """Uploads via rclone to a Cloudflare R2 remote.

    Requires rclone to be installed and configured with a remote named "r2"
    (see docker/ubuntu22_04/Dockerfile for how this is baked in at build time).

    Args:
        bucket: R2 bucket name (e.g. "my-bucket"). Reads from the R2_BUCKET
            environment variable if not supplied.
        rclone_remote: Name of the rclone remote. Defaults to "r2".
        dry_run: If True, passes --dry-run to rclone (useful for CI checks).
    """

    def __init__(
        self,
        bucket: str,
        rclone_remote: str = "r2",
        dry_run: bool = False,
    ) -> None:
        self.bucket = bucket
        self.rclone_remote = rclone_remote
        self.dry_run = dry_run

    def upload(self, local_dir: Path, remote_path: str) -> None:
        """Copy local_dir to r2:<bucket>/<remote_path> using rclone with checksum verification."""
        destination = f"{self.rclone_remote}:{self.bucket}/{remote_path}"
        cmd = [
            "rclone",
            "copy",
            str(local_dir),
            destination,
            "--progress",
            "--checksum",
            "--transfers",
            "200",
            "--checkers",
            "200",
        ]
        if self.dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd, check=True)  # nosec B603


class LocalFakeUploader:
    """Copies files to a local directory instead of uploading.

    Intended for use in unit tests and CI where real R2 credentials are not
    available. The dest_dir acts as a fake "bucket root": files are copied to
    dest_dir / remote_path, mirroring the R2 path structure.

    Args:
        dest_dir: Root directory to copy files into (analogous to the R2 bucket
            root). Created if it does not exist.
    """

    def __init__(self, dest_dir: Path) -> None:
        self.dest_dir = Path(dest_dir)

    def upload(self, local_dir: Path, remote_path: str) -> None:
        """Copy all files from local_dir into dest_dir/remote_path, mirroring the R2 path."""
        destination = self.dest_dir / remote_path
        destination.mkdir(parents=True, exist_ok=True)
        for src_file in Path(local_dir).iterdir():
            if src_file.is_file():
                shutil.copy2(src_file, destination / src_file.name)
