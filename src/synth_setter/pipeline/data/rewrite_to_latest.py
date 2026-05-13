#!/usr/bin/env python
import concurrent.futures
import os
from pathlib import Path

import click
import h5py


def rewrite_shard(shard: Path, output_dir: Path) -> (Path, bool, str):
    """Rewrites a given HDF5 shard file so that it is created with libver="latest" and writes the
    new file to output_dir.

    Parameters:
        shard (Path): Path to the input HDF5 shard file.
        output_dir (Path): Directory where the new file will be written.

    Returns:
        A tuple (output_file, success, error_message).
    """
    new_file = output_dir / shard.name
    temp_file = output_dir / f"{shard.stem}.temp.h5"

    try:
        with h5py.File(shard, "r") as f_in, h5py.File(temp_file, "w", libver="latest") as f_out:
            # Copy every top-level item (dataset or group) from the input file.
            for key in f_in:
                f_in.copy(key, f_out)
        # Atomically replace the temporary file with the final new file.
        os.replace(temp_file, new_file)
        return (new_file, True, "")
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        return (shard, False, str(e))


@click.command()
@click.argument("input_dir", type=str)
@click.argument("output_dir", type=str)
@click.option(
    "--pattern",
    "-p",
    type=str,
    default="shard-*.h5",
    help="Glob pattern to find shard files in the input directory.",
)
@click.option(
    "--workers", "-w", type=int, default=4, help="Number of worker processes to run in parallel."
)
def main(input_dir, output_dir, pattern, workers):
    """Rewrite each HDF5 file matching the given pattern in INPUT_DIR so that it is created with
    libver="latest" (i.e. with a superblock version >= 3) and write the new files to OUTPUT_DIR."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = list(input_dir.glob(pattern))
    if not shard_files:
        click.echo(f"No files found matching pattern '{pattern}' in {input_dir}.")
        return

    click.echo(
        f"Found {len(shard_files)} file(s). Rewriting with libver='latest' using {workers} workers..."
    )

    # Process the shards in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_shard = {
            executor.submit(rewrite_shard, shard, output_dir): shard for shard in shard_files
        }
        for future in concurrent.futures.as_completed(future_to_shard):
            shard = future_to_shard[future]
            try:
                output_path, success, error = future.result()
                if success:
                    click.echo(f"Successfully rewritten: {output_path}")
                else:
                    click.echo(f"Error rewriting {shard}: {error}")
            except Exception as exc:
                click.echo(f"Unexpected error rewriting {shard}: {exc}")


if __name__ == "__main__":
    main()
