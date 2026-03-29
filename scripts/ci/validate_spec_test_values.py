#!/usr/bin/env python3
"""Assert test-specific values in a materialized spec from ci-materialize-test.yaml.

This script is tightly coupled to configs/dataset/ci-materialize-test.yaml:   num_shards=3,
base_seed=42, splits 1/1/1, param_spec=surge_simple,   sample_rate=16000, shard_size=32,
velocity=100.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    """Validate test-specific values and print results."""
    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} <spec.json>\n")
        sys.exit(1)

    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())

    # 3 shards (ci-materialize-test.yaml: num_shards=3, splits 1/1/1)
    assert len(spec["shards"]) == 3, f"Expected 3 shards, got {len(spec['shards'])}"
    sys.stdout.write(f"  num_shards: {len(spec['shards'])} (expected 3)\n")

    # Deterministic seeds: base_seed=42 + shard_id
    seeds = [s["seed"] for s in spec["shards"]]
    assert seeds == [42, 43, 44], f"Expected seeds [42, 43, 44], got {seeds}"
    sys.stdout.write(f"  seeds: {seeds} (expected [42, 43, 44])\n")

    # Zero-padded filenames
    filenames = [s["filename"] for s in spec["shards"]]
    expected_filenames = ["shard-000000.h5", "shard-000001.h5", "shard-000002.h5"]
    assert filenames == expected_filenames, f"Bad filenames: {filenames}"
    sys.stdout.write(f"  filenames: {filenames}\n")

    # Config passthrough fields
    assert spec["param_spec"] == "surge_simple", f"Bad param_spec: {spec['param_spec']}"
    assert spec["sample_rate"] == 16000, f"Bad sample_rate: {spec['sample_rate']}"
    assert spec["shard_size"] == 32, f"Bad shard_size: {spec['shard_size']}"
    assert spec["base_seed"] == 42, f"Bad base_seed: {spec['base_seed']}"
    assert spec["velocity"] == 100, f"Bad velocity: {spec['velocity']}"
    sys.stdout.write("  config passthrough: all correct\n")

    sys.stdout.write("All test assertions passed.\n")


if __name__ == "__main__":
    main()
