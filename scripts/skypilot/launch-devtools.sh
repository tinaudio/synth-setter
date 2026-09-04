#!/bin/bash
# Launch a persistent RunPod devtools cluster.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

sky launch "${script_dir}/synth-devtools.yaml" \
  -c synth-devtools-02 \
  -d \
  -r \
  -y
