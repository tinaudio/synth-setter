#!/usr/bin/env bash
# Codespace first-run setup. Runs once after the container is created.
# The base image (tinaudio/perm:dev-snapshot) already ships all deps,
# Surge XT, xvfb, rclone, and baked R2/W&B credentials — so this script
# only wires up the workspace checkout to the image's venv.
set -euo pipefail

cd /workspaces/synth-setter

# Skills live in a git submodule (.claude/skills → tinaudio/skills).
git submodule update --init --recursive

# Install the workspace as an editable package into the image's venv.
# --no-deps skips re-downloading the ~2.5GB of deps already in the image.
uv pip install --no-deps -e .

# Pre-commit hooks (pre-commit itself is in the image's deps).
pre-commit install

# plumb writes a native git hook — re-run pre-commit install to chain it.
if command -v plumb >/dev/null 2>&1; then
  plumb init || true
  pre-commit install
fi

echo "Codespace ready. Run 'make test' to verify."
