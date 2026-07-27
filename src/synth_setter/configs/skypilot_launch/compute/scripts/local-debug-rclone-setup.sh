#!/usr/bin/env bash
# Install rclone in setup so the kind probe doesn't depend on a custom image
# (SkyPilot's stock kubernetes image ships python + sudo + sshd only).
set -euo pipefail
if ! command -v rclone >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq rclone
fi
rclone --version
