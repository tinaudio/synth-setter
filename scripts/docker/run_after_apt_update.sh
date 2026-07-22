#!/bin/bash
# Prevent commands from running after a failed or partial apt index refresh.

set -euo pipefail

apt-get update --error-on=any
"$@"
