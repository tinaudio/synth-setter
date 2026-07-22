#!/bin/bash
# Refresh package indexes before running an apt-dependent command.

set -euo pipefail

apt-get update
"$@"
