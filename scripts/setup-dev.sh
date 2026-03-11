#!/usr/bin/env bash
# scripts/setup-dev.sh — single entrypoint for local dev environment setup.
# Works on macOS (Homebrew) and Linux (apt-get).
#
# Usage:
#   bash scripts/setup-dev.sh          # full setup
#   bash scripts/setup-dev.sh --ci     # CI mode: skip Python deps & pre-commit install
set -euo pipefail

CI_MODE=false
if [[ "${1:-}" == "--ci" ]]; then
    CI_MODE=true
fi

need_install() { ! command -v "$1" &>/dev/null; }

is_mac()   { [[ "$(uname)" == "Darwin" ]]; }
has_apt()  { command -v apt-get &>/dev/null; }

install_or_skip() {
    local tool="$1"
    if ! need_install "$tool"; then
        echo "  $tool: already installed"
        return 0
    fi
    return 1
}

# ── Python environment ────────────────────────────────────────────────
if [[ "$CI_MODE" == false ]]; then
    echo "==> Python dependencies"
    python -m pip install --upgrade pip
    pip install -r requirements-torch.txt
    pip install -r requirements-app.txt
    pip install sh  # needed by some tests

    echo "==> pre-commit hooks"
    pip install pre-commit
    pre-commit install
fi

# ── Shell tooling (only bats — shellcheck & checkmake come via pre-commit) ──
echo "==> Shell tools (bats)"

if install_or_skip bats; then :; else
    echo "  Installing bats..."
    if is_mac; then
        brew install bats-core
    elif has_apt; then
        sudo apt-get update -qq && sudo apt-get install -y -qq bats
    else
        echo "  ERROR: cannot install bats — need brew (macOS) or apt (Linux)" >&2
        exit 1
    fi
fi

echo ""
echo "Setup complete. Run 'make format' and 'make test-bash' to verify."
