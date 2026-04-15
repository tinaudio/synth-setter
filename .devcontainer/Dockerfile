FROM tinaudio/synth-setter:dev-snapshot

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Non-root dev user with bash as the login shell. No sudo: passwordless
# sudo is security theatre (dev becomes root trivially), and the escape
# hatch for missing system packages is adding them to the base image
# (docker/ubuntu22_04/Dockerfile) and rebuilding.
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash -m $USERNAME

USER $USERNAME

# Claude Code CLI (userspace install via nvm — no root, no sudo).
# Anthropic's reference devcontainer installs Claude Code globally as root
# and adds a scoped sudoers grant for a firewall script. We install under
# the dev user's home instead so this Dockerfile stays root-free. Firewall
# + scoped sudo are tracked separately (#549).
ARG NVM_VERSION=0.40.1
ARG NODE_VERSION=20.18.0
ARG CLAUDE_CODE_VERSION=2.1.105

ENV NVM_DIR="/home/$USERNAME/.nvm"
ENV PATH="$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH"

# nvm.sh is intended for bash/zsh. It happens to source under Ubuntu's
# /bin/sh (dash) today because nvm maintains POSIX compatibility, but the
# dependency on bash is implicit and could break if nvm's internals change.
# Set SHELL explicitly so the source + nvm install + npm install chain
# runs under bash, and enable pipefail so any stage failure aborts the
# layer cleanly.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Clone nvm at a pinned tag instead of piping install.sh through bash.
# The cloned tree is auditable on disk, the tag is immutable upstream,
# and we never execute unverified network content during the image build.
RUN git clone --depth 1 --branch "v${NVM_VERSION}" https://github.com/nvm-sh/nvm.git "$NVM_DIR" \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install "$NODE_VERSION" \
    && nvm alias default "$NODE_VERSION" \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"
