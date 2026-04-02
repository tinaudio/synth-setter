# syntax=docker/dockerfile:1.6
ARG BUILD_MODE=source

# release-xt/1.3.4 SHA
ARG SURGE_GIT_REF=f7b97c682ade0b87da85ca5968b63d5c7c98e68d
# Default to main branch
ARG SYNTH_PERMUTATIONS_GIT_REF=main
# ubuntu:22.04 digest retrieved via:
#      docker pull ubuntu:22.04
#      docker inspect ubuntu:22.04 --format='{{index .RepoDigests 0}}'
ARG BASE_IMAGE=ubuntu@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751
# TODO(ktinubu): Rename to something more useful and distrinct from BASE_IMAGE
# TODO(ktinubu): Check for unset TARGETARCH or TARGETPLATFORM. Fallback to amd, linux/amd but add tag to container
# TODO(ktinubu): Add appropriate image labels.
ARG IMAGE=dev-snapshot
# R2 bucket name baked into the image (used by entrypoint + rclone)
ARG R2_BUCKET=

FROM ${BASE_IMAGE} AS amd64-vars
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
WORKDIR /tmp-artifacts
RUN echo "x86_64" > /tmp-artifacts/cmake-arch-name.txt
RUN echo "1.3.4/surge-xt-linux-x64-1.3.4.deb" > /tmp-artifacts/surge-package-filepath.txt

FROM ${BASE_IMAGE} AS arm64-vars
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ARG TARGETARCH
ARG TARGETPLATFORM
ARG BUILD_MODE
RUN echo "TARGETARCH=${TARGETARCH:-<unset>}" >&2
RUN echo "TARGETPLATFORM=${TARGETPLATFORM:-<unset>}" >&2
WORKDIR /tmp-artifacts
RUN echo "aarch64" > /tmp-artifacts/cmake-arch-name.txt
RUN if [ "${BUILD_MODE:-}" = "prebuilt" ] && [ "${TARGETARCH:-}" = "arm64" ]; then \
      echo "ERROR: arm64 prebuilt package is not currently available. Set BUILD_MODE=source." >&2; \
      exit 1; \
    fi
FROM ${BASE_IMAGE} AS arch-vars
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
WORKDIR /artifacts
ARG TARGETARCH
WORKDIR /artifacts

COPY --from=arm64-vars /tmp-artifacts/ /tmp-artifacts/arm64/
COPY --from=amd64-vars /tmp-artifacts/ /tmp-artifacts/amd64/

# `cp` with Trailing dot copies everything inside the source to dest.
RUN case "${TARGETARCH:-}" in \
      arm64) cp -av  /tmp-artifacts/arm64/. /artifacts ;; \
      amd64) cp -av /tmp-artifacts/amd64/. /artifacts ;; \
      *) echo "ERROR: TARGETARCH must be amd64 or arm64 (got '${TARGETARCH:-<unset>}')" >&2; exit 1 ;; \
    esac

# TODO: add appropriate build labels
# TODO: Consider taking python runtime from `python:3.10-*` stage.
# This would involving copying`/usr/local`.
# TODO: Add diagram of stages and dependencies.
# TODO(deduplicate package lists)
# ==========================================
FROM ${BASE_IMAGE} AS builder-base
WORKDIR /home/build/
ENV LANG=C.UTF-8
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Non-interactive configuration of tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
RUN { echo 'tzdata tzdata/Areas select Etc'; echo 'tzdata tzdata/Zones/Etc select UTC'; } | debconf-set-selections

ARG TARGETARCH
ARG BUILD_MODE
ARG IMAGE
# Validate build args
RUN case "${TARGETARCH:-}" in \
      amd64|arm64) ;; \
      *) echo "ERROR: TARGETARCH must be amd64 or arm64 (got '${TARGETARCH:-<unset>}')" >&2; exit 1 ;; \
    esac
RUN case "${BUILD_MODE}" in \
      source|prebuilt) ;; \
      *) echo "ERROR: BUILD_MODE must be source or prebuilt (got '${BUILD_MODE}')" >&2; exit 1 ;; \
    esac
RUN case "${IMAGE}" in \
      "dev-snapshot") ;; \
      *) echo "ERROR: IMAGE must be dev-snapshot (got '${IMAGE}')" >&2; exit 1 ;; \
    esac
RUN --mount=type=secret,id=git_pat \
    set -eu; \
    if [ ! -s /run/secrets/git_pat ]; then \
        echo "ERROR: git_pat secret is missing or empty" >&2; \
        exit 1; \
    fi;

# Install base build tools (gcc-12, g++-12, ninja, flex) shared by all builder stages.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
  software-properties-common \
  gcc-12 \
  g++-12 \
  git \
  ninja-build \
  flex


# ==========================================
# Source build: installs CMake 3.28 and JUCE prerequisites, clones
# surge-synthesizer/surge at SURGE_GIT_REF, builds Surge XT from source,
# installs it system-wide, and symlinks the VST3 into /usr/lib/vst3/.
# ==========================================
FROM builder-base AS builder-install-surge-from-source
COPY --from=arch-vars /artifacts/cmake-arch-name.txt /artifacts/cmake-arch-name.txt
RUN mkdir /home/build/cmake-install
WORKDIR /home/build/cmake-install
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends wget


RUN cmake_arch_name="$(cat /artifacts/cmake-arch-name.txt )"; \
    echo "cmake-arch-name: ${cmake_arch_name}"; \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-${cmake_arch_name}.sh && \
    chmod 755 ./cmake-3.28.1-linux-${cmake_arch_name}.sh && \
    ./cmake-3.28.1-linux-${cmake_arch_name}.sh --skip-license --prefix=/usr
# Then install the prerequistes for a juce build
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git \
        vim \
        cmake \
        ninja-build \
              devscripts \
              libx11-dev \
              libxcb-cursor-dev \
              libxcb-keysyms1-dev \
              libxcb-util-dev \
              libxkbcommon-dev \
              libxkbcommon-x11-dev \
              xcb \
              libgtk-3-dev \
              libwebkit2gtk-4.0 \
              libwebkit2gtk-4.0-dev \
              libcurl4-openssl-dev \
              alsa \
              alsa-tools \
              libasound2-dev \
              libjack-dev \
              libfreetype-dev \
              libfontconfig1-dev \
              libfontconfig1-dev \
              libxinerama-dev \
              libxcb-xinerama0 \
              libxinerama1 \
              x11proto-xinerama-dev \
              libxrandr-dev \
              libgl1-mesa-dev \
              libxcursor-dev \
              libxcursor1 \
              libxcb-cursor-dev \
              libxcb-cursor0 \
              rpm

# Finally, pick gcc12
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

RUN chmod 777 /home/build
RUN mkdir /home/build/dev
RUN mkdir /home/build/plugin
WORKDIR /home/build/plugin

WORKDIR /home/build/plugin
# Now clone and build surge from source
RUN git clone https://github.com/surge-synthesizer/surge.git
WORKDIR /home/build/plugin/surge
ARG SURGE_GIT_REF
RUN git checkout --detach ${SURGE_GIT_REF}
RUN git submodule update --init --recursive
RUN --mount=type=cache,id=surge-cmake-build,target=/home/build/plugin/surge/ignore/sxt \
    cmake -Bignore/sxt -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr
RUN --mount=type=cache,id=surge-cmake-build,target=/home/build/plugin/surge/ignore/sxt \
    cmake --build ignore/sxt --config Release --parallel $(nproc)
RUN --mount=type=cache,id=surge-cmake-build,target=/home/build/plugin/surge/ignore/sxt \
    cmake --install ignore/sxt
RUN cmake_arch_name="$(cat /artifacts/cmake-arch-name.txt )";  \
  mkdir -p /usr/lib/vst3; \
  # Check both multiarch vst3 dirs for the installed Surge XT.vst3 and link it to /usr/lib/vst3/Surge XT.vst3
  if [ -e "/usr/lib/${cmake_arch_name}-linux-gnu/vst3/Surge XT.vst3" ]; then \
    ln -sfn "/usr/lib/${cmake_arch_name}-linux-gnu/vst3/Surge XT.vst3" "/usr/lib/vst3/Surge XT.vst3"; \
  else \
    echo "ERROR: Surge XT.vst3 not found in /usr/lib/${cmake_arch_name}-linux-gnu/ vst3 dirs" >&2; \
    exit 1; \
  fi
RUN rm -rf /home/build/plugin/surge /home/build/cmake-install

# ==========================================
# Prebuilt install: downloads the Surge XT .deb for the target arch and
# installs it together with its runtime dependencies.
# ==========================================
FROM builder-base AS builder-install-surge-from-prebuilt
WORKDIR /home/build
COPY --from=arch-vars /artifacts/surge-package-filepath.txt /surge-package-filepath.txt
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends wget
RUN pkg="$(cat /surge-package-filepath.txt)"; \
    echo $pkg; \
    wget "https://github.com/surge-synthesizer/releases-xt/releases/download/${pkg}"

# Add any additional dependencies needed to install the prebuilt package
# TODO: check to see which of the installs below are actually needed to install the prebuilt package,  remove the ones that aren't needed.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libcairo-dev \
    libxkbcommon-x11-dev \
    libxkbcommon-dev \
    libxcb-cursor-dev \
    libxcb-keysyms1-dev \
    libxcb-util-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libasound2-dev \
    libjack-jackd2-dev
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    pkg="$(cat /surge-package-filepath.txt)" && \
    fname="${pkg##*/}" && \
    apt-get update && apt-get install -y --no-install-recommends "./${fname}" && \
    rm "./${fname}"

# ==========================================
# Source stage: download synth-setter at a specific git ref via tarball.
# Uses GitHub archive endpoint — no full clone, no .git metadata.
# ==========================================
FROM builder-base AS synth-setter-src

ARG SYNTH_PERMUTATIONS_GIT_REF
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl
WORKDIR /home/build/synth-setter
# Best practice: use a commit SHA for SYNTH_PERMUTATIONS_GIT_REF for deterministic builds.
RUN --mount=type=secret,id=git_pat \
    set +x; \
    token="$(cat /run/secrets/git_pat)"; \
    curl -fsSL \
      -H "Authorization: Bearer ${token}" \
      "https://api.github.com/repos/tinaudio/synth-setter/tarball/${SYNTH_PERMUTATIONS_GIT_REF}" \
      -o /tmp/src.tgz && \
    tar -xzf /tmp/src.tgz --strip-components=1 && \
    rm /tmp/src.tgz


# ==========================================
# Python base: Python + venv on top of the chosen Surge install.
# ==========================================
FROM builder-install-surge-from-${BUILD_MODE} AS python-base
# Check that vst3 was installed.
RUN ls /usr/lib/vst3/ && test -n "$(ls -A /usr/lib/vst3/Surge\ XT.vst3 2>/dev/null)" \
    || (echo "ERROR: Surge XT.vst3 not found in /usr/lib/vst3/" >&2; exit 1)

# Install Python and create venv
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev
RUN python3 --version && \
    python3 -c "import sys; exit(0) if sys.version_info[:2] == (3,10) else exit(1)"
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /usr/local/bin/
ENV VIRTUAL_ENV=/venv/main
# PYTHONDONTWRITEBYTECODE=1 — no .pyc files written in the image
# PYTHONUNBUFFERED=1 — stdout/stderr stream immediately (important for logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python --version && \
    python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

# ==========================================
# Shared runtime base: Python + runtime deps on top of python-base.
# Both dev and final-stage inherit from this stage.
# ==========================================
FROM python-base AS builder-install-synth-setter-deps
# Install all Python dependencies via uv in a single resolution pass.
# uv's built-in cache (mounted below) persists across builds — torch wheels
# (~2.5 GB) are only downloaded once and survive requirements-app.txt changes.
#
# --torch-backend: tells uv which PyTorch variant to install (e.g. cu128, cpu).
# This replaces --extra-index-url + --index-strategy unsafe-best-match,
# which was needed because the PyTorch CUDA index carried stale copies of
# common packages.
COPY --from=synth-setter-src /home/build/synth-setter/requirements*.txt /artifacts/
ARG TORCH_BACKEND=cu128
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        --torch-backend ${TORCH_BACKEND} \
        -r /artifacts/requirements.txt && \
    rm -rf /artifacts/requirements*.txt

# Runtime apt packages for headless VST loading + rclone for R2 uploads
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      xvfb \
      xauth \
      dbus-x11 \
      xsettingsd \
      libgl1-mesa-dri \
      libgl1-mesa-glx \
      mesa-utils \
      openbox \
      x11-utils \
      x11-xserver-utils \
      rclone; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /artifacts
COPY --from=synth-setter-src /home/build/synth-setter/scripts/run-linux-vst-headless.sh /artifacts/
COPY --from=synth-setter-src /home/build/synth-setter/src/data/vst/core.py /artifacts/
RUN chmod +x ./run-linux-vst-headless.sh
# Validate that Surge XT loads correctly under headless X11
RUN ./run-linux-vst-headless.sh \
    python -X faulthandler -c "from core import load_plugin; load_plugin('/usr/lib/vst3/Surge XT.vst3')"

RUN mkdir -p /home/build/synth-setter/plugins && \
    ln -s "/usr/lib/vst3/Surge XT.vst3" "/home/build/synth-setter/plugins/Surge XT.vst3"

WORKDIR /home/build/synth-setter

# ==========================================
# R2 config: bake Cloudflare R2 credentials into rclone config using BuildKit
# secrets. Secrets never appear in image layers or `docker history`, but the
# resulting /root/.config/rclone/rclone.conf IS stored in the image filesystem.
#
# SECURITY: This image contains baked cloud storage credentials.
#     - Push ONLY to a private registry (Docker Hub private repo, GHCR private package, etc.)
#     - Anyone who can pull this image has read/write access to the R2 bucket
#     - Rotate R2 API tokens after each build campaign (tokens can't be revoked per-image)
#     - See future hardening TODOs in the project README
# ==========================================
FROM builder-install-synth-setter-deps AS r2-config-base
ARG R2_BUCKET
ARG R2_ENDPOINT
ENV R2_BUCKET=${R2_BUCKET}
RUN --mount=type=secret,id=r2_access_key_id \
    --mount=type=secret,id=r2_secret_access_key \
    set -eu; \
    if [ ! -s /run/secrets/r2_access_key_id ] || [ ! -s /run/secrets/r2_secret_access_key ] || [ -z "${R2_ENDPOINT}" ]; then \
        echo "WARNING: R2_ENDPOINT build-arg or R2 secrets (r2_access_key_id, r2_secret_access_key) are missing or empty." >&2; \
        echo "         R2 upload/download will not be available in this image." >&2; \
        mkdir -p /root/.config/rclone; \
        printf '[r2]\ntype = s3\nprovider = Cloudflare\n# credentials not configured\n' \
            > /root/.config/rclone/rclone.conf; \
    else \
        mkdir -p /root/.config/rclone; \
        printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\n' \
            "$(cat /run/secrets/r2_access_key_id)" \
            "$(cat /run/secrets/r2_secret_access_key)" \
            "${R2_ENDPOINT}" \
            > /root/.config/rclone/rclone.conf; \
        echo "R2 rclone config written." >&2; \
    fi; \
    chmod 600 /root/.config/rclone/rclone.conf

# ==========================================
# W&B auth: bake Weights & Biases API key into ~/.netrc using BuildKit
# secrets. The key never appears in image layers or `docker history`, but
# the resulting ~/.netrc IS stored in the image filesystem.
#
# W&B reads ~/.netrc automatically (same mechanism as `wandb login`).
# ==========================================
RUN --mount=type=secret,id=wandb_api_key \
    set -eu; \
    if [ ! -s /run/secrets/wandb_api_key ]; then \
        echo "WARNING: wandb_api_key secret is missing or empty." >&2; \
        echo "         W&B logging will require WANDB_API_KEY at runtime." >&2; \
    else \
        printf 'machine api.wandb.ai\n  login user\n  password %s\n' \
            "$(cat /run/secrets/wandb_api_key)" \
            >> /root/.netrc; \
        chmod 600 /root/.netrc; \
        echo "W&B netrc config written." >&2; \
    fi

# ==========================================
# dev-snapshot: deps + live git clone at SYNTH_PERMUTATIONS_GIT_REF.
# Contains baked R2 credentials — keep registry PRIVATE.
# Useful for CI or one-off runs without mounting local source.
# Usage: make docker-build-dev-snapshot GIT_REF=<sha> GIT_PAT=<token>
# ==========================================
FROM r2-config-base AS dev-snapshot

ARG SYNTH_PERMUTATIONS_GIT_REF
ENV SYNTH_PERMUTATIONS_GIT_REF=${SYNTH_PERMUTATIONS_GIT_REF}

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends git ca-certificates

WORKDIR /home/build/synth-setter

RUN --mount=type=secret,id=git_pat \
    set +x; \
    token="$(cat /run/secrets/git_pat)"; \
    git init && \
    git remote add origin "https://${token}@github.com/tinaudio/synth-setter.git" && \
    git fetch origin && \
    git checkout --detach "${SYNTH_PERMUTATIONS_GIT_REF}"

# Install the project package so `import src` works from any working directory.
# --no-deps: all runtime deps already installed from requirements*.txt wheels.
RUN uv pip install --no-deps -e .

# Run not-slow unit tests (mocked subprocess + LocalFakeUploader — no VST or R2 needed)
RUN pytest -k "not slow" -v

COPY scripts/docker_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []

FROM ${IMAGE} AS freeze-deps
# Record package versions for audit / reproducibility
RUN mkdir -p /usr/local/share/audit && \
    uv pip freeze | sort > /usr/local/share/audit/pip-freeze.txt && \
    dpkg-query -W -f='${Package}=${Version}\n' | sort > /usr/local/share/audit/dpkg.txt
