# Platform routing for this amd64-only base image is pinned in
# devcontainer.json's build.options (--platform=linux/amd64). That sets
# the build target for the whole build — pulling the amd64 manifest AND
# scheduling RUN steps on an amd64-capable worker — which a FROM --platform
# flag alone does not do (it only selects the manifest variant).
FROM tinaudio/perm:dev-snapshot

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Non-root user with passwordless sudo. Follows the official VS Code pattern:
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Give the dev user write access only where post-create.sh's
# `uv pip install --no-deps -e .` needs to create files (bin/ for entry
# points, site-packages/ for the .pth file). Avoids a recursive chown of
# the entire prebuilt ~2.5GB venv layer, which would copy-up every file
# into a new image layer and balloon build time and image size.
RUN SITE_PACKAGES="$(/venv/main/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
    && chown -R $USER_UID:$USER_GID /venv/main/bin "$SITE_PACKAGES"

# R2 (rclone) and W&B credentials may be baked into the base image at
# /root/.config/rclone/rclone.conf and /root/.netrc. Copy them into the dev
# user's home when present so rclone and wandb work out of the box without
# sudo, while still allowing builds without credentials baked in.
RUN mkdir -p /home/$USERNAME/.config \
    && chown $USER_UID:$USER_GID /home/$USERNAME/.config \
    && if [ -d /root/.config/rclone ]; then \
         cp -r /root/.config/rclone /home/$USERNAME/.config/rclone \
         && chown -R $USER_UID:$USER_GID /home/$USERNAME/.config/rclone \
         && chmod 600 /home/$USERNAME/.config/rclone/rclone.conf; \
       fi \
    && if [ -f /root/.netrc ]; then \
         cp /root/.netrc /home/$USERNAME/.netrc \
         && chown $USER_UID:$USER_GID /home/$USERNAME/.netrc \
         && chmod 600 /home/$USERNAME/.netrc; \
       fi
