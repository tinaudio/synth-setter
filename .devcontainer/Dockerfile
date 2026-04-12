FROM tinaudio/perm:dev-snapshot

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Non-root user with passwordless sudo. Follows the official VS Code pattern:
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Hand /venv/main to the dev user so post-create.sh's `uv pip install -e .`
# works without sudo. Without this, the editable install fails because
# /venv/main was created by root in the base image.
RUN chown -R $USER_UID:$USER_GID /venv/main

# R2 (rclone) and W&B credentials are baked into the base image at
# /root/.config/rclone/rclone.conf and /root/.netrc. Copy them into the dev
# user's home so rclone and wandb work out of the box without sudo.
RUN mkdir -p /home/$USERNAME/.config \
    && cp -r /root/.config/rclone /home/$USERNAME/.config/rclone \
    && cp /root/.netrc /home/$USERNAME/.netrc \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME/.config /home/$USERNAME/.netrc \
    && chmod 600 /home/$USERNAME/.netrc \
    && chmod 600 /home/$USERNAME/.config/rclone/rclone.conf
