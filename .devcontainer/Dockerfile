FROM tinaudio/perm:dev-snapshot

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Non-root dev user with bash as the login shell. No sudo: passwordless
# sudo is security theatre (dev becomes root trivially), and the escape
# hatch for missing system packages is adding them to the base image
# (docker/ubuntu22_04/Dockerfile) and rebuilding.
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash -m $USERNAME

# R2 (rclone) and W&B credentials are baked into /root in the base image.
# Copy them into the dev user's home when present so rclone and wandb work
# out of the box.
# TODO: remove once the base image bakes credentials into /home/dev directly.
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

USER $USERNAME
