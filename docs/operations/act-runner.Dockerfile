FROM catthehacker/ubuntu:act-latest

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    # catthehacker's base ships a packagecloud.io git-lfs apt source that now 403s;
    # git-lfs is already installed and we only need gh, so drop it before apt-get update.
    find /etc/apt/sources.list.d/ -type f -exec grep -q 'packagecloud\.io' {} \; -delete; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg; \
    install -m 0755 -d /etc/apt/keyrings; \
    curl -fsSL \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
        https://cli.github.com/packages/githubcli-archive-keyring.gpg; \
    chmod a+r /etc/apt/keyrings/githubcli-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends gh; \
    rm -rf /var/lib/apt/lists/*; \
    gh --version
