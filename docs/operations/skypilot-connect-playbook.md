# SkyPilot Client Connection Playbook

A companion to the *SkyPilot API Server on OCI Always-Free + Cloudflare Tunnel* playbook. This one is from the **client side**: how to authenticate to and use a SkyPilot API server that sits behind Cloudflare Access.

You'll set up two complementary auth paths:

- **Path A — Browser / SSO (email + one-time PIN):** what you use day-to-day from your laptop.
- **Path B — Service token via local `cloudflared access tcp` proxy:** what you use for CI, scripts, the Python SDK, or any headless context.

Both end with the same result: `sky api info` returns OK, and the Python SDK works without the SDK ever knowing Cloudflare exists.

______________________________________________________________________

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Architecture: how each path works](#architecture-how-each-path-works)
3. [Path A: Browser/SSO login](#path-a-browsersso-login)
4. [Path B: Service token + local TCP proxy](#path-b-service-token--local-tcp-proxy)
5. [Smoke tests](#smoke-tests)
6. [Switching between paths](#switching-between-paths)
7. [Troubleshooting](#troubleshooting)

______________________________________________________________________

## Prerequisites

You should have already:

- Stood up the API server per the server playbook — `https://sky.yourdomain.com/api/health` responds (when authenticated).
- Configured a Cloudflare Access application protecting the hostname, with at least one Allow policy for your email.
- Have `cloudflared` and `uv` installable on the client machine.

Replace `sky.yourdomain.com` with your real hostname throughout.

______________________________________________________________________

## Architecture: how each path works

```
                     PATH A — Browser/SSO
   ┌──────────┐           ┌────────────┐           ┌─────────────┐
   │  sky CLI │  HTTPS +  │ Cloudflare │  Tunnel   │  SkyPilot   │
   │          │  cookie   │   Access   │           │  API server │
   │ (laptop) │──────────▶│   (SSO)    │──────────▶│             │
   └──────────┘           └────────────┘           └─────────────┘
        ▲
        │ cookie cached by `cloudflared access login`
        │ (via browser PIN flow)


                  PATH B — Service token via TCP proxy
   ┌──────────┐    plain    ┌──────────────┐  HTTPS +    ┌────────────┐  Tunnel  ┌─────────────┐
   │  sky CLI │   HTTP      │ cloudflared  │  service    │ Cloudflare │          │  SkyPilot   │
   │ Python   │────────────▶│ access tcp   │  token      │   Access   │─────────▶│  API server │
   │ SDK      │ localhost   │ (background) │  headers    │            │          │             │
   └──────────┘   :8080     └──────────────┘────────────▶└────────────┘          └─────────────┘
```

**The key insight for Path B:** `cloudflared access tcp` creates a local TCP listener that handles the Access auth handshake transparently using your service token. Your application just sees a plain HTTP endpoint on `localhost`. The SkyPilot SDK doesn't need to know anything about Cloudflare headers, cookies, or service tokens.

______________________________________________________________________

## Path A: Browser/SSO login

Use this on your laptop, day to day.

### A.1 Install `cloudflared`

**macOS:**

```bash
brew install cloudflared
```

**Linux:**

```bash
curl -L --output cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

**Windows:** download the `.msi` from the [cloudflared releases page](https://github.com/cloudflare/cloudflared/releases/latest), or use WSL.

Verify:

```bash
cloudflared --version
```

### A.2 Cache an Access cookie

```bash
cloudflared access login https://sky.yourdomain.com
```

A browser window opens. Enter your email, retrieve the one-time PIN from your inbox, paste it back. The Access cookie is cached at `~/.cloudflared/sky.yourdomain.com.tok` (or similar).

### A.3 Verify the cookie works

```bash
cloudflared access curl https://sky.yourdomain.com/api/health
```

You should see SkyPilot's JSON health response. If you get a 302 redirect to a Cloudflare login page, the cookie didn't take — re-run `cloudflared access login`.

### A.4 Install SkyPilot client

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc   # or ~/.bashrc on Linux
uv tool install --python 3.11 "skypilot-nightly"
sky --version
```

You don't need cloud extras (`[runpod]`, `[aws]`, etc.) on the client — the *server* talks to clouds, the client just talks to the server.

### A.5 Log into the server

```bash
sky api login -e https://sky.yourdomain.com
```

SkyPilot detects the auth proxy, opens a browser if needed, and stores the cookie/token in `~/.sky/config.yaml`. If it asks for a token, paste the value from the browser flow.

```bash
sky api info
```

You should see the server's status, version, and your user identity.

### A.6 Cookie expiry

The Access session you set on the server (e.g., `24 hours` or `1 month`) determines how long this cookie is valid. When it expires, `sky` commands start failing — re-run `cloudflared access login` and you're back in.

If you find this annoying mid-job, switch that session to Path B.

______________________________________________________________________

## Path B: Service token + local TCP proxy

Use this for:

- **Python SDK code** that doesn't know how to inject Access headers.
- **CI runners** (GitHub Actions, etc.) where there's no human to type a PIN.
- **Long-running scripts** that would outlive an SSO session.
- **Any headless box** where browser auth isn't an option.

### B.1 Create a service token in Cloudflare

In the Zero Trust dashboard:

1. **Access → Service Auth → Service Tokens → Create Service Token**
2. **Name:** `skypilot-sdk` (or per-machine: `skypilot-laptop`, `skypilot-ci-prod`, etc.)
3. **Service Token Duration:** `1 year` is fine. Set a calendar reminder to rotate.
4. Click **Generate token**.
5. Copy both the **Client ID** (looks like `abc123def456.access`) and **Client Secret** *immediately* — the secret is shown exactly once.

### B.2 Attach it to your Access application

The token is useless until the Access policy accepts it.

1. **Access → Applications → SkyPilot → Edit → Policies tab.**
2. Add a new policy:
   - **Policy name:** `Service tokens`
   - **Action:** `Service Auth` ← **critical**, do not leave it on the default Allow.
   - **Configure rules → Include:**
     - Selector: `Service Token`
     - Value: select `skypilot-sdk`.
3. Save.

Keep your existing email-allow policy in place — the two coexist. SSO users go through one policy, service tokens through the other.

### B.3 Store credentials securely on the client

```bash
# Add to ~/.zshrc, ~/.bashrc, or your secrets manager
export CF_ACCESS_CLIENT_ID="abc123def456.access"
export CF_ACCESS_CLIENT_SECRET="your-secret-value"
```

For CI: stash these in repository / pipeline secrets, never in code.

Smoke test the token directly (no proxy yet):

```bash
curl -i \
  -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" \
  -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" \
  https://sky.yourdomain.com/api/health
```

Expected: `HTTP/2 200` and the SkyPilot health JSON. If you get `HTTP/2 302` to a login page, the Service Auth policy isn't right — re-check step B.2.

### B.4 Start the local TCP proxy

This is the magic step. `cloudflared access tcp` opens a local listener that forwards traffic through Access using your service token:

```bash
cloudflared access tcp \
  --hostname sky.yourdomain.com \
  --url localhost:8080 \
  --service-token-id "$CF_ACCESS_CLIENT_ID" \
  --service-token-secret "$CF_ACCESS_CLIENT_SECRET"
```

Leave it running. You should see something like:

```
INF Start Websocket listener host=localhost:8080
```

In another terminal:

```bash
curl http://localhost:8080/api/health
```

Note: **plain `http://localhost:8080`** — no Access headers, no `https://`. The proxy is doing the TLS + auth on your behalf. You should see the same health JSON.

### B.5 Run the proxy in the background

For a launchd/systemd-style daemon, see the [Daemonizing the proxy](#daemonizing-the-proxy) section below. For ad-hoc use:

```bash
nohup cloudflared access tcp \
  --hostname sky.yourdomain.com \
  --url localhost:8080 \
  --service-token-id "$CF_ACCESS_CLIENT_ID" \
  --service-token-secret "$CF_ACCESS_CLIENT_SECRET" \
  > ~/cloudflared-sky.log 2>&1 &

echo $! > ~/cloudflared-sky.pid
```

Kill with `kill $(cat ~/cloudflared-sky.pid)`.

### B.6 Point SkyPilot at the local proxy

```bash
sky api login -e http://localhost:8080
sky api info
```

Note the `http://` (not `https://`) and `localhost:8080` (not `sky.yourdomain.com`). From SkyPilot's perspective, the API server is on localhost. The auth/TLS gymnastics are entirely below the SDK.

### B.7 Use the Python SDK

No code changes needed beyond pointing `SKYPILOT_API_SERVER_ENDPOINT` (or having run `sky api login`) at the local proxy:

```python
import sky

# Talks to localhost:8080 → cloudflared → Access → real API server
print(sky.status())

task = sky.Task(run="echo hello from skypilot")
job_id, _ = sky.launch(task, cluster_name="smoke")
sky.tail_logs(cluster_name="smoke", job_id=job_id)
sky.down("smoke")
```

### Daemonizing the proxy

#### macOS (launchd)

```bash
mkdir -p ~/Library/LaunchAgents

cat > ~/Library/LaunchAgents/com.user.cloudflared-sky.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.user.cloudflared-sky</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/cloudflared</string>
    <string>access</string>
    <string>tcp</string>
    <string>--hostname</string>
    <string>sky.yourdomain.com</string>
    <string>--url</string>
    <string>localhost:8080</string>
    <string>--service-token-id</string>
    <string>YOUR_CLIENT_ID.access</string>
    <string>--service-token-secret</string>
    <string>YOUR_CLIENT_SECRET</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/cloudflared-sky.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/cloudflared-sky.err</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.user.cloudflared-sky.plist
```

Note: hardcoding the secret in a plist isn't great. For better hygiene, write a wrapper shell script that reads the secret from the macOS Keychain (`security find-generic-password`) and have launchd run that instead.

Verify:

```bash
launchctl list | grep cloudflared-sky
curl http://localhost:8080/api/health
```

Unload with `launchctl unload ~/Library/LaunchAgents/com.user.cloudflared-sky.plist`.

#### Linux (systemd user service)

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/cloudflared-sky.service <<EOF
[Unit]
Description=cloudflared TCP proxy to SkyPilot API server
After=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.config/cloudflared-sky.env
ExecStart=/usr/local/bin/cloudflared access tcp \\
  --hostname sky.yourdomain.com \\
  --url localhost:8080 \\
  --service-token-id \${CF_ACCESS_CLIENT_ID} \\
  --service-token-secret \${CF_ACCESS_CLIENT_SECRET}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# Secrets out of the unit file
cat > ~/.config/cloudflared-sky.env <<'EOF'
CF_ACCESS_CLIENT_ID=abc123def456.access
CF_ACCESS_CLIENT_SECRET=your-secret-value
EOF
chmod 600 ~/.config/cloudflared-sky.env

systemctl --user daemon-reload
systemctl --user enable --now cloudflared-sky
systemctl --user status cloudflared-sky
```

To survive logout: `sudo loginctl enable-linger $USER`.

______________________________________________________________________

## Smoke tests

Drop these into a file and run after either auth path is set up. They verify the full chain: TCP → Cloudflare Access → tunnel → SkyPilot.

### `smoke_test.sh` — bash-level checks

```bash
#!/usr/bin/env bash
# Smoke-test the SkyPilot client connection.
# Usage:
#   ./smoke_test.sh https://sky.yourdomain.com   # Path A (SSO)
#   ./smoke_test.sh http://localhost:8080        # Path B (TCP proxy)

set -euo pipefail

ENDPOINT="${1:-${SKYPILOT_API_SERVER_ENDPOINT:-}}"
if [[ -z "$ENDPOINT" ]]; then
  echo "Usage: $0 <endpoint-url>" >&2
  exit 1
fi

pass() { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; exit 1; }

echo "Endpoint: $ENDPOINT"
echo

echo "[1/5] Reachability"
if [[ "$ENDPOINT" == http://localhost* ]]; then
  curl -sf "$ENDPOINT/api/health" >/dev/null \
    || fail "Local proxy not responding. Is cloudflared access tcp running?"
elif [[ -n "${CF_ACCESS_CLIENT_ID:-}" && -n "${CF_ACCESS_CLIENT_SECRET:-}" ]]; then
  curl -sf \
    -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" \
    -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" \
    "$ENDPOINT/api/health" >/dev/null \
    || fail "Health check failed with service token. Check token + Access policy."
else
  cloudflared access curl "$ENDPOINT/api/health" >/dev/null 2>&1 \
    || fail "Health check failed via cloudflared cookie. Run cloudflared access login."
fi
pass "API server is reachable"

echo "[2/5] Status code on protected route (no creds)"
code=$(curl -s -o /dev/null -w '%{http_code}' "$ENDPOINT/api/health" || true)
if [[ "$ENDPOINT" == http://localhost* ]]; then
  [[ "$code" == "200" ]] && pass "Local proxy returns 200 (auth handled by proxy)" \
    || fail "Local proxy returned $code (expected 200)"
else
  [[ "$code" == "302" || "$code" == "401" || "$code" == "403" ]] \
    && pass "Naked request blocked ($code) — Access is enforcing" \
    || fail "Naked request returned $code — Access may not be in front of the server!"
fi

echo "[3/5] sky api info"
sky api login -e "$ENDPOINT" >/dev/null 2>&1 || true
sky api info > /tmp/sky-info.txt 2>&1 \
  || { cat /tmp/sky-info.txt; fail "sky api info failed"; }
grep -q "healthy" /tmp/sky-info.txt && pass "sky api info reports healthy" \
  || { cat /tmp/sky-info.txt; fail "sky api info did not return healthy"; }

echo "[4/5] sky check (cloud credentials on the server)"
sky check > /tmp/sky-check.txt 2>&1 || true
tail -20 /tmp/sky-check.txt
grep -qE "enabled|✓" /tmp/sky-check.txt \
  && pass "At least one cloud is enabled on the server" \
  || fail "No cloud reported enabled. Fix credentials on the server."

echo "[5/5] sky status (request round-trip)"
sky status >/dev/null 2>&1 \
  && pass "sky status round-trip succeeded" \
  || fail "sky status failed — likely an auth or network issue"

echo
echo "All checks passed for $ENDPOINT"
```

Make it executable and run:

```bash
chmod +x smoke_test.sh

# Path A
./smoke_test.sh https://sky.yourdomain.com

# Path B
./smoke_test.sh http://localhost:8080
```

### `smoke_test.py` — Python SDK checks

```python
"""SkyPilot SDK smoke test.

Run after `sky api login` is configured. Exits 0 if everything works.

  $ python smoke_test.py
"""

from __future__ import annotations

import os
import sys
import time

import requests


def info(msg: str) -> None:
    print(f"  • {msg}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def die(msg: str) -> None:
    print(f"  ✗ {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    print("SkyPilot SDK smoke test")
    print("=" * 40)

    # ----- 1. Discover endpoint -----
    endpoint = os.environ.get("SKYPILOT_API_SERVER_ENDPOINT")
    if not endpoint:
        try:
            import sky.server.common as sky_common  # private but stable
            endpoint = sky_common.get_server_url()
        except Exception:  # pragma: no cover
            die("Could not determine endpoint. Run `sky api login` first.")
    info(f"Endpoint: {endpoint}")

    # ----- 2. Raw HTTP health check -----
    try:
        r = requests.get(f"{endpoint}/api/health", timeout=10)
    except requests.exceptions.RequestException as e:
        die(f"HTTP request failed: {e}")
    if r.status_code != 200:
        die(f"/api/health returned {r.status_code} (expected 200). "
            f"If you're hitting Cloudflare directly, the local proxy may be down "
            f"or your service token may not be on the Access policy.")
    payload = r.json()
    if payload.get("status") != "healthy":
        die(f"Server reports non-healthy status: {payload}")
    ok(f"Server healthy (api_version={payload.get('api_version')}, "
       f"version={payload.get('version')})")

    # ----- 3. SDK import + version -----
    try:
        import sky
    except ImportError:
        die("`sky` not importable. `uv tool install skypilot-nightly` then re-run.")
    ok(f"SDK imports (sky.__version__={getattr(sky, '__version__', '?')})")

    # ----- 4. Authenticated SDK call -----
    try:
        request_id = sky.status()
        # sky.status returns a request id in async mode; resolve it.
        clusters = sky.stream_and_get(request_id) if hasattr(sky, "stream_and_get") \
            else sky.get(request_id)
    except Exception as e:
        die(f"sky.status() failed: {e}")
    ok(f"sky.status() returned {len(clusters) if clusters else 0} clusters")

    # ----- 5. (Optional) round-trip latency -----
    t0 = time.perf_counter()
    requests.get(f"{endpoint}/api/health", timeout=10)
    dt_ms = (time.perf_counter() - t0) * 1000
    info(f"Health-check round-trip: {dt_ms:.0f} ms")

    print("\nAll SDK smoke checks passed.")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python smoke_test.py
```

If you've also set `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` and you're targeting the public hostname directly (no local proxy), you'll need to either run the proxy first or modify the script to inject those headers. The recommended path is just to run the proxy and let the SDK think it's talking to localhost.

### One-shot launch test

If you want to test that jobs actually run end-to-end (costs a few cents on RunPod):

```bash
sky launch -y --infra runpod --gpus T4:1 --cluster smoke-test -- \
  'echo "hello from $(hostname)" && nvidia-smi'
sky logs smoke-test
sky down -y smoke-test
```

The `T4:1` is cheap and widely available. Swap for `A100:1` etc. if you want to verify higher-tier accelerators are reachable.

______________________________________________________________________

## Switching between paths

You can have both paths set up simultaneously — they don't conflict:

- `sky api login -e https://sky.yourdomain.com` → uses cookie auth (Path A).
- `sky api login -e http://localhost:8080` → uses local proxy (Path B).

`sky api login` overwrites the endpoint in `~/.sky/config.yaml`, so whichever you logged into last is the active one. To check:

```bash
sky api info
```

The `Endpoint:` line tells you which path is currently active.

For mixed workflows (browser for dashboard, SDK via proxy), the cleanest pattern is:

```bash
# In your shell rc — pin the SDK to localhost
export SKYPILOT_API_SERVER_ENDPOINT=http://localhost:8080
```

This env var overrides whatever's in `config.yaml` for SDK and CLI calls, so even if you logged into `https://sky.yourdomain.com` for the browser session, scripts pick up the local proxy.

______________________________________________________________________

## Troubleshooting

### `cloudflared access login` opens a browser but never completes

Your terminal can't reach the local callback port. Try:

```bash
cloudflared access login --no-browser https://sky.yourdomain.com
```

Manually open the printed URL, complete the flow, then paste the resulting token if asked.

### `sky api login` returns 403 with the cookie path

The auth proxy detection in SkyPilot can miss the cookie. Force it:

```bash
sky api login --relogin -e https://sky.yourdomain.com
```

If still 403, switch to the service token path — it's more deterministic.

### `cloudflared access tcp` exits immediately

Check the logs in the same terminal. Common causes:

- **`failed to get Access App`** → the hostname doesn't match an Access application. Make sure `--hostname` is exactly `sky.yourdomain.com`, no trailing slash, no `https://`.
- **`401 Unauthorized`** → service token isn't on the Access policy. Re-check step B.2 (Action must be `Service Auth`, not `Allow`).
- **`address already in use`** → something is on `localhost:8080`. Pick another port (`--url localhost:8081`) and update `sky api login` to match.

### `curl http://localhost:8080/api/health` hangs

The proxy started but can't reach Cloudflare. Check connectivity:

```bash
curl -v https://sky.yourdomain.com  # should at least get a Cloudflare response
```

If your network blocks Cloudflare, you have bigger problems. Otherwise, kill and restart the proxy.

### Python SDK calls hang for a long time then fail

SkyPilot's HTTP client has long timeouts. If the proxy died mid-request, you may wait minutes before getting an error. Always check that the proxy is running before assuming it's a SkyPilot bug:

```bash
ps aux | grep '[c]loudflared access tcp'
curl http://localhost:8080/api/health
```

### Service token rejected after a long time

Tokens have an expiration set at creation (default 1 year). After expiry, every request returns 401. Generate a new token and update env vars / launchd / systemd. Cloudflare doesn't auto-renew.

### "Works on my laptop, fails in CI"

Three usual suspects:

1. **Token not in CI secrets** — verify with `echo $CF_ACCESS_CLIENT_ID | wc -c` (should be ~30+ chars).

2. **Proxy not started in CI job** — you need a step that runs `cloudflared access tcp ... &` and waits a few seconds before the SkyPilot step. Example:

   ```yaml
   - name: Start Cloudflare proxy
     run: |
       cloudflared access tcp \
         --hostname sky.yourdomain.com \
         --url localhost:8080 \
         --service-token-id "$CF_ACCESS_CLIENT_ID" \
         --service-token-secret "$CF_ACCESS_CLIENT_SECRET" &
       # Wait for the listener
       for i in {1..15}; do
         curl -sf http://localhost:8080/api/health && break || sleep 1
       done
     env:
       CF_ACCESS_CLIENT_ID: ${{ secrets.CF_ACCESS_CLIENT_ID }}
       CF_ACCESS_CLIENT_SECRET: ${{ secrets.CF_ACCESS_CLIENT_SECRET }}

   - name: Run SkyPilot job
     run: |
       sky api login -e http://localhost:8080
       sky launch -y task.yaml
   ```

3. **CI runner architecture mismatch** — the cloudflared binary you're shipping doesn't match the runner. Use the official GitHub release for the right arch (amd64 for most CI, arm64 for Graviton runners).

______________________________________________________________________

## What to keep handy

A printable cheat sheet for once everything works:

```bash
# Path A — refresh SSO cookie (when sky commands start failing)
cloudflared access login https://sky.yourdomain.com

# Path B — start the local proxy (if not running as a service)
cloudflared access tcp \
  --hostname sky.yourdomain.com \
  --url localhost:8080 \
  --service-token-id "$CF_ACCESS_CLIENT_ID" \
  --service-token-secret "$CF_ACCESS_CLIENT_SECRET" &

# Active endpoint check
sky api info

# Switch active endpoint
sky api login -e https://sky.yourdomain.com   # SSO
sky api login -e http://localhost:8080        # via proxy

# Smoke
./smoke_test.sh "$(sky api info | awk '/Endpoint/ {print $2}')"
```
