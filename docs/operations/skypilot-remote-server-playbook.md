# SkyPilot API Server on OCI Always-Free + Cloudflare Tunnel

A step-by-step playbook for standing up a personal SkyPilot client-server deployment with zero monthly cost, using:

- **OCI Always-Free Ampere A1** (Ubuntu 24.04) as the host for the SkyPilot API server
- **Cloudflare Tunnel** for HTTPS exposure with no open inbound ports
- **Cloudflare Access** for SSO authentication
- **uv** for Python and SkyPilot installation
- **RunPod** as the compute backend for actual GPU jobs (optional — substitute or add any cloud SkyPilot supports)

End result: a private `https://sky.yourdomain.com` endpoint that you authenticate to with email-based SSO, hosting a SkyPilot API server you can drive from any laptop.

______________________________________________________________________

## Table of contents

01. [Prerequisites](#prerequisites)
02. [Architecture overview](#architecture-overview)
03. [Part 1: Provision the OCI VM](#part-1-provision-the-oci-vm)
04. [Part 2: Install SkyPilot with uv](#part-2-install-skypilot-with-uv)
05. [Part 3: Configure cloud credentials on the server](#part-3-configure-cloud-credentials-on-the-server)
06. [Part 4: Start the SkyPilot API server](#part-4-start-the-skypilot-api-server)
07. [Part 5: Cloudflare Tunnel](#part-5-cloudflare-tunnel)
08. [Part 6: Cloudflare Access (SSO)](#part-6-cloudflare-access-sso)
09. [Part 7: Connect your laptop](#part-7-connect-your-laptop)
10. [Part 8: First job](#part-8-first-job)
11. [Operational notes](#operational-notes)
12. [Troubleshooting](#troubleshooting)

______________________________________________________________________

## Prerequisites

You need:

- **An OCI tenancy** with the Always-Free tier active. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/). Credit card required for verification but never charged on free tier.
- **A domain on Cloudflare.** Either transfer one or just point your domain's nameservers at Cloudflare. Free.
- **A Cloudflare Zero Trust account.** Free for up to 50 users. Activate at [one.dash.cloudflare.com](https://one.dash.cloudflare.com).
- **A laptop** (Mac, Linux, or WSL on Windows) for the SkyPilot client.
- **Cloud account(s) where you want to actually launch jobs** — RunPod, AWS, GCP, etc. The OCI VM is just the SkyPilot brain; jobs run elsewhere.

Throughout this playbook, replace these placeholders with your own values:

- `yourdomain.com` → your actual domain (e.g., `tinaudioskypilot.com`)
- `sky.yourdomain.com` → your chosen subdomain
- `<vm-public-ip>` → your OCI VM's public IP

______________________________________________________________________

## Architecture overview

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────────────┐
│  Your       │  HTTPS  │  Cloudflare      │ Tunnel  │  OCI A1 VM          │
│  Laptop     │────────▶│  Edge + Access   │────────▶│  (Ubuntu 24.04)     │
│  sky CLI    │  (SSO)  │  (auth gate)     │ outbound│  - cloudflared      │
└─────────────┘         └──────────────────┘         │  - sky api server   │
                                                     │    (127.0.0.1:46580)│
                                                     └──────────┬──────────┘
                                                                │
                                                       sky launches jobs on:
                                                                │
                                                     ┌──────────▼──────────┐
                                                     │  RunPod / AWS /     │
                                                     │  GCP / Kubernetes   │
                                                     └─────────────────────┘
```

Key properties:

- **No inbound ports open on the VM.** `cloudflared` connects outbound to Cloudflare's edge; all traffic returns through that tunnel. The OCI VM's security list only needs SSH for your IP.
- **TLS handled by Cloudflare.** No certificates to manage on your VM.
- **Auth handled by Cloudflare.** SkyPilot itself runs without auth (`--host 127.0.0.1`), trusting the upstream gate.
- **Free.** OCI A1 is always-free for 4 OCPU / 24 GiB across instances. Cloudflare Tunnel and Access are free for personal use.

______________________________________________________________________

## Part 1: Provision the OCI VM

### 1.1 Create the instance

In the OCI console:

1. Navigate to **Compute → Instances → Create instance**.
2. **Name:** `skypilot-server` (or whatever you like).
3. **Image:** Click "Change image" → **Ubuntu** → **Canonical Ubuntu 24.04** (Minimal is fine).
4. **Shape:** Click "Change shape" → **Ampere** → **VM.Standard.A1.Flex**.
   - **OCPUs:** 2 (Always-Free includes 4 across instances; 2 leaves headroom for another VM later.)
   - **Memory (GB):** 12
5. **Networking:** Use defaults — a public subnet with auto-assigned public IP. SSH-only ingress is fine.
6. **SSH keys:** Either upload your public key or let OCI generate a key pair (download it immediately — you can't retrieve it later).
7. **Boot volume:** Default 50 GB is fine.
8. Click **Create**.

### 1.2 Capacity errors

Always-Free A1 capacity is regularly exhausted in popular regions (Ashburn, Phoenix, Frankfurt). If you get "Out of host capacity":

- Try a different availability domain in the same region (the AD dropdown).
- Try a less popular region (e.g., Tokyo, Mumbai, Saudi Arabia).
- Retry every few minutes — capacity frees up sporadically.
- A common trick is to script the create-instance API call and retry on the capacity error.

### 1.3 Lock down the security list

Once the VM is up, find its VCN's default security list and confirm it only allows:

- **Ingress:** TCP 22 from `0.0.0.0/0` (or better, your IP only).
- **Egress:** all (default).

**Do not open port 46580 or any other inbound port.** Cloudflare Tunnel does not require it.

### 1.4 SSH in

```bash
ssh -i /path/to/your/private-key ubuntu@<vm-public-ip>
```

You should land at `ubuntu@skypilot-server:~$`.

### 1.5 Update the system

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

Wait 30 seconds, SSH back in.

______________________________________________________________________

## Part 2: Install SkyPilot with uv

### 2.1 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

You should see a version string.

### 2.2 Install SkyPilot as a uv tool

`uv tool install` puts SkyPilot in an isolated environment with its own Python interpreter, and exposes the `sky` command on your `PATH`. No venv activation needed.

Pick the cloud extras you actually want. For RunPod + OCI + Kubernetes:

```bash
uv tool install --python 3.11 --with pip "skypilot-nightly[runpod,oci,kubernetes]"
```

If you want everything:

```bash
uv tool install --python 3.11 --with pip "skypilot-nightly[all]"
```

Verify:

```bash
sky --version
which sky
```

`sky` should resolve to something like `~/.local/bin/sky`.

> **Why `--with pip`?** SkyPilot internally shells out to `pip` for some installation steps. Including `pip` in the tool environment avoids "pip not found" errors later.

> **Why `skypilot-nightly` instead of `skypilot`?** Nightly tracks the latest features and bug fixes. Stable releases (`skypilot`) are sometimes months behind, especially for client-server features that are still evolving. If you prefer stability, swap to `skypilot` — everything else in this playbook works the same.

______________________________________________________________________

## Part 3: Configure cloud credentials on the server

The API server uses **its own** credentials to launch jobs, not your laptop's. Set up creds for whichever clouds you'll target.

### 3.1 RunPod

1. In the [RunPod console](https://runpod.io), go to **Settings → API Keys → Create API Key**. Copy it.
2. On the VM:

```bash
mkdir -p ~/.runpod
cat > ~/.runpod/config.toml <<'EOF'
api_key = "your-runpod-api-key-here"
EOF
chmod 600 ~/.runpod/config.toml
```

### 3.2 OCI (optional — only if you want to launch jobs *on* OCI)

The Always-Free A1 quota is consumed by the API server VM itself, so you typically won't launch additional OCI jobs. Skip unless you have paid OCI capacity.

If you do want it:

```bash
sudo apt install -y python3-oci-cli
oci setup config
```

Walk through the prompts. You'll need your tenancy OCID, user OCID, and region (find them in the OCI console under Profile → Tenancy / User Settings). Generate a new API key when prompted, then upload the resulting public key under your OCI user → API Keys.

### 3.3 AWS (optional)

```bash
sudo apt install -y awscli
aws configure
```

Enter your access key, secret, default region.

### 3.4 GCP (optional)

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-arm.tar.gz
tar -xf google-cloud-cli-linux-arm.tar.gz
./google-cloud-sdk/install.sh
exec -l "$SHELL"
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 3.5 Kubernetes (optional)

Drop a kubeconfig at `~/.kube/config` with credentials for whatever cluster you want SkyPilot to use.

### 3.6 Verify

```bash
sky check
```

You should see at least one cloud showing **enabled**. If everything fails, fix credentials before continuing.

______________________________________________________________________

## Part 4: Start the SkyPilot API server

### 4.1 Start it

```bash
sky api start --deploy --host 127.0.0.1
```

`--deploy` makes it long-running. `--host 127.0.0.1` binds to localhost only — Cloudflare Tunnel reaches it over loopback, and nothing external can hit it directly.

> **Display quirk:** the startup message may show `http://0.0.0.0:46580` even though it's actually bound to `127.0.0.1`. Verify with:
>
> ```bash
> sudo ss -tlnp | grep 46580
> ```
>
> The first column should show `127.0.0.1:46580`, not `0.0.0.0:46580`.

### 4.2 Health check

```bash
curl http://127.0.0.1:46580/api/health
```

Should return JSON like:

```json
{"status":"healthy","api_version":"49","version":"0.12.2rc1",...}
```

If you see `"external_proxy_auth_enabled":true` in there, that's correct — SkyPilot trusts the upstream gate (Cloudflare) for auth.

### 4.3 Auto-start on reboot (recommended)

`sky api start --deploy` doesn't survive a reboot by default. Create a systemd unit so it does:

```bash
sudo tee /etc/systemd/system/skypilot-api.service > /dev/null <<EOF
[Unit]
Description=SkyPilot API Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Environment="PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/.local/bin/sky api start --deploy --host 127.0.0.1 --foreground
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable skypilot-api
sudo systemctl start skypilot-api
sudo systemctl status skypilot-api
```

> **Note:** if `sky api start` doesn't accept `--foreground` in your version, drop that flag. The service will still work; systemd just tracks it slightly less precisely.

If you went the systemd route, stop the manual `sky api start` you ran earlier — you only want one running.

______________________________________________________________________

## Part 5: Cloudflare Tunnel

### 5.1 Make sure your domain is on Cloudflare

In the Cloudflare dashboard, your domain should appear with status "Active". If you registered it through Cloudflare Registrar, this is automatic. Otherwise update your nameservers at the original registrar to Cloudflare's two assigned NS records, then wait for propagation (usually \<1 hour).

### 5.2 Install cloudflared on the VM

OCI A1 is ARM64:

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

For x86 VMs, swap `arm64` → `amd64`.

### 5.3 Authenticate cloudflared

```bash
cloudflared tunnel login
```

This prints a URL. Open it in your laptop's browser, log into Cloudflare, and pick the domain you'll use. A certificate is written to `~/.cloudflared/cert.pem`.

### 5.4 Create the tunnel

```bash
cloudflared tunnel create skypilot
```

Output includes a tunnel UUID and the path to a credentials JSON. Note both. Example:

```
Tunnel credentials written to /home/ubuntu/.cloudflared/abc123-...-def.json
Created tunnel skypilot with id abc123-...-def
```

### 5.5 Route DNS

```bash
cloudflared tunnel route dns skypilot sky.yourdomain.com
```

This automatically creates a CNAME in Cloudflare DNS pointing `sky.yourdomain.com` at the tunnel.

### 5.6 Write the tunnel config

```bash
nano ~/.cloudflared/config.yml
```

Paste, replacing `<UUID>` with your tunnel UUID:

```yaml
tunnel: <UUID>
credentials-file: /home/ubuntu/.cloudflared/<UUID>.json

ingress:
  - hostname: sky.yourdomain.com
    service: http://127.0.0.1:46580
  - service: http_status:404
```

The trailing `http_status:404` is required — it's the catch-all for any other hostname hitting this tunnel.

### 5.7 Install as a system service

`cloudflared service install` runs as root, which means it looks for config at `/etc/cloudflared/`, not `~/.cloudflared/`. Copy the files over and fix the credentials path:

```bash
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/config.yml /etc/cloudflared/config.yml
sudo cp ~/.cloudflared/<UUID>.json /etc/cloudflared/
sudo sed -i 's|/home/ubuntu/.cloudflared/|/etc/cloudflared/|' /etc/cloudflared/config.yml
```

Now install and start:

```bash
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl status cloudflared
```

You should see `active (running)`. If not, check `sudo journalctl -u cloudflared -f` — usually a typo in `config.yml` or a wrong UUID path.

### 5.8 Verify the tunnel from your laptop

```bash
curl https://sky.yourdomain.com/api/health
```

You should see the same JSON health response that `curl http://127.0.0.1:46580/api/health` returned on the VM. **At this point the endpoint is publicly reachable** — anyone on the internet who guesses the URL can hit it. Lock it down before doing anything else.

______________________________________________________________________

## Part 6: Cloudflare Access (SSO)

### 6.1 Add the application

In the Cloudflare Zero Trust dashboard ([one.dash.cloudflare.com](https://one.dash.cloudflare.com)):

1. **Access → Applications → Add an application**.
2. Pick **Self-hosted**.
3. **Application Configuration:**
   - **Application name:** `SkyPilot`
   - **Session Duration:** `1 month` for a personal lab, or `24 hours` if you want daily re-auth.
   - **Public hostnames:**
     - **Subdomain:** `sky`
     - **Domain:** `yourdomain.com`
     - **Path:** *leave empty* — protects the entire subdomain
4. **Authenticate with Cloudflare One Client:** leave **off** (only relevant if you're rolling out WARP).
5. Click **Next**.

### 6.2 Identity providers

Enable **One-time PIN** at minimum. This emails a 6-digit code to whatever email is typed at login — Cloudflare won't actually grant access unless that email matches your Allow rule (next step), so this is safe.

You can additionally enable Google, GitHub, etc. for one-click SSO. Skip for now if you want; you can add it later.

Click **Next**.

### 6.3 Add a policy

- **Policy name:** `Me`
- **Action:** `Allow`
- **Configure rules → Include:**
  - Selector: `Emails`
  - Value: your email address (you can list multiple — personal, work, etc.)

Click **Next** then **Add application**.

### 6.4 Verify Access is in front of the server

From your laptop:

```bash
curl -i https://sky.yourdomain.com/api/health
```

You should now get a `HTTP/2 302` redirect to `https://<your-team>.cloudflareaccess.com/cdn-cgi/access/login/...` — **not** the JSON response you got before. That's Access intercepting unauthenticated requests.

### 6.5 Sanity-check the policy

In a browser, visit `https://sky.yourdomain.com`. You'll get a Cloudflare login page asking for an email.

- Type your real email → you should receive a PIN → entering it grants access.
- Type a random email like `nobody@example.com` → either the request is blocked outright or a PIN is emailed but rejected.

The second case (PIN emailed but rejected) is normal behavior for One-time PIN — Cloudflare sends the code regardless, but only validates it for emails matching your Allow rule.

______________________________________________________________________

## Part 7: Connect your laptop

### 7.1 Install cloudflared

**macOS:**

```bash
brew install cloudflared
```

**Linux:**

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

### 7.2 Cache an Access cookie

```bash
cloudflared access login https://sky.yourdomain.com
```

Browser opens, enter your email, paste the PIN, done. The cookie is cached at `~/.cloudflared/`.

Verify:

```bash
cloudflared access curl https://sky.yourdomain.com/api/health
```

You should see the JSON health response. This confirms the auth handshake works end-to-end.

### 7.3 Install SkyPilot on your laptop

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc   # or ~/.bashrc on Linux
uv tool install --python 3.11 "skypilot-nightly"
sky --version
```

You don't need cloud extras on the laptop — the laptop just talks to the API server, which has the cloud creds.

### 7.4 Point sky at your server

```bash
sky api login -e https://sky.yourdomain.com
sky api info
```

If `sky api info` returns server details (version, status, your user), you're fully wired up.

### 7.5 Fallback: service tokens

If `sky api login` hangs or returns 403, SkyPilot's HTTP client probably isn't picking up the cloudflared SSO cookie. Use a service token instead.

In Zero Trust dashboard:

1. **Access → Service Auth → Service Tokens → Create Service Token**
2. Name it `skypilot-cli`, duration `Non-expiring` or `1 year`.
3. Copy the **Client ID** and **Client Secret** (the secret is shown only once).

Edit your Access policy and add a second include rule:

- **Include → Service Auth → select `skypilot-cli`**

Save.

On your laptop, set environment variables:

```bash
export CF_ACCESS_CLIENT_ID="your-id.access"
export CF_ACCESS_CLIENT_SECRET="your-secret"
```

Add those to your shell rc file so they persist. Then retry:

```bash
sky api login -e https://sky.yourdomain.com
sky api info
```

> **Why service tokens for CLI?** SSO cookies have session expiry; in the middle of a long-running launch, your auth could lapse and break the connection. Service tokens don't expire on a session timer. SSO is great for browser/dashboard use; service tokens are right for scripted/CLI use.

______________________________________________________________________

## Part 8: First job

From your laptop:

```bash
sky check
```

This queries your remote API server, which queries the clouds it has credentials for. RunPod (or whatever you set up) should show **enabled**.

Run a trivial job:

```bash
sky launch --infra runpod --gpus A100:1 -- nvidia-smi
```

If RunPod has capacity at the price point, SkyPilot will provision the pod, run `nvidia-smi`, and tear it down (or leave it up depending on your config). Watch logs streaming to your terminal.

Tear it down explicitly when done:

```bash
sky down <cluster-name>
```

For real workloads, write a `task.yaml`:

```yaml
resources:
  cloud: runpod
  accelerators: A100:1

setup: |
  pip install torch transformers

run: |
  python train.py
```

Then `sky launch task.yaml`.

______________________________________________________________________

## Operational notes

### Updating SkyPilot

On the VM:

```bash
uv tool upgrade skypilot-nightly
sudo systemctl restart skypilot-api   # or: sky api stop && sky api start --deploy --host 127.0.0.1
```

On your laptop:

```bash
uv tool upgrade skypilot-nightly
```

Keep client and server within a minor version of each other — SkyPilot guarantees compatibility between adjacent minor versions.

### Updating cloudflared

```bash
sudo cloudflared update
sudo systemctl restart cloudflared
```

### Costs to watch

The OCI VM and Cloudflare components are free at the scale described here. Real costs come from:

- **Jobs you launch on RunPod / AWS / GCP** — track via those providers' dashboards.
- **Cloud storage** if you use `sky storage` for artifacts.
- **Egress** is free up to 10 TB/month on OCI; rarely an issue.

Set billing alerts on your job-running clouds. SkyPilot can't prevent you from spinning up an H100 cluster overnight.

### Backups

The API server keeps state in `~/.sky/` on the VM (cluster registry, request logs, etc.). If the VM dies, you lose this. For a personal setup it's not catastrophic — your jobs run on other clouds and persist there — but you can `tar czf sky-backup.tgz ~/.sky/ ~/.cloudflared/ ~/.runpod/` periodically and stash it in object storage.

### Adding more users

Cloudflare Access free tier is up to 50 users. To add a teammate:

1. Edit the Access policy → add their email to the Include list.
2. They install cloudflared + SkyPilot on their laptop, run `cloudflared access login`, then `sky api login -e https://sky.yourdomain.com`.

For more granular per-user permissions inside SkyPilot itself (workspaces, RBAC), see SkyPilot's own auth docs. The Helm-based deployment has more multi-user features than this VM-based setup.

### Logs

- **SkyPilot API server logs:** `~/.sky/api_server/server.log` on the VM, or `journalctl -u skypilot-api -f` if you used systemd.
- **cloudflared logs:** `journalctl -u cloudflared -f`.
- **Cloudflare Access audit logs:** Zero Trust dashboard → Logs → Access.

______________________________________________________________________

## Troubleshooting

### `sky check` fails with credential errors on the VM

The shell session running the API server doesn't see the credentials. Most common cause: you ran `aws configure` in one session but `sky api start` is running in another (or as a systemd service that doesn't have the same env). Fix:

- For systemd, ensure the service runs as `User=ubuntu` and that `~/.aws/credentials`, `~/.config/gcloud/`, etc., exist for that user.
- Avoid putting credentials only in env vars unless you also export them in the systemd unit.

### `cloudflared` service won't start

```bash
sudo journalctl -u cloudflared -n 50
```

Common issues:

- `Cannot determine default configuration path` → config wasn't copied to `/etc/cloudflared/`. Re-run the `sudo cp` step in 5.7.
- `failed to parse credentials` → the `credentials-file` path in `/etc/cloudflared/config.yml` still points to `/home/ubuntu/`. The `sed` step in 5.7 fixes this.
- `tunnel not found` → wrong UUID in `config.yml`.

### `curl https://sky.yourdomain.com/api/health` returns Cloudflare 502 or 521

The tunnel is up but cloudflared can't reach the SkyPilot server on `127.0.0.1:46580`.

- Check `sudo systemctl status skypilot-api` (or run `curl http://127.0.0.1:46580/api/health` on the VM).
- If the API server isn't running, restart it.

### `sky api login` hangs forever

Almost always an auth problem. The cloudflared SSO cookie isn't being passed in the HTTP request. Solution: set up service tokens (Part 7.5).

### `sky launch` works but logs don't stream

Cloudflare Tunnel sometimes has issues with very long-lived streaming connections on older `cloudflared` versions. Update with `sudo cloudflared update` and restart. If still broken, you can usually tail logs directly with `sky logs <cluster> --no-follow` and re-run.

### A1 capacity disappeared and OCI killed my VM

OCI sometimes reclaims always-free A1 instances if they appear idle for too long. Mitigations:

- Keep the VM mildly busy (the SkyPilot server itself usually does this).
- Convert the always-free instance to paid (pennies per month).
- Keep your `~/.cloudflared/` and `~/.sky/` configs backed up so you can rebuild quickly.

______________________________________________________________________

## Appendix: full command checklist

For a fresh build, in order:

```bash
# === On OCI VM (Ubuntu 24.04, ARM64) ===

# System
sudo apt update && sudo apt upgrade -y && sudo reboot
# (SSH back in)

# uv + SkyPilot
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv tool install --python 3.11 --with pip "skypilot-nightly[runpod,kubernetes]"

# RunPod creds
mkdir -p ~/.runpod
cat > ~/.runpod/config.toml <<'EOF'
api_key = "YOUR_RUNPOD_KEY"
EOF
chmod 600 ~/.runpod/config.toml
sky check

# SkyPilot API server (one-shot; for systemd, see Part 4.3)
sky api start --deploy --host 127.0.0.1
curl http://127.0.0.1:46580/api/health

# cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
cloudflared tunnel login
cloudflared tunnel create skypilot
cloudflared tunnel route dns skypilot sky.yourdomain.com

# Tunnel config (edit ~/.cloudflared/config.yml — see Part 5.6)
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/config.yml /etc/cloudflared/config.yml
sudo cp ~/.cloudflared/<UUID>.json /etc/cloudflared/
sudo sed -i 's|/home/ubuntu/.cloudflared/|/etc/cloudflared/|' /etc/cloudflared/config.yml
sudo cloudflared service install
sudo systemctl start cloudflared

# === In Cloudflare Zero Trust dashboard ===
# Add Self-hosted application for sky.yourdomain.com
# Enable One-time PIN, add Allow policy with your email

# === On laptop ===
brew install cloudflared           # or apt equivalent
cloudflared access login https://sky.yourdomain.com
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --python 3.11 "skypilot-nightly"
sky api login -e https://sky.yourdomain.com
sky api info
sky check
sky launch --infra runpod --gpus A100:1 -- nvidia-smi
```
