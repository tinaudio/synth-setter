# SkyPilot API Server on OCI Always-Free with k3s and Cloudflare Tunnel

A deployment playbook for a single-user SkyPilot API server with no recurring infrastructure cost.

## Stack

- **OCI Always-Free Ampere A1 VM** (Ubuntu 24.04 ARM64) — host
- **k3s** — single-node Kubernetes cluster
- **SkyPilot Helm chart** — API server with HTTP basic authentication via ingress-nginx
- **Cloudflare Tunnel** — outbound-only HTTPS exposure
- **HTTP basic authentication** — single-credential auth suitable for headless clients

The result is `https://sky.yourdomain.com`, accessible from any client capable of HTTP basic auth, including browsers, the SkyPilot CLI, the Python SDK, and CI runners.

This playbook does not cover Cloudflare Access, SSO/OAuth, or the bare-Docker SkyPilot deployment. These are documented elsewhere.

______________________________________________________________________

## Table of contents

01. [Prerequisites](#prerequisites)
02. [Architecture](#architecture)
03. [Part 1 — Provision the OCI VM](#part-1--provision-the-oci-vm)
04. [Part 2 — Install k3s and Helm](#part-2--install-k3s-and-helm)
05. [Part 3 — Deploy SkyPilot via Helm](#part-3--deploy-skypilot-via-helm)
06. [Part 4 — Configure Cloudflare Tunnel](#part-4--configure-cloudflare-tunnel)
07. [Part 5 — Add cloud credentials](#part-5--add-cloud-credentials)
08. [Part 6 — Connect from a client machine](#part-6--connect-from-a-client-machine)
09. [Part 7 — Headless and CI usage](#part-7--headless-and-ci-usage)
10. [Operational notes](#operational-notes)
11. [Troubleshooting](#troubleshooting)
12. [Appendix — Full command checklist](#appendix--full-command-checklist)

______________________________________________________________________

## Prerequisites

The following are required before beginning:

- An OCI tenancy with the Always-Free tier active. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/). A credit card is required for verification but is not charged for Always-Free resources.
- A domain managed through Cloudflare on the free plan. Domains can be registered through Cloudflare Registrar or transferred by updating nameservers at the existing registrar.
- A client machine running macOS, Linux, or WSL.
- API credentials for at least one cloud provider (RunPod, AWS, GCP, OCI, Lambda, etc.) where SkyPilot will launch jobs.

Throughout this document, the following placeholders should be replaced with actual values:

- `yourdomain.com` — the registered domain
- `sky.yourdomain.com` — the chosen subdomain for the API server
- `<vm-public-ip>` — the OCI VM's public IP address

______________________________________________________________________

## Architecture

```
┌──────────────┐  HTTPS  ┌──────────────┐ Tunnel  ┌────────────────────────────┐
│  Client      │────────▶│  Cloudflare  │────────▶│  OCI A1 VM (Ubuntu 24.04)  │
│  (CLI/SDK)   │ +basic  │     edge     │outbound │  ┌──────────────────────┐  │
└──────────────┘  auth   └──────────────┘         │  │  cloudflared         │  │
                                                  │  ↓                      │  │
                                                  │  ┌──────────────────────┐  │
                                                  │  │  k3s                 │  │
                                                  │  │   • ingress-nginx    │  │
                                                  │  │     (basic auth)     │  │
                                                  │  │   • SkyPilot API     │  │
                                                  │  │     server pod       │  │
                                                  │  └──────────────────────┘  │
                                                  └────────────┬───────────────┘
                                                               │
                                                       launches jobs on
                                                               │
                                                  ┌────────────▼───────────┐
                                                  │  RunPod / AWS / GCP /  │
                                                  │  OCI / Lambda / etc.   │
                                                  └────────────────────────┘
```

Key properties:

- No inbound ports are exposed on the VM. Cloudflare Tunnel initiates outbound connections to the Cloudflare edge. The OCI security list permits only SSH (port 22) for administration.
- TLS termination is handled at the Cloudflare edge using a Cloudflare-managed certificate.
- Authentication is enforced by ingress-nginx before requests reach the SkyPilot pod.
- Authentication uses HTTP basic auth, which is supported by all standard HTTP clients without additional tooling.

______________________________________________________________________

## References

SkyPilot:

- [SkyPilot documentation home](https://docs.skypilot.co/)
- [Deploying the API server (Helm)](https://docs.skypilot.co/en/latest/reference/api-server/api-server-admin-deploy.html)
- [Helm chart values reference](https://docs.skypilot.co/en/latest/reference/api-server/helm-values-spec.html)
- [Authentication and RBAC](https://docs.skypilot.co/en/latest/reference/auth.html)
- [Connecting to an API server](https://docs.skypilot.co/en/latest/reference/api-server/api-server.html)
- [API server troubleshooting](https://docs.skypilot.co/en/latest/reference/api-server/api-server-troubleshooting.html)
- [Python SDK reference](https://docs.skypilot.co/en/latest/reference/api.html)
- [CLI reference](https://docs.skypilot.co/en/latest/reference/cli.html)
- [SkyPilot YAML reference](https://docs.skypilot.co/en/latest/reference/yaml-spec.html)
- [Installation and cloud setup](https://docs.skypilot.co/en/latest/getting-started/installation.html)

Cloudflare:

- [Cloudflare Tunnel overview](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
- [Create a remotely-managed tunnel (dashboard)](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/)
- [Migrate a locally-managed tunnel to remote management](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/configure-tunnels/remote-management/)
- [cloudflared download and installation](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
- [Public hostnames](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/routing-to-tunnel/dns/)

Infrastructure components:

- [Oracle Cloud Always-Free resources](https://www.oracle.com/cloud/free/)
- [k3s lightweight Kubernetes](https://docs.k3s.io/)
- [Helm package manager](https://helm.sh/docs/)
- [ingress-nginx Helm chart values](https://artifacthub.io/packages/helm/ingress-nginx/ingress-nginx)
- [Oracle API key setup](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm)
- [RunPod API keys](https://docs.runpod.io/get-started/api-keys)
- [uv Python toolchain](https://docs.astral.sh/uv/)

______________________________________________________________________

## Part 1 — Provision the OCI VM

### 1.1 Create the instance

In the OCI console:

1. Navigate to **Compute → Instances → Create instance**.

2. **Name:** `skypilot-server`.

3. **Image:** select **Canonical Ubuntu 24.04** (ARM64).

4. **Shape:** select **Ampere → VM.Standard.A1.Flex** with the following configuration:

   - **OCPUs:** `4`
   - **Memory (GB):** `24`

   These values are the maximum for the Always-Free tier. The SkyPilot pod will not schedule on configurations smaller than 2 OCPUs / 4 GiB after accounting for k3s system overhead.

5. **Networking:** retain defaults (public subnet, auto-assigned public IP).

6. **SSH keys:** upload a public key, or download a generated key pair.

7. **Boot volume:** 100 GB.

8. Click **Create**.

### 1.2 Handling capacity errors

Always-Free A1 capacity is frequently exhausted in popular regions. If the request returns "Out of host capacity", consider the following:

- Try a different availability domain within the same region.
- Try a less-utilized region (Tokyo, Mumbai, São Paulo).
- Retry periodically; capacity becomes available sporadically.
- Use the OCI API with retry logic to automate this.

### 1.3 Connect via SSH

```bash
ssh ubuntu@<vm-public-ip>
```

### 1.4 Update the system

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

Reconnect after approximately 30 seconds.

### 1.5 Configure iptables for k3s networking

The default Ubuntu image on OCI ships with restrictive iptables rules. The following rules are required for k3s pod networking:

```bash
sudo iptables -I INPUT 1 -i lo -j ACCEPT
sudo iptables -I FORWARD 1 -i cni0 -j ACCEPT
sudo iptables -I FORWARD 1 -o cni0 -j ACCEPT
sudo netfilter-persistent save
```

The OCI Security List does not require modification. Cloudflare Tunnel uses outbound connections only.

______________________________________________________________________

## Part 2 — Install k3s and Helm

### 2.1 Install k3s

```bash
curl -sfL https://get.k3s.io | sh -
```

This installs the Kubernetes API server, kubelet, and containerd as a single binary managed by systemd.

### 2.2 Make the kubeconfig readable

```bash
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
```

### 2.3 Persist environment variables

```bash
cat >> ~/.bashrc <<'EOF'
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
export NAMESPACE=skypilot
export RELEASE_NAME=skypilot
EOF
source ~/.bashrc
```

Verify:

```bash
kubectl get nodes
```

The node should report a status of `Ready`.

### 2.4 Install Helm

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

### 2.5 Add the SkyPilot Helm repository

```bash
helm repo add skypilot https://helm.skypilot.co
helm repo update
```

### 2.6 Install htpasswd

```bash
sudo apt install -y apache2-utils
```

This is required for generating basic auth credentials in Part 3.

______________________________________________________________________

## Part 3 — Deploy SkyPilot via Helm

### 3.1 Generate basic auth credentials

```bash
WEB_USERNAME=skypilot
WEB_PASSWORD=$(openssl rand -hex 16)
echo "USERNAME: $WEB_USERNAME"
echo "PASSWORD: $WEB_PASSWORD"

AUTH_STRING=$(htpasswd -nb $WEB_USERNAME $WEB_PASSWORD)
```

The password should be stored in a password manager or equivalent. It cannot be recovered, only reset.

Note: `openssl rand -hex` produces only hexadecimal characters, which avoids URL-encoding requirements when embedding the password in `https://user:pass@host` connection strings. Passwords containing `+`, `/`, `=`, `@`, `:`, or other reserved URL characters require percent-encoding.

### 3.2 Install the chart

```bash
helm upgrade --install $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE \
  --create-namespace \
  --set ingress.authCredentials="$AUTH_STRING" \
  --set ingress-nginx.controller.service.type=NodePort \
  --set ingress-nginx.controller.service.nodePorts.http=30080 \
  --set apiService.resources.requests.cpu=2 \
  --set apiService.resources.requests.memory=4Gi \
  --set apiService.resources.limits.cpu=3 \
  --set apiService.resources.limits.memory=8Gi
```

Flag descriptions:

- `ingress.authCredentials` — htpasswd-formatted credentials used by ingress-nginx for basic auth enforcement.
- `ingress-nginx.controller.service.type=NodePort` — exposes ingress-nginx on a host port. The default `LoadBalancer` type requires cloud LoadBalancer integration not available on a single-node k3s cluster.
- `nodePorts.http=30080` — pins the NodePort to a stable value for cloudflared configuration.
- `apiService.resources.*` — overrides the chart's default resource requests (4 CPU / 8 Gi), which exceed the available capacity on a 4-OCPU node after k3s system overhead.

The double quotes around `"$AUTH_STRING"` are required because htpasswd output contains `$` characters that would otherwise be interpreted by the shell.

### 3.3 Wait for pods to start

```bash
kubectl get pods -n $NAMESPACE -w
```

Two pods will be created: `skypilot-api-server-...` and `skypilot-ingress-nginx-controller-...`. The API server pod requires 2-5 minutes on first deployment due to image pull and initialization. The deployment is complete when the API server pod reports `2/2 Running` (the second container is a logrotate sidecar).

### 3.4 Verify locally

```bash
# Expected: HTTP 401 with WWW-Authenticate: Basic header
curl -i http://localhost:30080/api/health

# Expected: HTTP 200 with health JSON
curl -i -u "${WEB_USERNAME}:${WEB_PASSWORD}" http://localhost:30080/api/health
```

If the unauthenticated request returns 200, the auth credentials were not applied. Inspect the deployed values:

```bash
helm get values skypilot -n $NAMESPACE
```

If `ingress.authCredentials` is missing, re-run the install command with `--reuse-values --set ingress.authCredentials="$AUTH_STRING"` after confirming `$AUTH_STRING` is non-empty.

______________________________________________________________________

## Part 4 — Configure Cloudflare Tunnel

### 4.1 Verify domain is on Cloudflare

In the Cloudflare dashboard, the domain should display status **Active**. If not, update nameservers at the registrar to the two values shown in the Cloudflare dashboard and wait for propagation (typically under one hour).

### 4.2 Install cloudflared on the VM

For ARM64 (OCI A1):

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

### 4.3 Create a tunnel via the Cloudflare dashboard

Tunnels created via the dashboard are remotely managed, allowing route configuration through the UI. Tunnels created via the `cloudflared tunnel create` CLI command are locally managed and require config file edits on the VM.

In the Zero Trust dashboard at [one.dash.cloudflare.com](https://one.dash.cloudflare.com):

1. Navigate to **Networks → Tunnels → Create a tunnel**.
2. **Connector type:** Cloudflared.
3. **Tunnel name:** `skypilot`.
4. Save the tunnel.
5. On the installation screen, copy the install token (the value following `--token` in the displayed command).

### 4.4 Install the tunnel as a system service

On the VM:

```bash
sudo cloudflared service install <YOUR_TOKEN>
sudo systemctl status cloudflared
```

The service should report `active (running)`. The tunnel in the dashboard should report status **HEALTHY** with one connector.

### 4.5 Configure the public hostname

In the dashboard, on the tunnel's configuration page:

1. Open the **Public Hostname** tab.
2. Click **Add a public hostname**.
3. Configure as follows:
   - **Subdomain:** `sky`
   - **Domain:** select from dropdown
   - **Path:** leave empty
   - **Service Type:** `HTTP`
   - **URL:** `localhost:30080`
4. Save.

Cloudflare automatically creates a CNAME DNS record pointing `sky.yourdomain.com` to the tunnel.

### 4.6 Verify external access

From a client machine:

```bash
# Expected: HTTP 401 with WWW-Authenticate: Basic header
curl -i https://sky.yourdomain.com/api/health

# Expected: HTTP 200 with health JSON
curl -i -u "skypilot:YOUR_PASSWORD" https://sky.yourdomain.com/api/health
```

The full request path is: client → Cloudflare edge → tunnel → cloudflared → ingress-nginx (auth check) → SkyPilot API server.

The dashboard at `https://sky.yourdomain.com` will prompt for basic auth in a browser.

______________________________________________________________________

## Part 5 — Add cloud credentials

The API server has no cloud credentials configured by default. Add credentials for each cloud provider that will be used.

### 5.1 RunPod

Generate an API key in the [RunPod console](https://www.runpod.io/console/user/settings) under Settings → API Keys, with Read & Write permissions.

```bash
kubectl create secret generic runpod-credentials \
  --namespace $NAMESPACE \
  --from-literal=api_key=YOUR_RUNPOD_API_KEY

helm upgrade --install $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE \
  --reuse-values \
  --set runpodCredentials.enabled=true
```

Wait for the pod to redeploy:

```bash
kubectl get pods -n $NAMESPACE -w
```

Verify:

```bash
POD=$(kubectl get pods -n $NAMESPACE -l app=${RELEASE_NAME}-api -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NAMESPACE $POD -c skypilot-api -- sky check runpod
```

Expected output: `RunPod: enabled`.

### 5.2 OCI

The Helm chart does not provide a built-in flag for OCI. Credentials must be mounted via a Kubernetes secret and extra volume mounts.

**Step 1 — Configure OCI credentials on the client machine.**

If `~/.oci/` is not already configured on the client, follow [Oracle's API key setup guide](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm). The result will be:

- `~/.oci/config`
- `~/.oci/oci_api_key.pem`

**Step 2 — Edit `~/.oci/config` to use a relative key path.**

The `key_file` value must be set to a relative path that resolves correctly inside the SkyPilot pod, which runs as the root user. Open `~/.oci/config` on the client and modify the `key_file` line as follows:

Original:

```
key_file=/Users/username/.oci/oci_api_key.pem
```

Modified:

```
key_file=~/.oci/oci_api_key.pem
```

The `~/` prefix resolves to `/root/.oci/oci_api_key.pem` inside the pod, where the key will be mounted.

**Step 3 — Copy credentials to the VM.**

From the client machine:

```bash
scp -r ~/.oci ubuntu@<vm-public-ip>:~/
```

**Step 4 — Create the Kubernetes secret on the VM.**

```bash
kubectl create secret generic oci-credentials \
  --namespace $NAMESPACE \
  --from-file=config=$HOME/.oci/config \
  --from-file=oci_api_key.pem=$HOME/.oci/oci_api_key.pem
```

**Step 5 — Mount the secret into the pod via Helm values.**

```bash
helm upgrade --install $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE \
  --reuse-values \
  --set 'apiService.extraVolumes[0].name=oci-creds' \
  --set 'apiService.extraVolumes[0].secret.secretName=oci-credentials' \
  --set 'apiService.extraVolumes[0].secret.defaultMode=384' \
  --set 'apiService.extraVolumeMounts[0].name=oci-creds' \
  --set 'apiService.extraVolumeMounts[0].mountPath=/root/.oci' \
  --set 'apiService.extraVolumeMounts[0].readOnly=true'
```

Note: `defaultMode=384` corresponds to file mode `0600` in decimal. The OCI SDK rejects private keys with broader permissions.

**Step 6 — Verify.**

```bash
kubectl get pods -n $NAMESPACE -w   # wait for 2/2 Running

POD=$(kubectl get pods -n $NAMESPACE -l app=${RELEASE_NAME}-api -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NAMESPACE $POD -c skypilot-api -- ls -la /root/.oci/
kubectl exec -n $NAMESPACE $POD -c skypilot-api -- sky check oci
```

Expected output: `OCI: enabled`.

If the verification fails with the error "key_file's value '/home/ubuntu/...' must be a valid file path", the relative path edit in step 2 was missed. Correct the file locally, then:

```bash
kubectl delete secret oci-credentials -n $NAMESPACE
kubectl create secret generic oci-credentials \
  --namespace $NAMESPACE \
  --from-file=config=$HOME/.oci/config \
  --from-file=oci_api_key.pem=$HOME/.oci/oci_api_key.pem
kubectl rollout restart deployment/skypilot-api-server -n $NAMESPACE
```

### 5.3 AWS

```bash
kubectl create secret generic aws-credentials \
  --namespace $NAMESPACE \
  --from-literal=aws_access_key_id=YOUR_ACCESS_KEY_ID \
  --from-literal=aws_secret_access_key=YOUR_SECRET_ACCESS_KEY

helm upgrade --install $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE \
  --reuse-values \
  --set awsCredentials.enabled=true
```

### 5.4 GCP

```bash
kubectl create secret generic gcp-credentials \
  --namespace $NAMESPACE \
  --from-file=gcp-cred.json=PATH_TO_YOUR_SERVICE_ACCOUNT_JSON

helm upgrade --install $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE \
  --reuse-values \
  --set gcpCredentials.enabled=true \
  --set gcpCredentials.projectId=YOUR_GCP_PROJECT_ID
```

### 5.5 Lambda

```bash
kubectl create secret generic lambda-credentials \
  --namespace $NAMESPACE \
  --from-literal=api_key=YOUR_LAMBDA_API_KEY

helm upgrade --install $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE \
  --reuse-values \
  --set lambdaCredentials.enabled=true
```

### 5.6 Verify all configured clouds

```bash
POD=$(kubectl get pods -n $NAMESPACE -l app=${RELEASE_NAME}-api -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NAMESPACE $POD -c skypilot-api -- sky check
```

All configured clouds should report `enabled`.

______________________________________________________________________

## Part 6 — Connect from a client machine

### 6.1 Install the SkyPilot client

The client only communicates with the API server, so cloud-specific extras are not required on client machines.

Using `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc   # or ~/.bashrc on Linux
uv tool install --python 3.11 --with pip "skypilot-nightly"
sky --version
```

Or using pip:

```bash
pip install "skypilot-nightly"
```

### 6.2 Authenticate

```bash
sky api login -e "https://${WEB_USERNAME}:${WEB_PASSWORD}@sky.yourdomain.com"
sky api info
```

`sky api info` should print server version and status.

### 6.3 Persist for SDK and CLI use

Add the following to the shell rc file (`~/.zshrc`, `~/.bashrc`):

```bash
export SKYPILOT_API_SERVER_ENDPOINT="https://skypilot:YOUR_PASSWORD@sky.yourdomain.com"
```

When this environment variable is set, `sky` CLI invocations and Python SDK calls automatically target the remote server. The variable takes precedence over any endpoint configured via `sky api login`, which makes it the recommended approach for any environment running automated jobs.

After setting the variable, reload the shell:

```bash
source ~/.zshrc   # or source ~/.bashrc
```

#### CLI usage

All `sky` commands automatically use the remote API server:

```bash
sky api info             # confirm connection
sky check                # list enabled clouds on the server
sky status               # list current clusters
sky launch task.yaml     # submit a job
sky logs <cluster>       # tail logs
sky down <cluster>       # tear down
```

No login step or token management is required. The CLI authenticates transparently via the credentials embedded in the URL.

#### Python SDK usage

The SDK reads the same environment variable. No additional configuration is required in code:

```python
import sky

# Confirm connection to the remote server
print(sky.api_info())

# Define and launch a task
task = sky.Task(
    name="example",
    setup="pip install numpy",
    run="python -c 'import numpy; print(numpy.__version__)'",
)
task.set_resources(sky.Resources(infra="runpod", accelerators="T4:1"))

request_id = sky.launch(task, cluster_name="example")
sky.stream_and_get(request_id)

# Tear down when finished
sky.down("example")
```

For programmatic use cases where the environment variable cannot be set (for example, multiple SkyPilot clients in the same Python process targeting different servers), the endpoint can be configured per-process:

```python
import os
os.environ["SKYPILOT_API_SERVER_ENDPOINT"] = "https://skypilot:YOUR_PASSWORD@sky.yourdomain.com"

import sky   # import after setting the env var
```

The environment variable must be set before `sky` is imported; SkyPilot reads it at module load time.

#### Verifying the active endpoint

At any time, the currently active endpoint can be confirmed:

```bash
sky api info
```

The output includes the endpoint URL, server version, and authentication status.

### 6.4 Run a job

```bash
sky check
sky launch --infra runpod --gpus T4:1 -- nvidia-smi
sky down -y sky-cmd
```

Python SDK equivalent:

```python
import sky

task = sky.Task(run="echo hello && nvidia-smi")
task.set_resources(sky.Resources(infra="runpod", accelerators="T4:1"))

request_id = sky.launch(task, cluster_name="smoke")
sky.stream_and_get(request_id)

sky.down("smoke")
```

______________________________________________________________________

## Part 7 — Headless and CI usage

HTTP basic auth is supported natively by all standard HTTP clients. No additional tooling is required for headless or CI environments beyond setting the endpoint environment variable.

### 7.1 GitHub Actions

```yaml
- name: Run SkyPilot job
  env:
    SKYPILOT_API_SERVER_ENDPOINT: https://skypilot:${{ secrets.SKYPILOT_PASSWORD }}@sky.yourdomain.com
  run: |
    pip install skypilot-nightly
    sky check
    sky launch -y task.yaml
```

The password should be stored in repository secrets. The username `skypilot` may be included in the URL.

### 7.2 Cron and shell scripts

```bash
#!/bin/bash
export SKYPILOT_API_SERVER_ENDPOINT="https://skypilot:${SKY_PASS}@sky.yourdomain.com"
sky launch -y nightly-job.yaml
```

The `SKY_PASS` value should be sourced from a secrets manager, environment file (excluded from version control), or equivalent.

### 7.3 URL encoding for special characters

Passwords containing reserved URL characters (`@`, `:`, `/`, `#`, `?`, `+`, `=`) must be percent-encoded when embedded in connection strings. The recommended approach is to generate passwords using `openssl rand -hex` to avoid this requirement.

To rotate to a URL-safe password:

```bash
NEW_PASSWORD=$(openssl rand -hex 16)
NEW_AUTH=$(htpasswd -nb skypilot $NEW_PASSWORD)
helm upgrade $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE --reuse-values \
  --set ingress.authCredentials="$NEW_AUTH"
echo "New password: $NEW_PASSWORD"
```

______________________________________________________________________

## Operational notes

### Updating SkyPilot

On the VM:

```bash
helm repo update
helm upgrade $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE --reuse-values
```

On client machines:

```bash
uv tool upgrade skypilot-nightly
```

Client and server should be kept within one minor version of each other.

### Updating cloudflared

```bash
sudo cloudflared update
sudo systemctl restart cloudflared
```

### Rotating the basic auth password

```bash
NEW_PASSWORD=$(openssl rand -hex 16)
NEW_AUTH=$(htpasswd -nb skypilot $NEW_PASSWORD)
helm upgrade $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE --reuse-values \
  --set ingress.authCredentials="$NEW_AUTH"
```

After rotation, update `SKYPILOT_API_SERVER_ENDPOINT` in all locations where it is configured (client shell rc files, CI secrets, scheduled job environments).

### Cost considerations

The infrastructure described in this playbook incurs no recurring cost. Costs are generated only by:

- Jobs launched on RunPod, AWS, GCP, or other cloud providers (tracked via those providers' billing dashboards).
- Cloud storage usage when `sky storage` is configured.

Billing alerts should be configured on each cloud provider account. SkyPilot does not enforce spending limits.

### Backups

The API server stores state in a PersistentVolumeClaim within k3s. To create a backup:

```bash
POD=$(kubectl get pods -n $NAMESPACE -l app=${RELEASE_NAME}-api -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NAMESPACE $POD -c skypilot-api -- tar czf - /root/.sky | cat > sky-backup.tgz
```

Store the resulting archive in object storage or another durable location.

### Logs

| Component        | Command                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| SkyPilot API pod | `kubectl logs -n $NAMESPACE $POD -c skypilot-api -f`                    |
| ingress-nginx    | `kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=ingress-nginx -f` |
| cloudflared      | `sudo journalctl -u cloudflared -f`                                     |
| k3s              | `sudo journalctl -u k3s -f`                                             |

### OCI idle reclaim policy

OCI may reclaim Always-Free instances that remain idle for seven consecutive days (95th-percentile CPU below 20%). The combined activity of k3s and cloudflared is generally sufficient to remain above this threshold. If a reclaim notice is received, a periodic keepalive can be added:

```bash
echo '*/30 * * * * root timeout 30 yes > /dev/null' | sudo tee /etc/cron.d/keepalive
```

### Multi-user access

This deployment uses a single shared credential. Multi-user access with per-user permissions requires enabling the chart's OAuth/SSO configuration via `auth.oauth.*` Helm values, which is out of scope for this playbook.

______________________________________________________________________

## Troubleshooting

### Pod stuck in Pending status with "Insufficient cpu/memory"

The pod's resource requests exceed available capacity. Verify with:

```bash
kubectl describe pod -n $NAMESPACE <pod-name>
```

The Events section will report the specific resource constraint. Reduce requests:

```bash
helm upgrade $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE --reuse-values \
  --set apiService.resources.requests.cpu=1 \
  --set apiService.resources.requests.memory=2Gi
```

If reduced requests still fail to schedule, verify the node's actual capacity:

```bash
kubectl describe node | grep -A 5 Allocatable
```

The output should show `cpu: 4` and `memory: ~23Gi` for a correctly-sized A1 instance. If smaller values are reported, the VM was provisioned with insufficient resources. Resize via the OCI console: stop the instance, edit shape configuration to 4 OCPU / 24 GB, and start the instance.

### Pod in CrashLoopBackOff with "exec format error"

The image lacks ARM64 layers. Verify with:

```bash
docker manifest inspect berkeleyskypilot/skypilot-nightly:latest 2>/dev/null | grep arch
```

The output should list `arm64`. If only `amd64` is present, building from source is required. Current SkyPilot nightly images include ARM64 builds.

### External requests return HTTP 502 or 503

The tunnel reaches Cloudflare but cloudflared cannot connect to ingress-nginx. Verify on the VM:

```bash
curl -i http://localhost:30080/api/health      # should return 401
kubectl get svc -n $NAMESPACE                  # confirm NodePort is 30080
sudo systemctl status cloudflared              # confirm running
```

### External requests return HTTP 302 redirect to a Cloudflare login page

A Cloudflare Access application is protecting the hostname. To remove it:

1. Open the Zero Trust dashboard.
2. Navigate to **Access → Applications**.
3. Locate the application protecting `sky.yourdomain.com`.
4. Open the action menu and select **Delete**.

### `helm upgrade` returns "Kubernetes cluster unreachable"

The `KUBECONFIG` environment variable is not set in the current shell:

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
```

If this resolves the error but the variable is lost in subsequent sessions, the `~/.bashrc` modification from Part 2.3 was not applied.

### `kubectl get pods` returns "connection refused"

The command is being executed on the client machine rather than the VM. The k3s cluster is hosted on the VM. Either SSH into the VM, or configure remote kubectl access (out of scope).

### Authentication not enforced (HTTP 200 without credentials)

Inspect the deployed Helm values:

```bash
helm get values skypilot -n $NAMESPACE
```

If `ingress.authCredentials` is absent or null, the `$AUTH_STRING` variable was empty during install. Re-run with `--reuse-values --set ingress.authCredentials="$AUTH_STRING"` after confirming the variable is populated:

```bash
echo "$AUTH_STRING"
```

### Cloudflare tunnel reports "locally managed"

The tunnel was created via the `cloudflared tunnel create` CLI command rather than the dashboard. Routes are configured in `/etc/cloudflared/config.yml` rather than the UI.

To migrate the tunnel to remote management so routes can be edited in the dashboard:

```bash
# Ensure cloudflared is recent (2024.x or newer)
sudo cloudflared update
sudo systemctl restart cloudflared
cloudflared --version

# Stop the tunnel
sudo systemctl stop cloudflared

# Run the migration
sudo cloudflared tunnel migrate --config /etc/cloudflared/config.yml

# Restart
sudo systemctl start cloudflared
sudo systemctl status cloudflared
```

After migration, the dashboard's Public Hostname tab becomes editable and the "locally managed" notice is removed.

If `cloudflared tunnel migrate` fails or is unavailable, perform a manual migration:

1. Record the current routes from `/etc/cloudflared/config.yml` (the `hostname` and `service` pairs).
2. In the Cloudflare Zero Trust dashboard, delete the existing tunnel.
3. Create a new tunnel via the dashboard (Networks → Tunnels → Create a tunnel) and copy the new install token.
4. On the VM:
   ```bash
   sudo cloudflared service uninstall
   sudo rm /etc/cloudflared/config.yml
   sudo cloudflared service install <NEW_TOKEN>
   ```
5. In the dashboard, recreate the public hostname routes recorded in step 1.

Manual migration causes brief downtime during the swap, typically under one minute.

The dashboard-based flow described in Part 4 avoids this scenario for new deployments.

### Old SkyPilot process still running

If `sky api start` was previously executed directly on the VM (e.g., via uv), the process may still be listening on port 46580. Identify with:

```bash
sudo ss -tlnp | grep 46580
```

If a non-root process appears, terminate it:

```bash
ps aux | grep "sky.server.server --host=127.0.0.1" | grep -v grep | awk '{print $2}' | xargs -r kill
```

The root-owned process inside the k3s pod must not be terminated.

### `sky check oci` reports invalid key file path

The relative key path edit in Part 5.2 step 2 was not applied. Correct `~/.oci/config` to use `key_file=~/.oci/oci_api_key.pem`, then:

```bash
kubectl delete secret oci-credentials -n $NAMESPACE
kubectl create secret generic oci-credentials \
  --namespace $NAMESPACE \
  --from-file=config=$HOME/.oci/config \
  --from-file=oci_api_key.pem=$HOME/.oci/oci_api_key.pem
kubectl rollout restart deployment/skypilot-api-server -n $NAMESPACE
```

______________________________________________________________________

## Appendix — Full command checklist

Replace `<vm-public-ip>` and `yourdomain.com` placeholders with actual values.

### On the OCI VM

```bash
# Part 1: System preparation
sudo apt update && sudo apt upgrade -y
sudo reboot
# Reconnect via SSH

sudo iptables -I INPUT 1 -i lo -j ACCEPT
sudo iptables -I FORWARD 1 -i cni0 -j ACCEPT
sudo iptables -I FORWARD 1 -o cni0 -j ACCEPT
sudo netfilter-persistent save

# Part 2: k3s and Helm
curl -sfL https://get.k3s.io | sh -
sudo chmod 644 /etc/rancher/k3s/k3s.yaml

cat >> ~/.bashrc <<'EOF'
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
export NAMESPACE=skypilot
export RELEASE_NAME=skypilot
EOF
source ~/.bashrc

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm repo add skypilot https://helm.skypilot.co
helm repo update
sudo apt install -y apache2-utils

# Part 3: SkyPilot deployment
WEB_USERNAME=skypilot
WEB_PASSWORD=$(openssl rand -hex 16)
echo "Password: $WEB_PASSWORD"
AUTH_STRING=$(htpasswd -nb $WEB_USERNAME $WEB_PASSWORD)

helm upgrade --install $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE \
  --create-namespace \
  --set ingress.authCredentials="$AUTH_STRING" \
  --set ingress-nginx.controller.service.type=NodePort \
  --set ingress-nginx.controller.service.nodePorts.http=30080 \
  --set apiService.resources.requests.cpu=2 \
  --set apiService.resources.requests.memory=4Gi \
  --set apiService.resources.limits.cpu=3 \
  --set apiService.resources.limits.memory=8Gi

kubectl get pods -n $NAMESPACE -w   # Wait for 2/2 Running

curl -i -u "${WEB_USERNAME}:${WEB_PASSWORD}" http://localhost:30080/api/health

# Part 4: Cloudflare Tunnel
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb

# Create tunnel via Cloudflare Zero Trust dashboard, copy install token
sudo cloudflared service install <YOUR_TOKEN>

# In dashboard: configure public hostname sky.yourdomain.com → http://localhost:30080

# Part 5: Cloud credentials (RunPod example)
kubectl create secret generic runpod-credentials \
  --namespace $NAMESPACE \
  --from-literal=api_key=YOUR_RUNPOD_KEY

helm upgrade --install $RELEASE_NAME skypilot/skypilot-nightly --devel \
  --namespace $NAMESPACE \
  --reuse-values \
  --set runpodCredentials.enabled=true

POD=$(kubectl get pods -n $NAMESPACE -l app=${RELEASE_NAME}-api -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NAMESPACE $POD -c skypilot-api -- sky check
```

### On the client machine

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
uv tool install --python 3.11 --with pip "skypilot-nightly"

curl -i -u "skypilot:YOUR_PASSWORD" https://sky.yourdomain.com/api/health

echo 'export SKYPILOT_API_SERVER_ENDPOINT="https://skypilot:YOUR_PASSWORD@sky.yourdomain.com"' >> ~/.zshrc
source ~/.zshrc

sky api info
sky check

sky launch --infra runpod --gpus T4:1 -- nvidia-smi
sky down -y sky-cmd
```
