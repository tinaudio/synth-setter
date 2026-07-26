# Continuous Profiling Reference

The runtime image ships [Grafana Alloy](https://github.com/grafana/alloy), configured to collect
CPU profiles with `pyroscope.ebpf` and forward them to Grafana Cloud Pyroscope.

Profiling is **off by default** and requires no Python code changes: the eBPF profiler attaches
from outside the interpreter and symbolizes Python frames itself. Only CPU profiles are shipped —
no metrics remote-write and no log forwarding. Run metrics remain W&B's job
([W&B integration](wandb-integration.md)).

## 1. What is baked into the image

| Path                                      | Purpose                                                    |
| ----------------------------------------- | ---------------------------------------------------------- |
| `/usr/local/bin/alloy`                    | Alloy binary, pinned and SHA256-verified at build          |
| `/etc/alloy/profiling.alloy`              | `discovery.process` → `pyroscope.ebpf` → `pyroscope.write` |
| `/usr/local/bin/start_alloy_profiling.sh` | Opt-in launcher with a preflight                           |

No credential enters an image layer. The config resolves every secret through `sys.env` when Alloy
starts, so the published image stays safe on public registries.

The config keeps only processes whose executable is the runtime venv interpreter
(`/venv/main/bin/python*`), which is what every training and pipeline entrypoint runs under. Alloy
itself, apt's `python3`, and shell helpers are dropped before profiles are sent.

## 2. Requirements

`pyroscope.ebpf` fails **silently** — an unmet precondition yields zero profiles rather than an
error. The launcher therefore refuses to start unless all of the following hold:

| Requirement           | Why                                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| Runs as root          | Loading eBPF programs and reading `/proc/<pid>/` of other processes                                   |
| Host PID namespace    | eBPF reports host-side PIDs; in a nested namespace they match no `/proc` entry, so every target drops |
| `/sys/kernel/tracing` | The tracer attaches tracepoints through tracefs                                                       |
| Credentials set       | Endpoint, user, and API key (below)                                                                   |

Each failure prints a specific diagnostic naming the remedy. This is the point of the preflight:
without it, a misconfigured pod looks healthy while collecting nothing.

Beyond the `SYS_ADMIN` the devcontainers already grant for FUSE, the eBPF unwinder needs `BPF`,
`PERFMON`, `SYS_PTRACE`, `SYS_RESOURCE`, and `DAC_READ_SEARCH`. Every
`.devcontainer/*/devcontainer.json` grants these.

## 3. Compute-backend support

| Backend            | Container launch                                                                     | Profiling         |
| ------------------ | ------------------------------------------------------------------------------------ | ----------------- |
| OCI                | `oci-docker-run.sh`, which passes `--privileged --pid=host`                          | Supported         |
| Local `docker run` | Yours to control                                                                     | Supported         |
| Local devcontainer | Needs one manual flag (below)                                                        | Opt-in            |
| **RunPod / Vast**  | SkyPilot sets `image_id: docker:<image>`; the PID namespace is not ours to configure | **Not supported** |

RunPod and Vast pods run in their own PID namespace and expose no knob to change it, so
`pyroscope.ebpf` cannot correlate its targets there. The launcher detects this and exits with a
diagnostic rather than pretending to work. Profiling a RunPod workload would require the in-process
Pyroscope SDK instead — a different design, not enabled here.

### Devcontainers

The capabilities are already granted, but `--pid=host` is **not** — it would let processes inside
the container see and signal every process on your machine, which is a poor default for a
dev environment where agents run `pkill`-style cleanup. To profile locally, add it deliberately to
the flavor you use:

```jsonc
// .devcontainer/gpu/devcontainer.json
"runArgs": ["--pid=host", ...]
```

## 4. Credentials

From the Grafana Cloud stack's Pyroscope details page. Template entries live in `.env.example`.

| Variable                           | Value                                               |
| ---------------------------------- | --------------------------------------------------- |
| `GRAFANA_CLOUD_PYROSCOPE_ENDPOINT` | e.g. `https://profiles-prod-000.grafana.net`        |
| `GRAFANA_CLOUD_PYROSCOPE_USER`     | Numeric stack / instance ID                         |
| `GRAFANA_CLOUD_PYROSCOPE_API_KEY`  | Access-policy token with the `profiles:write` scope |

Two knobs control the collector itself:

| Variable                              | Default        | Effect                                |
| ------------------------------------- | -------------- | ------------------------------------- |
| `SYNTH_SETTER_PROFILING_ENABLED`      | unset (off)    | `1`/`true`/`yes` starts the collector |
| `SYNTH_SETTER_PROFILING_SERVICE_NAME` | `synth-setter` | Groups profiles in Pyroscope          |

Set `SYNTH_SETTER_PROFILING_SERVICE_NAME` per workload (`synth-setter-train`,
`synth-setter-generate-dataset`) so flame graphs stay separable.

## 5. Running it

The launcher `exec`s Alloy, so background it alongside the workload:

```bash
docker run --rm --privileged --pid=host \
  -v /sys/kernel/tracing:/sys/kernel/tracing:ro \
  -e SYNTH_SETTER_PROFILING_ENABLED=1 \
  -e SYNTH_SETTER_PROFILING_SERVICE_NAME=synth-setter-train \
  -e GRAFANA_CLOUD_PYROSCOPE_ENDPOINT \
  -e GRAFANA_CLOUD_PYROSCOPE_USER \
  -e GRAFANA_CLOUD_PYROSCOPE_API_KEY \
  synth-setter:dev-snapshot \
  bash -c 'start_alloy_profiling.sh & synth-setter-train ...'
```

With profiling disabled the launcher logs one line and exits `0`, so the same command is safe to
use unconditionally.

Profiles appear in Grafana Cloud under the `process_cpu` profile type, filtered by
`service_name`. Expect the first samples within about a minute — `discovery.process` refreshes on a
60s interval, so a process that starts and exits inside one window is never discovered.

## 6. Troubleshooting

| Symptom                                 | Cause                                                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `needs the host PID namespace`          | Relaunch with `--pid=host`; on RunPod/Vast this is unavailable (§3)                                     |
| `must run as root`                      | The devcontainer's non-root default; use the `root_gpu` flavor or `sudo`                                |
| `tracefs is not mounted`                | Bind-mount `/sys/kernel/tracing` read-only                                                              |
| No profiles, no errors                  | The workload finished inside one 60s discovery window, or it does not run under `/venv/main/bin/python` |
| Frames show module names, not functions | Stripped binary — expected for some system libraries; Python frames are unaffected                      |

Verify the config after editing it, using the Alloy binary in the image:

```bash
alloy validate /etc/alloy/profiling.alloy
```
