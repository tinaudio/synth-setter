# Security Policy

## Project Status

synth-setter is a **research/pre-release project** (0.x). APIs, data formats, and
interfaces may change without notice. That said, we take security seriously and
welcome responsible vulnerability reports.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Use [GitHub's private vulnerability reporting](https://github.com/tinaudio/synth-setter/security/advisories/new)
to submit a report. This keeps the details confidential until a fix is available.

### What to Include

- A description of the vulnerability and its potential impact.
- Steps to reproduce, including code snippets or configuration if applicable.
- The affected component(s) (e.g., module path, Docker image, CI workflow).
- Any suggested fix or mitigation, if you have one.

## Response Timeline

- **Acknowledgement:** within 48 hours of receiving the report.
- **Triage:** within 7 days we will assess severity and confirm whether the
  report is accepted or declined.
- **Resolution:** timeline depends on severity and complexity. We will keep
  you informed of progress.

## Scope

### In Scope

- The synth-setter codebase (`src/`, `pipeline/`, `scripts/`, `configs/`).
- Published Docker images.
- CI/CD workflows (`.github/workflows/`).

### Out of Scope

- Third-party dependencies (report these to the upstream project).
- RunPod infrastructure and Cloudflare R2 service configuration.
- Vulnerabilities that require physical access to a machine running the project.
