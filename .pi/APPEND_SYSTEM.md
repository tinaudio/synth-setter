# Pi model policy for synth-setter

Do not select Anthropic providers or models in this repository.
Do not launch subagents whose configured or inferred provider/model resolves to Anthropic.
Use `openai-codex` by default. Use `kimi-coding` and `openrouter` only for the exact pinned free-pool review models in `.pi/settings.json` and `agent/_shared/pi_review_routing.py`.
If a task cannot be completed under that provider policy, stop and explain the constraint instead of switching to Anthropic.
