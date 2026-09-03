# Pi model policy for synth-setter

Do not select Anthropic providers or models in this repository.
Do not launch subagents whose configured or inferred provider/model resolves to Anthropic.
Default subagent `model` arguments to `openai-codex/gpt-5.6-sol`. Never pass the provider-only `openai-codex`; model scope requires a fully qualified `provider/model-id` selector.
Use `openrouter` only for the exact pinned GLM-5.3-Flash secondary-review model in `.pi/settings.json` and `agent/_shared/pi_review_routing.py`.
If a task cannot be completed under that provider policy, stop and explain the constraint instead of switching to Anthropic.
