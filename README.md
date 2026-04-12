# MAizeBargAIn Purple Agent

Purple agent for the **MAizeBargAIn** Meta-Game Negotiation Assessor on AgentBeats (Sprint 2, Multi-agent Evaluation track).

Built on the same scaffolding as our Sprint 1 Build-What-I-Mean agent. The strategy:
- **LLM reasoning** (OpenAI `o4-mini` by default, or any Claude model via `LLM_PROVIDER=anthropic`) for strategic decisions.
- **Deterministic M1–M5 validator** wraps every LLM response and auto-corrects or rejects rule-violating actions.
- **Provably-safe heuristic fallback** ensures we always emit a feasible, M1–M5-compliant action even if the LLM and its retry both fail.

## The five mistakes we protect against

The green agent's README calls these out explicitly as the things LLM negotiators get wrong:

| Code | Mistake |
|---|---|
| M1 | Making an offer worse for you than your previous offer |
| M2 | Making an offer worse for you than your BATNA |
| M3 | Proposing extreme divisions (all or nothing) |
| M4 | Accepting an offer worse for you than your BATNA |
| M5 | Walking away from an offer better for you than your BATNA |

Every one of these is deterministically checkable given the game state, so we check them in `agent/negotiation.py::validate_action` and either auto-fix (e.g., swap WALK → ACCEPT when the offer actually beats BATNA) or retry the LLM with explicit feedback.

## Layout

```
.
├── server.py                  # A2A entrypoint
├── core/
│   ├── executor.py            # A2A plumbing (one Agent per context)
│   ├── llm.py                 # OpenAI / Anthropic wrapper
│   └── parser.py              # Observation extraction + action serialization
├── agent/
│   ├── agent.py               # Decision pipeline
│   ├── state.py               # Per-session state
│   ├── negotiation.py         # Offer math, BATNA value, M1-M5 validator
│   └── prompts.py             # Circle-5-style LLM prompts
├── Dockerfile
├── amber.json5
└── .github/workflows/build-and-publish.yml
```

## Run locally

```bash
# Install
uv sync

# Set keys (default provider is OpenAI)
export OPENAI_API_KEY=sk-...

# Or use Claude:
# export LLM_PROVIDER=anthropic
# export ANTHROPIC_API_KEY=sk-ant-...
# export ANTHROPIC_MODEL=claude-sonnet-4-20250514

uv run python server.py --host 0.0.0.0 --port 9018
```

Verify:
```bash
curl http://localhost:9018/.well-known/agent-card.json
```

## Submission to the Sprint 2 leaderboard

1. Push this repo to GitHub. The CI builds a Docker image to `ghcr.io/<your-org>/<repo>:latest`.
2. On [agentbeats.dev](https://agentbeats.dev) → **Register Agent** → select Purple → paste your image URL.
3. Submit against the **Meta-Game Negotiation Assessor** (MAizeBargAIn green agent). The green agent's README recommends `challenger_circle=5`.
4. Fill out the [Sprint 2 submission form](https://docs.google.com/forms/d/e/1FAIpQLSeN2IAXtM1h-XjYQO3rb9fL9OnMCpVzHTMB4LaVTytJ7sdjRw/viewform) before 11:59pm PT April 12.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | Required if provider=openai |
| `OPENAI_MODEL` | `o4-mini` | |
| `ANTHROPIC_API_KEY` | — | Required if provider=anthropic |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | |
| `LLM_TEMPERATURE` | `0.0` | |
| `LLM_MAX_TOKENS` | `1024` | |
