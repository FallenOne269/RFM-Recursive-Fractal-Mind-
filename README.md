# Recursive Fractal AI (RFAI)

This repository implements a lightweight end-to-end reference for **Recursive Fractal AI (RFAI)**. The system combines Adaptive Multiscale Iterated Function Systems (AMIFS), Dynamic Fractal Encoding (DFE), Recursive Structural Adaptation (RSA), graph-based fractal nodes, semantic goals, and a recursive core engine. Everything is deterministic, typed, and runs on Python 3.11+ with only `numpy` as a runtime dependency (optionally `cupy` if available for GPU acceleration).

## Architecture overview

### AMIFS
Adaptive Multiscale Iterated Function Systems generate fractal attractors using affine transforms. Parameters adapt to features and context using a cached adaptive component and history-aware contextual component. Precision is configurable for stability.

### Dynamic Fractal Encoding (DFE)
DFE converts arbitrary data into **Fractal Information Motifs (FIMs)** that pair a semantic vector with an AMIFS-generated pattern signature. A registry tracks FIM lifecycles and supports parallel encoding.

### Recursive Structural Adaptation (RSA)
RSA evaluates node performance versus targets and semantic goals to emit `AdaptationSignal`s. Signals adjust node complexity, trigger bifurcation (splitting) when complexity or error rises, and consolidation when stable. Safety caps enforce maximum graph depth and node counts with hysteresis to prevent oscillation.

### Memory layer & smart nodes
Each node keeps a bounded `MemoryLayer` of adaptation history. `SmartFractalNode` manages complexity updates, bifurcation, and consolidation heuristics. `SemanticSmartNode` augments adaptation using semantic goal similarity. `FractalGraph` maintains parent/child relations and propagates signals deterministically.

### Recursive core
`RecursiveFractalAlgorithm` provides a generic recursive processing loop with self-improvement hooks that adjust thresholds based on feedback. `FractalState` wraps state dictionaries with safe copying.

### Simulations
Three deterministic simulation environments model different challenges:
- **Scenario A (TaskHierarchyEnv):** ordered subtasks with dependencies.
- **Scenario B (ObstacleAvoidanceEnv):** moving obstacles on a grid with oscillation detection.
- **Scenario C (ResourceAllocationEnv):** allocating resources across compartments for balance.

`scenarios.py` orchestrates each scenario using the full stack, and `metrics.py` computes task completion, efficiency, recovery, coordination, oscillation, recursion depth, node counts, and bifurcations/consolidations.

## Usage

Install in editable mode:

```bash
pip install -e .
```

Run the demo to execute all scenarios and print a summary table:

```bash
python -m rfai.demo
```

Use the CLI for fine-grained control (scenario selection, steps, seeds, and RSA limits):

```bash
python -m rfai.cli run --scenario A --steps 200 --seed 0
python -m rfai.cli run --scenario all --steps 100 --max-depth 4 --max-nodes 20
```

Run tests:

```bash
pytest
```

## Design choices

- **Determinism first:** all randomness uses `numpy.random.Generator` with explicit seeds, and adaptation caches reuse computed deltas.
- **Safety caps:** RSA enforces `max_depth` and `max_nodes` to avoid unbounded growth, with hysteresis-based split/merge gating.
- **Lightweight patterns:** FIM metadata stores compact signatures (centroid and spread) instead of full point clouds to keep memory low.
- **CPU/GPU agnostic:** AMIFS automatically uses `cupy` if available; otherwise falls back to `numpy` while keeping interfaces identical.

## Resonant Fractal Cognition (RFC) — a new architecture derived from this research

Alongside the RFAI reference implementation, this repository now contains
**RFC**, a distinct cognitive architecture synthesised from the research
collected here (the FRM blueprint, the FRCL analysis, RFAI/RFIM, and the agent
specification in `AGENTS.md`).

RFC's premise is that belief should be an **interference pattern** rather than a
stored value. Evidence is split across a dyadic ladder of scales; hypotheses are
oscillators whose natural frequencies sit on that same ladder; a hypothesis
grows when evidence at *its* scale arrives in phase with it; and a claim only
scores well when its scales agree, because the read-out sums them as waves.

Where the source blueprint stacks three layers — a quantum-inspired substrate, a
neuro-symbolic middle, a metacognitive top — and then observes that integrating
them is the hard part, RFC has **one operator** applied at different scales to
different evidence. Perception, symbol formation, recursion on the residual, and
the system's reflection on its own telemetry are all the same code.

- Symbols crystallise out of sustained resonance and feed back as priors.
- Constraints are physics: a hypothesis violating an invariant is phase-inverted
  and cut off from drive, so its amplitude is provably non-increasing and it can
  never be selected.
- Metacognition re-runs the same operator over the system's own episode history,
  with bounded parameter nudges and rollback. It works out which rung of its own
  scale ladder to stop believing by asking, of each rung, whether removing it
  would have changed an answer that turned out to be wrong — credit graded by
  reward, never by whether the other rungs agree.

```bash
python -m rfc.demo                    # narrated walkthrough plus benchmarks
python -m rfc.cli bench               # baselines and ablations (~90s)
```

Measured results — including the two tasks where a five-line cosine baseline
wins, and the gap that remains between the self-tuning loop and a hand-set
configuration — are reported in full in
**[docs/RFC_ARCHITECTURE.md](docs/RFC_ARCHITECTURE.md)**.


## MCP server (custom app / remote connector)

`mcp_server/` exposes this repository over the [Model Context Protocol](https://modelcontextprotocol.io),
so an MCP client (e.g. a Sesame custom app or Claude Desktop) can query the codebase, run its
engines and trigger test runs.

| Tool | Purpose |
|------|---------|
| `project_overview` | README, layout, modules, docs, test files, git branch/HEAD/recent commits |
| `list_files` / `read_file` / `search_code` | Browse and search the repo (sandboxed to the repo root; skips `.git`, venvs, caches, `.env`/key files) |
| `describe_architecture` | `rfai`/`rfc`/`src` modules, RFAI scenarios, RFC default config + tunable parameters, orchestrator pipeline with config schemas |
| `run_rfai_scenario` | Run simulation A/B/C (or all) through AMIFS/DFE/RSA and return metrics |
| `run_rfc_episode` | Run Resonant Fractal Cognition over the composite task (accuracy, last percept, explanation; optional forbidden labels) |
| `run_orchestrator_cycle` | Run one `RFAISystem` cycle (fractal engine -> swarm -> quantum -> meta learner) on numeric input |
| `ask_gemini` / `list_gemini_models` | Optional: send a free-form prompt to Google Gemini (external LLM, **not** RFM's engines; results carry `rfm_engines_used: false`) / list models the key can call |
| `list_tests` | Collect pytest node ids |
| `run_tests` | Run pytest (optional `target`, `-k` `keyword`, `timeout_seconds`); `wait=false` returns a `run_id` immediately |
| `get_test_run` / `list_test_runs` | Poll a run or list recent runs with pass/fail counts, failure details and output tail |

Resources: `rfm://readme`, `rfm://agents`, `rfm://docs/rfc-architecture`.
Only one test run executes at a time. Run history is kept in memory for the lifetime of the server.

```bash
pip install -r requirements.txt numpy
python -m mcp_server                                            # stdio (local clients)
python -m mcp_server --transport streamable-http --port 8000    # HTTP at /mcp, localhost only
```

### Natural-language prompts via Gemini (optional)

RFM's engines only take numeric/structured input, so `ask_gemini` forwards free-form prompts
to Google Gemini instead. Answers are Gemini's reasoning, not RFM's. The key is read on the
server at call time (no restart needed), sent only in the `x-goog-api-key` header, and never
logged or returned. HTTP 429 (free-tier quota) is reported as an error; there is no paid fallback.

```bash
export GEMINI_API_KEY=...                        # or GEMINI_API_KEY_FILE=/path/to/0600-file
export GEMINI_MODEL=gemini-3.8-flash             # optional; default gemini-3.8-flash
```

A key from a Google AI Studio project without billing stays on the free tier (rate-limited,
and prompts may be used to improve Google's products). See https://ai.google.dev/gemini-api/docs/pricing.

### Remote access with OAuth (Sesame custom app)

Set `RFM_MCP_OAUTH=1` to protect the HTTP transport with OAuth 2.1, following the
[MCP authorization spec](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization).
The server acts as its own authorization server. It publishes protected-resource and
authorization-server metadata, supports dynamic client registration, uses authorization code + PKCE (S256),
rotates refresh tokens, and supports revocation. When a client is authorized, the operator approves it on a
consent page protected by `RFM_MCP_OAUTH_PASSWORD`.

```bash
export RFM_MCP_OAUTH=1
export RFM_MCP_PUBLIC_URL=https://rfm.example.com           # public HTTPS origin (behind TLS proxy)
export RFM_MCP_OAUTH_PASSWORD='a-long-random-passphrase'    # >= 12 chars, keep secret
export RFM_MCP_OAUTH_STATE_FILE=/var/lib/rfm-mcp/oauth.json  # optional: survive restarts
python -m mcp_server --transport streamable-http --host 127.0.0.1 --port 8000
```

In Sesame, add a custom app with the URL `https://rfm.example.com/mcp` and pick OAuth. Sesame discovers
the endpoints, registers itself, and opens the consent page. Enter the password there and click **Approve**.
If the client needs a pre-registered client ID/secret instead of dynamic registration, set
`RFM_MCP_OAUTH_CLIENT_ID`, `RFM_MCP_OAUTH_CLIENT_SECRET` and `RFM_MCP_OAUTH_REDIRECT_URIS`
(comma-separated, use the callback URL Sesame shows) and enter the same values in Sesame.

| Variable | Default | Purpose |
|----------|---------|---------|
| `RFM_REPO_ROOT` | this repo | Repository the file/test tools operate on |
| `RFM_MCP_TEST_TIMEOUT` / `RFM_MCP_PYTHON` | `600` / current interpreter | Test-run timeout and Python used for pytest |
| `RFM_MCP_TRANSPORT` / `RFM_MCP_HOST` / `RFM_MCP_PORT` | `stdio` / `127.0.0.1` / `8000` | Transport settings |
| `RFM_MCP_OAUTH` | off | Enable OAuth on HTTP transports (refused with stdio) |
| `RFM_MCP_PUBLIC_URL` | required with OAuth | Externally reachable origin; issuer and resource URL (`<url>/mcp`) |
| `RFM_MCP_OAUTH_PASSWORD` | required with OAuth | Operator password for the consent page |
| `RFM_MCP_OAUTH_STATE_FILE` | in memory | JSON file (mode 0600) persisting clients and tokens |
| `RFM_MCP_OAUTH_CLIENT_ID` / `_CLIENT_SECRET` / `_REDIRECT_URIS` | unset | Optional pre-registered client |
| `GEMINI_API_KEY` / `GEMINI_API_KEY_FILE` | unset | Enables `ask_gemini` / `list_gemini_models` |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Default Gemini model |
| `RFM_MCP_ACCESS_TOKEN_TTL` / `RFM_MCP_REFRESH_TOKEN_TTL` | `3600` / `2592000` | Token lifetimes (seconds) |

Requests whose `Host` is neither the public host nor localhost are rejected (DNS-rebinding protection),
so your proxy must forward the original `Host` header or connect to `127.0.0.1`. Terminate TLS in front
of the server. Never expose plain HTTP publicly. `GET /healthz` is an unauthenticated liveness probe.
Without OAuth the HTTP transport has no authentication, so keep it bound to localhost.

## Repository layout

- `rfai/`: core package modules (AMIFS, DFE, RSA, nodes, graph, core, simulation, CLI/demo)
- `rfc/`: Resonant Fractal Cognition — scale space, resonant field, the scale-invariant operator, constraints, symbol lattice, metacognition, benchmarks
- `docs/RFC_ARCHITECTURE.md`: RFC's design, provenance, measured results, and limitations
- `mcp_server/`: MCP server (repo query, engine tools, test runs, OAuth for remote clients)
- `tests/`: pytest suite validating AMIFS, DFE, graph/nodes/RSA logic, fractal core, scenarios, and the RFC substrate/engine/integration

