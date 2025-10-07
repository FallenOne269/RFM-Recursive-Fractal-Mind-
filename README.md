# Recursive Fractal Mind (RFAI)

Recursive Fractal Mind (RFAI) is a modular orchestration framework that models
recursive intelligence through coordinated subsystems. The orchestrator exposes
an HTTP API that combines the Fractal Engine, Swarm Coordinator, Quantum
Processor, and Meta Learner to process numeric task payloads.

## Architecture Overview

```
[Client]
   |
   v
[FastAPI Service] -- Auth --> [RFAISystem Orchestrator]
                                   |
                                   +--> Fractal Engine (recursive signal transform)
                                   +--> Swarm Coordinator (agent consensus)
                                   +--> Quantum Processor (probabilistic synthesis)
                                   +--> Meta Learner (adaptive insights)
```

Each subsystem can be swapped or disabled at runtime. Configuration is validated
with Pydantic models to ensure safe defaults.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python install.py --dev
```

The `--dev` flag upgrades dependencies suitable for local development. Use
`python install.py --verify` to run the automated test suite after installing
requirements.

## Usage

### Running the API

```bash
export API_KEY=changeme
uvicorn src.api:app --reload
```

### Processing a Task

```bash
curl -X POST \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{"values": [1, 2, 3, 4]}' \
  http://127.0.0.1:8000/process_task
```

### Status and Health

- `GET /health` – service heartbeat
- `GET /status` – orchestrator status (requires API key)

## Testing

```bash
pytest -v
```

## State Persistence

Use `src.utils.state_manager.save_state` and `load_state` to persist orchestrator
state with checksum validation.

## Docker

```bash
docker build -t rfai-system .
docker run -e API_KEY=changeme -p 8000:8000 rfai-system
```

## Kubernetes

Deploy manifests from `deploy/k8s/` after pushing the container image to your
registry. The deployment includes liveness and readiness probes hitting
`/health`.

## Project Layout

- `src/rfai_system.py` – orchestrator
- `src/fractal_engine/` – fractal processing subsystem
- `src/swarm_coordinator/` – swarm intelligence layer
- `src/quantum_processor/` – quantum-classical processor stub
- `src/meta_learner/` – adaptive meta-learning component
- `src/api.py` – FastAPI surface
- `tests/` – pytest suite covering subsystems, persistence, and API
- `deploy/k8s/` – Kubernetes deployment assets

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on development workflow,
code style, and pull request expectations.
