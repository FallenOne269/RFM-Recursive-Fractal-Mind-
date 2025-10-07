# Recursive Fractal Mind (RFAI)

Recursive Fractal Mind (RFAI) is a modular orchestration framework that models
fractal cognition, collaborative agent swarms, probabilistic reasoning, and
meta-learning adaptation. The system ships as a production-ready Python package
with a REST API, persistence layer, and deployment manifests.

## Architecture Overview

The platform is composed of four subsystems that are dynamically loaded through
a plugin registry:

- **Fractal Engine (`src/fractal_engine/`)** – multi-scale signal processing
  that produces hierarchical representations.
- **Swarm Coordinator (`src/swarm_coordinator/`)** – simulates collaborative
  agent behaviour and resource allocation.
- **Quantum Processor (`src/quantum_processor/`)** – optional probabilistic
  solver that augments fractal outputs.
- **Meta Learner (`src/meta_learner/`)** – adapts learning rates and system
  behaviour based on performance feedback.

`src/rfai_system.py` exposes the orchestrator that runs a recursive cycle and
aggregates subsystem results. Components register themselves via
`PluginRegistry` which allows swapping implementations without modifying the
orchestrator.

Additional modules include:

- **FastAPI application (`src/api/app.py`)** providing `/health`, `/status`, and
  `/process_task` endpoints secured by an API key.
- **Persistence layer (`src/persistence/state_manager.py`)** for state
  versioning, checksums, and recovery.
- **Configuration utilities (`src/utils/`)** with JSON schema validation and
  path sanitisation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python install.py --dev --verify
```

The installer pins dependencies from `requirements.txt`, optionally installs
developer tooling, creates a user configuration at `config/my_config.json`, and
can run a verification cycle.

## Usage

```python
from rfai_system import RecursiveFractalMind

orchestrator = RecursiveFractalMind()
result = orchestrator.run_cycle({
    "id": "example-001",
    "type": "pattern_recognition",
    "complexity": 0.4,
    "payload": [0.1, 0.2, 0.3, 0.4],
    "metadata": {"source": "docs"},
})
print(result["performance_score"])
```

Run the REST API locally:

```bash
export RFAI_API_KEY=local-key
uvicorn rfai_system.api.app:create_app --factory --reload
```

## Testing

```bash
pytest
```

The test suite covers unit tests for each subsystem, persistence guarantees, an
integration scenario spanning three recursion cycles, and lightweight
performance benchmarks.

## Deployment

- **Docker**: `docker build -t rfai-system .`
- **Compose**: `docker-compose up --build`
- **Kubernetes**: manifests in the `k8s/` directory (deployment, service,
  ingress).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on branching, testing, and
code quality checks.

## License

Distributed under the terms of the MIT License. See [LICENSE](LICENSE).
