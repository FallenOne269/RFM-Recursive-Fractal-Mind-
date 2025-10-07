# Recursive Fractal Autonomous Intelligence (RFAI)

The Recursive Fractal Autonomous Intelligence system is a modular orchestration
framework that combines fractal processing, swarm coordination, quantum
simulation, and adaptive meta-learning. The project has been refactored to
prioritise production-ready concerns: strong validation, persistence fidelity,
clear subsystem boundaries, and deployment artefacts.

## Key Features

- **Fractal Engine** – builds recursive representations with tunable depth and
  branching.
- **Swarm Coordinator** – manages specialised agents, tracks collaboration, and
  adapts experience buffers.
- **Quantum Processor** – optional quantum-classical hybrid simulation with
  entropy tracking.
- **Meta Learner** – monitors performance trends and adjusts learning rates.
- **Orchestrator** – dynamically loads plugins, validates configuration, and
  coordinates full processing loops.
- **FastAPI Service** – exposes `/health`, `/status`, and `/process_task`
  endpoints with a stable response schema.

## Project Layout

```
RFAI_System_Package/
├── config/                  # Default and example configuration files
├── examples/                # Basic and advanced usage demonstrations
├── install.py               # Installer with verification and test hooks
├── requirements.txt         # Runtime and tooling dependencies
├── src/
│   └── rfai_system/
│       ├── api/             # FastAPI application
│       ├── core/            # Orchestrator, plugin registry, base classes
│       ├── fractal_engine/  # Fractal engine implementation
│       ├── meta_learner/    # Meta-learning subsystem
│       ├── quantum_processor/
│       ├── swarm_coordinator/
│       └── utils/           # Validation helpers
└── tests/                   # Pytest suite with unit, integration, and API tests
```

## Installation

It is recommended to use a Python 3.10+ virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
cd RFAI_System_Package
python install.py --verify --noninteractive
```

The installer accepts the following flags:

- `--noninteractive` – skips prompts.
- `--verify` – runs orchestrator smoke tests and executes the pytest suite.
- `--dev` – installs development tooling (mypy, pre-commit) and executes tests.
- `--config <path>` – specify a custom configuration file for verification.
- `--pytest-args ...` – extra arguments forwarded to pytest when tests run.

An editable example configuration is generated at `config/my_config.json`.

## Quick Start

```python
from pathlib import Path
from rfai_system import Orchestrator

config_path = Path("config/default_config.json")
orchestrator = Orchestrator(config_path=str(config_path))

result = orchestrator.process_task(
    {
        "id": "demo",
        "type": "pattern_recognition",
        "complexity": 0.6,
        "data": [0.1, 0.2, 0.3],
    }
)
print(result["performance_score"])
```

Refer to `examples/basic_usage.py` and `examples/advanced_usage.py` for complete
pipelines, including persistence and benchmarking flows.

## Running Tests

All tests use pytest:

```bash
pytest
```

The suite covers validation edge cases, subsystem behaviour, state persistence,
FastAPI endpoints, and an end-to-end regression loop.

## FastAPI Service

Launch the API locally with uvicorn:

```bash
uvicorn rfai_system.api.server:app --reload
```

Endpoints:

- `GET /health` – readiness probe.
- `GET /status` – orchestrator status with subsystem summaries.
- `POST /process_task` – process a task. Responses always include the keys
  `fractal_output`, `swarm_output`, `quantum_output`, and `meta_output`.

Set the `RFAI_CONFIG_PATH` environment variable to point to a different
configuration file for the API process.

## Containerisation

A production-ready container image can be built with the provided Dockerfile:

```bash
docker build -t rfai-system .
docker run -p 8000:8000 rfai-system
```

Kubernetes manifests are provided under `k8s/` for deployment, service exposure,
ingress with TLS termination, and configuration via environment variables.

## Tooling

- Formatting and linting are handled through pre-commit (`.pre-commit-config.yaml`).
- Static typing via mypy and PEP-8 enforcement via flake8.
- CI should execute `pytest`, `flake8`, and `mypy` to maintain regression
  coverage and code quality.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for branch strategy, review
expectations, and development workflow.

## License

This project is released under the MIT License. See `LICENSE` for details.
