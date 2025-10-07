# Contributing to the RFAI System

Thank you for investing in the Recursive Fractal Autonomous Intelligence
project. This document outlines expectations for contributors and reviewers.

## Branch Strategy

Use the following branch prefixes to keep history organised:

- `refactor/core/*` – structural changes to the orchestrator, plugins, or core
  abstractions.
- `feature/api/*` – FastAPI endpoints, request/response models, or HTTP
  behaviour.
- `feature/<area>/*` – new functionality for specific subsystems.
- `bugfix/*` – fixes for reported issues.
- `docs/*` – documentation-only changes.

Always open pull requests against `main` and request review from at least one
maintainer.

## Development Workflow

1. Create a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
2. Install dependencies with development tooling: `python install.py --dev --noninteractive`.
3. Install the pre-commit hooks: `pre-commit install`.
4. Implement your changes with comprehensive unit tests.
5. Ensure the following commands succeed:
   - `pytest`
   - `flake8 src tests`
   - `mypy src`
6. Update documentation and type hints when introducing new public behaviour.
7. Provide a clear, test-backed description in your pull request.

## Code Style

- Follow PEP-8 and favour type annotations for all public functions.
- Use docstrings for every class, function, and module.
- Prefer dependency injection over global state. Subsystems should be
  configurable through explicit parameters.
- Handle filesystem or network interaction with appropriate validation and
  exception handling.

## Testing Expectations

- Unit tests should cover normal, boundary, and error scenarios.
- Include regression tests when fixing bugs.
- Integration tests must exercise orchestration across all subsystems.
- Performance-sensitive code should include benchmarks or runtime assertions to
  detect regressions.

## Security and Secrets

- Do not commit secrets or production configuration files.
- Use environment variables for sensitive values. The provided Kubernetes
  manifests expect this pattern.

## Communication

- Discuss significant design changes via GitHub issues before starting work.
- Document architectural decisions in the README or supplementary ADR files.

We appreciate your contributions and the care you bring to this system.
