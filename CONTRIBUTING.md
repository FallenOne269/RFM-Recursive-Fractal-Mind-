# Contributing to Recursive Fractal Mind

Thank you for your interest in improving the RFAI system! This guide outlines
the workflow used in this repository.

## Branching and Pull Requests

1. Create feature branches from `work` using descriptive names, e.g.
   `feature/add-swarm-simulation`.
2. Keep commits small and focused. Squash commits only when necessary to clean
   up the history before merging.
3. Open a pull request against `work` and fill out the PR template. Describe the
   motivation, implementation, and testing performed.
4. Request at least one review before merging. Address review comments promptly.

## Code Style

- Target Python 3.11+ and follow PEP 8.
- Use type hints everywhere. Run `mypy` locally before submitting a pull request.
- Format code with `black` (line length 88) and lint with `flake8`.
- Avoid global state. Use dependency injection to pass configuration.
- Log actionable information with the `logging` module instead of `print`.

## Testing

- Add unit tests for all new features or bug fixes.
- Run `pytest -v` before submitting a PR.
- Integration changes that affect the API must include FastAPI `TestClient`
  coverage.

## Tooling

Install developer tooling via:

```bash
python install.py --dev --verify
```

This installs dependencies and runs the test suite to verify the environment.

## Release Process

1. Ensure the CI workflow passes on the main branch.
2. Build and tag a Docker image: `docker build -t rfai-system:<tag> .`.
3. Push Kubernetes manifests updates as needed.
4. Create a release note summarizing major changes and deployment steps.
