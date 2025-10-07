# Contributing to Recursive Fractal Mind

Thank you for your interest in contributing! This document outlines the
preferred workflow for improving the Recursive Fractal Mind project.

## Development Workflow

1. Fork the repository and create a feature branch: `git checkout -b feat/my-change`.
2. Install dependencies and tooling: `python install.py --dev --noninteractive`.
3. Run the formatter and linters: `pre-commit run --all-files`.
4. Execute the full test suite: `pytest`.
5. Ensure documentation is updated for new features or behaviour changes.
6. Submit a pull request describing the change and referencing related issues.

## Code Quality

- Follow [PEP 8](https://peps.python.org/pep-0008/) and prefer expressive, typed
  interfaces.
- Keep functions small and focused; document public methods with docstrings.
- Avoid global state—prefer dependency injection and explicit configuration.
- Validate all external inputs and use safe subprocess invocations.

## Commit Messages

- Use the imperative mood, e.g., `Add FastAPI endpoints`.
- Reference issues when applicable, e.g., `Fix persistence checksum (#42)`.
- Squash commits as needed to keep history readable.

## Pull Request Checklist

- [ ] Tests pass locally (`pytest`).
- [ ] Linting succeeds (`black --check`, `flake8`, `mypy`).
- [ ] Documentation updated (README, examples, or docstrings).
- [ ] New dependencies are pinned in `requirements*.txt`.

We appreciate your contributions in advancing the Recursive Fractal Mind
platform!
