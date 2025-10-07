#!/usr/bin/env python3
"""Installation helper for the modular RFAI system."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default_config.json"


def check_python_version() -> None:
    """Ensure a supported Python version is in use."""
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or higher is required")
    print(f"✓ Python {sys.version.split()[0]} detected")


def install_dependencies(dev: bool = False) -> None:
    """Install runtime dependencies (and optional development extras)."""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    if dev:
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "mypy"], check=True)
    print("✓ Dependencies installed")


def verify_installation(config_path: Path | None = None) -> None:
    """Import the orchestrator and execute a smoke test."""
    print("Verifying installation...")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from rfai_system import Orchestrator

    config_file = config_path or DEFAULT_CONFIG
    orchestrator = Orchestrator(config_path=str(config_file))

    import numpy as np

    task = {
        "id": "install_test",
        "type": "verification",
        "complexity": 0.4,
        "data": np.random.randn(orchestrator.system_config.get("base_dimensions", 16)),
        "priority": 0.5,
    }
    result = orchestrator.process_task(task)
    if "performance_score" not in result:
        raise RuntimeError("Orchestrator did not return a performance score")
    print("✓ Verification successful")


def run_tests(pytest_args: list[str] | None = None) -> None:
    """Run the pytest suite for regression coverage."""
    args = [sys.executable, "-m", "pytest"]
    if pytest_args:
        args.extend(pytest_args)
    print("Running tests...")
    subprocess.run(args, check=True)


def create_example_config(destination: Path) -> None:
    """Generate a user-editable configuration example."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    template: Dict[str, Any] = {
        "system": {
            "name": "My RFAI Deployment",
            "max_fractal_depth": 3,
            "base_dimensions": 48,
            "swarm_size": 8,
            "quantum_enabled": False,
            "recursion_limit": 3,
        },
        "modules": {
            "fractal_engine": {"plugin": "fractal_engine", "settings": {"max_depth": 3, "base_dimensions": 48}},
            "swarm_coordinator": {"plugin": "swarm_coordinator", "settings": {"swarm_size": 8}},
            "quantum_processor": {"plugin": "quantum_processor", "enabled": False, "settings": {"qubits": 4}},
            "meta_learner": {
                "plugin": "meta_learner",
                "settings": {"base_learning_rate": 0.01, "adaptation_factor": 1.05, "performance_threshold": 0.7},
            },
        },
        "persistence": {"state_path": "state/my_rfai_state.json", "autosave": False},
    }
    destination.write_text(json.dumps(template, indent=2))
    print(f"✓ Example configuration written to {destination}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Install the Recursive Fractal Autonomous Intelligence system")
    parser.add_argument("--noninteractive", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--verify", action="store_true", help="Verify installation after dependency setup")
    parser.add_argument("--dev", action="store_true", help="Install development tooling")
    parser.add_argument("--config", type=Path, default=None, help="Custom configuration file for verification")
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to pytest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the installer."""
    args = parse_args(argv)
    check_python_version()
    install_dependencies(dev=args.dev)

    if args.verify:
        verify_installation(config_path=args.config)

    example_config_path = CONFIG_DIR / "my_config.json"
    create_example_config(example_config_path)

    should_run_tests = args.verify or args.dev
    if not args.noninteractive and not should_run_tests:
        response = input("Run test suite? (y/n): ").strip().lower()
        should_run_tests = response in {"y", "yes"}

    if should_run_tests:
        run_tests(pytest_args=args.pytest_args)

    print("\nInstallation complete. Explore examples/basic_usage.py to get started.")


if __name__ == "__main__":
    main()
