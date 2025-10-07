"""Installation utility for the Recursive Fractal Mind system."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfai_system import RecursiveFractalMind
from utils import load_config, sanitize_path

BASE_DIR = Path(__file__).resolve().parent


def run_pip_install(requirements_file: Path) -> None:
    subprocess.run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_file),
    ], check=True)


def install_dependencies(dev: bool) -> None:
    run_pip_install(BASE_DIR / "requirements.txt")
    if dev:
        run_pip_install(BASE_DIR / "requirements-dev.txt")


def create_user_config() -> Path:
    target = sanitize_path(BASE_DIR / "config" / "my_config.json", base_dir=BASE_DIR)
    if not target.exists():
        default_config = load_config()
        serialized = {
            "fractal_engine": {
                "max_depth": default_config.fractal.max_depth,
                "base_dimensions": default_config.fractal.base_dimensions,
                "noise_scale": default_config.fractal.noise_scale,
            },
            "swarm_coordinator": {
                "swarm_size": default_config.swarm.swarm_size,
                "max_parallel_tasks": default_config.swarm.max_parallel_tasks,
            },
            "quantum_processor": {
                "enabled": default_config.quantum.enabled,
                "qubit_count": default_config.quantum.qubit_count,
                "sampling_runs": default_config.quantum.sampling_runs,
            },
            "meta_learner": {
                "base_learning_rate": default_config.meta.base_learning_rate,
                "adaptation_factor": default_config.meta.adaptation_factor,
                "performance_threshold": default_config.meta.performance_threshold,
            },
            "persistence": {
                "state_dir": str(default_config.persistence.state_dir.relative_to(BASE_DIR)),
                "version": default_config.persistence.version,
            },
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    return target


def verify_installation() -> None:
    orchestrator = RecursiveFractalMind()
    result = orchestrator.run_cycle({
        "id": "install-check",
        "type": "diagnostic",
        "complexity": 0.3,
        "payload": [0.1, 0.2, 0.3],
        "metadata": {"source": "installer"},
    })
    print("Verification successful. Performance score:", result["performance_score"])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the Recursive Fractal Mind system")
    parser.add_argument("--verify", action="store_true", help="Verify installation after setup")
    parser.add_argument("--dev", action="store_true", help="Install developer tooling")
    parser.add_argument("--noninteractive", action="store_true", help="Run without prompts")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    install_dependencies(dev=args.dev)
    config_path = create_user_config()
    print(f"Configuration available at {config_path}")
    if args.verify:
        verify_installation()
    elif not args.noninteractive:
        answer = input("Run verification now? [y/N]: ").strip().lower()
        if answer == "y":
            verify_installation()


if __name__ == "__main__":
    main()
