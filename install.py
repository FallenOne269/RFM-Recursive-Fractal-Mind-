"""Installer utility for the RFAI system."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    """Run a subprocess command with error propagation."""

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def install_requirements(dev: bool = False) -> None:
    requirements = Path("requirements.txt")
    if not requirements.exists():
        raise SystemExit("requirements.txt not found")

    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    if dev:
        command.append("--upgrade")
    run_command(command)


def verify_installation() -> None:
    run_command([sys.executable, "-m", "pytest", "-q"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install and verify the RFAI system")
    parser.add_argument("--verify", action="store_true", help="Run the test suite after installation")
    parser.add_argument(
        "--noninteractive",
        action="store_true",
        help="Skip interactive prompts (reserved for future use)",
    )
    parser.add_argument("--dev", action="store_true", help="Install developer dependencies")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_requirements(dev=args.dev)
    if args.verify:
        verify_installation()


if __name__ == "__main__":
    main()
