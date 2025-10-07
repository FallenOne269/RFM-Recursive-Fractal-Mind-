#!/usr/bin/env python3
"""RFAI System Installation Script."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def check_python_version() -> bool:
    """Check if Python version is compatible."""

    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    print(
        "✓ Python "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected"
    )
    return True


def install_dependencies() -> bool:
    """Install required Python packages."""

    print("Installing dependencies...")
    requirements_file = PACKAGE_ROOT / "requirements.txt"

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def verify_installation() -> bool:
    """Verify that the installation works."""

    print("Verifying installation...")

    try:
        rfai_module = import_module("src.rfai_system")
        RecursiveFractalAutonomousIntelligence = getattr(
            rfai_module, "RecursiveFractalAutonomousIntelligence"
        )

        rfai = RecursiveFractalAutonomousIntelligence(
            max_fractal_depth=2,
            base_dimensions=16,
            swarm_size=4,
            quantum_enabled=False,
        )

        print("✓ RFAI system can be imported and initialized")

        import numpy as np

        test_task = {
            "id": "install_test",
            "type": "test",
            "complexity": 0.5,
            "data": np.random.randn(16),
            "priority": 0.8,
        }

        result = rfai.process_task(test_task)

        if "performance_score" in result:
            print("✓ RFAI system processing verification successful")
            return True

        print("❌ RFAI system processing failed")
        return False

    except Exception as exc:  # noqa: BLE001 - surface installation issues to the user
        print(f"❌ Installation verification failed: {exc}")
        return False


def run_tests() -> bool:
    """Run the test suite."""

    print("Running test suite...")

    try:
        result = subprocess.run(
            [sys.executable, str(PACKAGE_ROOT / "tests" / "test_rfai.py")],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print("✓ All tests passed")
            return True

        print("❌ Some tests failed")
        print("Test output:", result.stdout)
        print("Test errors:", result.stderr)
        return False

    except Exception as exc:  # noqa: BLE001 - best effort diagnostic
        print(f"❌ Failed to run tests: {exc}")
        return False


def create_example_config() -> None:
    """Create an example configuration file."""

    example_config = {
        "system_name": "My_RFAI_System",
        "max_fractal_depth": 4,
        "base_dimensions": 64,
        "swarm_size": 12,
        "quantum_enabled": True,
        "learning_settings": {
            "base_learning_rate": 0.001,
            "adaptation_factor": 1.1,
            "performance_threshold": 0.95,
        },
        "note": "Customize these settings for your specific use case",
    }

    config_dir = PACKAGE_ROOT / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    with (config_dir / "my_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(example_config, config_file, indent=2)

    print("✓ Example configuration created: config/my_config.json")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for automation friendly execution."""

    parser = argparse.ArgumentParser(description="Install the RFAI system")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the full test suite without prompting",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running the test suite without prompting",
    )
    return parser.parse_args()


def should_run_tests(args: argparse.Namespace) -> bool:
    """Determine whether the test suite should be executed."""

    if args.run_tests:
        return True
    if args.skip_tests:
        return False

    env_choice = os.environ.get("RFAI_RUN_TESTS")
    if env_choice is not None:
        return env_choice.strip().lower() in {"y", "yes", "true", "1"}

    run_tests_input = input("\nRun test suite? (y/n): ").strip().lower()
    return run_tests_input in {"y", "yes"}


def main() -> None:
    print("RFAI System Installation")
    print("=" * 30)

    args = parse_arguments()

    if not check_python_version():
        sys.exit(1)

    if not install_dependencies():
        sys.exit(1)

    if not verify_installation():
        print("\nInstallation verification failed, but the system may still work.")
        print("Try running the examples manually to test functionality.")

    create_example_config()

    if should_run_tests(args):
        run_tests()

    print("\n" + "=" * 50)
    print("🎉 RFAI SYSTEM INSTALLATION COMPLETE!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run basic example: python examples/basic_usage.py")
    print("2. Run advanced example: python examples/advanced_usage.py")
    print("3. Customize config/my_config.json for your needs")
    print("4. Import RFAI system in your own projects")
    print("\nFor help, see README.md or check the examples/")


if __name__ == "__main__":
    main()
