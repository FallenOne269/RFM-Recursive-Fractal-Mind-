#!/usr/bin/env python3
"""
RFAI System Installation Script
==============================

Automated installation and setup for the Recursive Fractal Autonomous Intelligence system.
"""

import os
import sys
import subprocess
import json
import argparse

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def verify_installation():
    """Verify that the installation works"""
    print("Verifying installation...")

    try:
        # Test basic import
        sys.path.insert(0, 'src')
        from rfai_system import RecursiveFractalAutonomousIntelligence

        # Test basic initialization
        rfai = RecursiveFractalAutonomousIntelligence(
            max_fractal_depth=2,  # Small for quick test
            base_dimensions=16,
            swarm_size=4,
            quantum_enabled=False  # Faster for testing
        )

        print("✓ RFAI system can be imported and initialized")

        # Test basic processing
        import numpy as np
        test_task = {
            'id': 'install_test',
            'type': 'test',
            'complexity': 0.5,
            'data': np.random.randn(16),
            'priority': 0.8
        }

        result = rfai.process_task(test_task)

        if 'performance_score' in result:
            print("✓ RFAI system processing verification successful")
            return True
        else:
            print("❌ RFAI system processing failed")
            return False

    except Exception as e:
        print(f"❌ Installation verification failed: {str(e)}")
        return False

def run_tests():
    """Run the test suite"""
    print("Running test suite...")

    try:
        result = subprocess.run([sys.executable, "tests/test_rfai.py"], 
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print("Test output:", result.stdout)
            print("Test errors:", result.stderr)
            return False

    except Exception as e:
        print(f"❌ Failed to run tests: {str(e)}")
        return False

def create_example_config():
    """Create an example configuration file"""
    example_config = {
        "system_name": "My_RFAI_System",
        "max_fractal_depth": 4,
        "base_dimensions": 64,
        "swarm_size": 12,
        "quantum_enabled": True,
        "learning_settings": {
            "base_learning_rate": 0.001,
            "adaptation_factor": 1.1,
            "performance_threshold": 0.95
        },
        "note": "Customize these settings for your specific use case"
    }

    with open("config/my_config.json", 'w') as f:
        json.dump(example_config, f, indent=2)

    print("✓ Example configuration created: config/my_config.json")

def main():
    parser = argparse.ArgumentParser(description="Install and verify the RFAI system")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run tests without prompting",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test prompt in non-interactive environments",
    )
    args = parser.parse_args()

    if args.run_tests and args.skip_tests:
        print("❌ Choose either --run-tests or --skip-tests, not both")
        sys.exit(2)

    print("RFAI System Installation")
    print("=" * 30)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Verify installation
    if not verify_installation():
        print("\nInstallation verification failed, but the system may still work.")
        print("Try running the examples manually to test functionality.")

    # Create example config
    create_example_config()

    # Run tests (optional)
    should_run_tests = args.run_tests
    if not args.run_tests and not args.skip_tests:
        try:
            run_tests_input = input("\nRun test suite? (y/n): ").strip().lower()
            should_run_tests = run_tests_input in ['y', 'yes']
        except EOFError:
            print("\nℹ️ No interactive input available, skipping tests prompt")

    if should_run_tests:
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
