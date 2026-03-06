from __future__ import annotations

from .sim import scenarios


def run_demo():
    results = {
        "A": scenarios.run_scenario_A(),
        "B": scenarios.run_scenario_B(),
        "C": scenarios.run_scenario_C(),
    }
    print("RFAI Demo Results")
    print("Scenario | Metrics")
    for name, metrics in results.items():
        print(f"{name}\t| {metrics}")


if __name__ == "__main__":
    run_demo()

