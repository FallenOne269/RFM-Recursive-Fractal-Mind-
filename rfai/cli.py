from __future__ import annotations

import argparse
from typing import Dict
import numpy as np

from .sim import scenarios


def _run_single(name: str, steps: int, seed: int, max_depth: int, max_nodes: int) -> Dict:
    if name == "A":
        return scenarios.run_scenario_A(steps=steps, seed=seed, max_depth=max_depth, max_nodes=max_nodes)
    if name == "B":
        return scenarios.run_scenario_B(steps=steps, seed=seed, max_depth=max_depth, max_nodes=max_nodes)
    if name == "C":
        return scenarios.run_scenario_C(steps=steps, seed=seed, max_depth=max_depth, max_nodes=max_nodes)
    raise ValueError(f"Unknown scenario {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="RFAI scenario runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run scenarios")
    run_p.add_argument("--scenario", choices=["A", "B", "C", "all"], default="all")
    run_p.add_argument("--steps", type=int, default=50)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--max-depth", type=int, default=3)
    run_p.add_argument("--max-nodes", type=int, default=10)
    args = parser.parse_args(argv)

    if args.scenario == "all":
        names = ["A", "B", "C"]
    else:
        names = [args.scenario]
    for name in names:
        res = _run_single(name, args.steps, args.seed, args.max_depth, args.max_nodes)
        print(f"Scenario {name}: {res}")


if __name__ == "__main__":
    main()

