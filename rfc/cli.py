"""Command line interface for Resonant Fractal Cognition.

    python -m rfc.cli demo
    python -m rfc.cli bench --seed 0
    python -m rfc.cli episode --seed 3 --steps 32
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional, Sequence

from .constraints import forbidden_labels
from .engine import RFCConfig, ResonantFractalCognition
from .operator import OperatorParams
from .tasks import format_report, make_composite, run_benchmark


def _episode(args: argparse.Namespace) -> int:
    dataset = make_composite(args.seed, trials=args.episodes)
    invariants = []
    if args.forbid:
        invariants.append(forbidden_labels("cli_forbidden", args.forbid))
    config = RFCConfig(
        dim=dataset.dim,
        steps=args.steps,
        max_depth=args.max_depth,
        seed=args.seed,
        reflect_every=args.reflect_every,
        params=OperatorParams(),
    )
    mind = ResonantFractalCognition(config, codebook=dataset.codebook, invariants=invariants)
    hits = 0
    for evidence, truth in dataset.samples:
        percept = mind.perceive(evidence)
        hits += percept.label == truth
        mind.learn(1.0 if percept.label == truth else 0.0)
        if args.verbose:
            print(f"truth={truth} answer={percept.label} confidence={percept.confidence:.3f} "
                  f"coherence={percept.coherence:.3f} depth={percept.depth_used}")
    print(f"accuracy {hits}/{len(dataset.samples)} = {hits / len(dataset.samples):.3f}")
    print(mind.explain(percept))
    if args.json:
        print(json.dumps(mind.state(), indent=2, default=str))
    return 0


def _bench(args: argparse.Namespace) -> int:
    results = run_benchmark(args.seed)
    if args.json:
        print(json.dumps(
            [{"task": r.task, "scores": r.scores, "detail": r.detail, "notes": r.notes} for r in results],
            indent=2,
            default=str,
        ))
    else:
        print(format_report(results))
    return 0


def _demo(_args: argparse.Namespace) -> int:
    from .demo import main as demo_main

    demo_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rfc", description="Resonant Fractal Cognition")
    sub = parser.add_subparsers(dest="command", required=True)

    episode = sub.add_parser("episode", help="run episodes over the composite task")
    episode.add_argument("--seed", type=int, default=0)
    episode.add_argument("--episodes", type=int, default=20)
    episode.add_argument("--steps", type=int, default=24)
    episode.add_argument("--max-depth", type=int, default=2)
    episode.add_argument("--reflect-every", type=int, default=8)
    episode.add_argument("--forbid", action="append", default=[], help="label the system may not answer")
    episode.add_argument("--verbose", action="store_true")
    episode.add_argument("--json", action="store_true")
    episode.set_defaults(func=_episode)

    bench = sub.add_parser("bench", help="run the benchmark suite")
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=_bench)

    demo = sub.add_parser("demo", help="narrated walkthrough plus benchmarks")
    demo.set_defaults(func=_demo)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
