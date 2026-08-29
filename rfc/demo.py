"""Runnable demonstration of Resonant Fractal Cognition.

``python -m rfc.demo`` walks through one episode in detail -- what resonated,
what crystallised, what was refused -- and then runs the benchmark suite.
"""

from __future__ import annotations

import numpy as np

from .constraints import forbidden_labels
from .engine import RFCConfig, ResonantFractalCognition
from .field import normalize_vector
from .tasks import format_report, make_composite, run_benchmark


def walkthrough(seed: int = 0) -> ResonantFractalCognition:
    """One episode, narrated."""

    dataset = make_composite(seed, trials=12)
    forbidden = sorted(dataset.codebook)[-1]
    mind = ResonantFractalCognition(
        RFCConfig(dim=dataset.dim, seed=seed),
        codebook=dataset.codebook,
        invariants=[forbidden_labels("house_rule", [forbidden],
                                     description=f"{forbidden} is off limits")],
    )

    print("=" * 72)
    print("Resonant Fractal Cognition -- one episode")
    print("=" * 72)
    print(f"concepts: {', '.join(sorted(dataset.codebook))}")
    print(f"invariant: '{forbidden}' may never be the answer\n")

    for index, (evidence, truth) in enumerate(dataset.samples[:6], start=1):
        percept = mind.perceive(evidence)
        mark = "ok " if percept.label == truth else "MISS"
        print(f"[{index}] truth={truth} answer={percept.label} ({mark}) "
              f"confidence={percept.confidence:.3f} coherence={percept.coherence:.3f} "
              f"depth={percept.depth_used}")

    print("\nlast episode, in full:")
    print(mind.explain(percept))
    print("\nstate:")
    for key, value in mind.state().items():
        print(f"  {key}: {value}")
    return mind


def decomposition_demo(seed: int = 5) -> None:
    """Two sources in one signal, named one after the other."""

    rng = np.random.default_rng(seed)
    dim = 64
    codebook = {name: normalize_vector(rng.normal(size=dim)) for name in "abcdef"}
    mind = ResonantFractalCognition(RFCConfig(dim=dim, seed=seed), codebook=codebook)
    mixture = codebook["b"] + codebook["e"] + 0.2 * rng.normal(size=dim)
    print("\n" + "=" * 72)
    print("Explaining away: one mixture, two sources")
    print("=" * 72)
    print(f"mixed b + e  ->  decompose: {mind.decompose(mixture, 2)}")


def main() -> None:
    walkthrough()
    decomposition_demo()
    print("\n" + "=" * 72)
    print("Benchmarks (RFC against baselines and against its own ablations)")
    print("=" * 72)
    print(format_report(run_benchmark(0)))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
