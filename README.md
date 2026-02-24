# Recursive Fractal AI (RFAI)

This repository implements a lightweight end-to-end reference for **Recursive Fractal AI (RFAI)**. The system combines Adaptive Multiscale Iterated Function Systems (AMIFS), Dynamic Fractal Encoding (DFE), Recursive Structural Adaptation (RSA), graph-based fractal nodes, semantic goals, and a recursive core engine. Everything is deterministic, typed, and runs on Python 3.11+ with only `numpy` as a runtime dependency (optionally `cupy` if available for GPU acceleration).

## Architecture overview

### AMIFS
Adaptive Multiscale Iterated Function Systems generate fractal attractors using affine transforms. Parameters adapt to features and context using a cached adaptive component and history-aware contextual component. Precision is configurable for stability.

### Dynamic Fractal Encoding (DFE)
DFE converts arbitrary data into **Fractal Information Motifs (FIMs)** that pair a semantic vector with an AMIFS-generated pattern signature. A registry tracks FIM lifecycles and supports parallel encoding.

### Recursive Structural Adaptation (RSA)
RSA evaluates node performance versus targets and semantic goals to emit `AdaptationSignal`s. Signals adjust node complexity, trigger bifurcation (splitting) when complexity or error rises, and consolidation when stable. Safety caps enforce maximum graph depth and node counts with hysteresis to prevent oscillation.

### Memory layer & smart nodes
Each node keeps a bounded `MemoryLayer` of adaptation history. `SmartFractalNode` manages complexity updates, bifurcation, and consolidation heuristics. `SemanticSmartNode` augments adaptation using semantic goal similarity. `FractalGraph` maintains parent/child relations and propagates signals deterministically.

### Recursive core
`RecursiveFractalAlgorithm` provides a generic recursive processing loop with self-improvement hooks that adjust thresholds based on feedback. `FractalState` wraps state dictionaries with safe copying.

### Simulations
Three deterministic simulation environments model different challenges:
- **Scenario A (TaskHierarchyEnv):** ordered subtasks with dependencies.
- **Scenario B (ObstacleAvoidanceEnv):** moving obstacles on a grid with oscillation detection.
- **Scenario C (ResourceAllocationEnv):** allocating resources across compartments for balance.

`scenarios.py` orchestrates each scenario using the full stack, and `metrics.py` computes task completion, efficiency, recovery, coordination, oscillation, recursion depth, node counts, and bifurcations/consolidations.

## Usage

Install in editable mode:

```bash
pip install -e .
```

Run the demo to execute all scenarios and print a summary table:

```bash
python -m rfai.demo
```

Use the CLI for fine-grained control (scenario selection, steps, seeds, and RSA limits):

```bash
python -m rfai.cli run --scenario A --steps 200 --seed 0
python -m rfai.cli run --scenario all --steps 100 --max-depth 4 --max-nodes 20
```

Run tests:

```bash
pytest
```

## Design choices

- **Determinism first:** all randomness uses `numpy.random.Generator` with explicit seeds, and adaptation caches reuse computed deltas.
- **Safety caps:** RSA enforces `max_depth` and `max_nodes` to avoid unbounded growth, with hysteresis-based split/merge gating.
- **Lightweight patterns:** FIM metadata stores compact signatures (centroid and spread) instead of full point clouds to keep memory low.
- **CPU/GPU agnostic:** AMIFS automatically uses `cupy` if available; otherwise falls back to `numpy` while keeping interfaces identical.

## Repository layout

- `rfai/`: core package modules (AMIFS, DFE, RSA, nodes, graph, core, simulation, CLI/demo)
- `tests/`: pytest suite validating AMIFS, DFE, graph/nodes/RSA logic, fractal core, and scenarios

