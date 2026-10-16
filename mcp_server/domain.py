"""RFM-specific MCP tools: describe and run the repository's engines in-process."""

from __future__ import annotations

import ast
import importlib
import dataclasses
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
COMPUTE = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

PACKAGES = ("rfai", "rfai/sim", "rfc", "src")
SCENARIOS = {
    "A": "TaskHierarchyEnv: ordered subtasks with dependencies",
    "B": "ObstacleAvoidanceEnv: moving obstacles on a grid with oscillation detection",
    "C": "ResourceAllocationEnv: allocating resources across compartments for balance",
}
MAX_VALUES = 10_000


def _rfc(module: str) -> ModuleType:
    # Imported dynamically so type-checking mcp_server doesn't pull in rfc's own errors.
    return importlib.import_module(f"rfc.{module}")


def _bounded(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        raise ToolError(f"{name} must be between {low} and {high}")


def _default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and dataclasses into plain JSON types."""
    return json.loads(json.dumps(value, default=_default))


def _module_summaries(package: str) -> dict[str, Optional[str]]:
    summaries: dict[str, Optional[str]] = {}
    for path in sorted((CODE_ROOT / package).glob("*.py")):
        try:
            doc = ast.get_docstring(ast.parse(path.read_text()))
        except SyntaxError:
            doc = None
        summaries[path.stem] = doc.strip().splitlines()[0] if doc else None
    return summaries


def register_domain_tools(mcp: FastMCP, repo_root: Path) -> None:
    @mcp.tool(annotations=READ_ONLY)
    def describe_architecture() -> dict[str, Any]:
        """Describe the RFM codebase: packages and modules, RFAI simulation scenarios,
        RFC default configuration and tunable parameters, and the orchestrator pipeline
        with each component's config schema."""
        RFCConfig = _rfc("engine").RFCConfig
        TUNABLE_PARAMETERS = _rfc("operator").TUNABLE_PARAMETERS
        from src.rfai_system import DEFAULT_COMPONENTS
        from src.utils.validation import CONFIG_MODELS

        return {
            "packages": {pkg: _module_summaries(pkg) for pkg in PACKAGES},
            "rfai_scenarios": SCENARIOS,
            "rfc": {
                "default_config": jsonable(dataclasses.asdict(RFCConfig())),
                "tunable_parameters": jsonable(TUNABLE_PARAMETERS),
            },
            "orchestrator_pipeline": [
                {
                    "component": name,
                    "class": f"{spec.module_path}.{spec.class_name}",
                    "output_key": spec.output_key,
                    "config_schema": CONFIG_MODELS[name].model_json_schema(),
                }
                for name, spec in DEFAULT_COMPONENTS.items()
            ],
        }

    @mcp.tool(annotations=COMPUTE)
    async def run_rfai_scenario(
        scenario: Literal["A", "B", "C", "all"] = "all",
        steps: int = 50,
        seed: int = 0,
        max_depth: int = 3,
        max_nodes: int = 10,
    ) -> dict[str, Any]:
        """Run a deterministic RFAI simulation (A: task hierarchy, B: obstacle
        avoidance, C: resource allocation, or all) through the full AMIFS/DFE/RSA stack
        and return its metrics (completion, efficiency, coordination, node counts)."""
        _bounded("steps", steps, 1, 5000)
        _bounded("max_depth", max_depth, 1, 10)
        _bounded("max_nodes", max_nodes, 1, 500)
        from rfai.sim import scenarios

        names = ["A", "B", "C"] if scenario == "all" else [scenario]

        def run() -> dict[str, Any]:
            return {
                name: getattr(scenarios, f"run_scenario_{name}")(
                    steps=steps, seed=seed, max_depth=max_depth, max_nodes=max_nodes
                )
                for name in names
            }

        results = await anyio.to_thread.run_sync(run)
        return {
            "parameters": {
                "steps": steps,
                "seed": seed,
                "max_depth": max_depth,
                "max_nodes": max_nodes,
            },
            "scenarios": {name: SCENARIOS[name] for name in names},
            "results": jsonable(results),
        }

    @mcp.tool(annotations=COMPUTE)
    async def run_rfc_episode(
        seed: int = 0,
        episodes: int = 20,
        steps: int = 24,
        max_depth: int = 2,
        reflect_every: int = 8,
        forbid: Optional[list[str]] = None,
        include_episodes: bool = False,
        include_state: bool = False,
    ) -> dict[str, Any]:
        """Run Resonant Fractal Cognition over the composite benchmark task (same as
        `python -m rfc.cli episode`): perceive each sample, learn from reward, and
        return accuracy, the final percept and its explanation. `forbid` lists labels
        the system may never answer (enforced as constraint invariants)."""
        _bounded("episodes", episodes, 1, 500)
        _bounded("steps", steps, 1, 200)
        _bounded("max_depth", max_depth, 0, 6)
        _bounded("reflect_every", reflect_every, 1, 1000)
        forbidden_labels = _rfc("constraints").forbidden_labels
        engine = _rfc("engine")
        RFCConfig, ResonantFractalCognition = (
            engine.RFCConfig,
            engine.ResonantFractalCognition,
        )
        OperatorParams = _rfc("operator").OperatorParams
        make_composite = _rfc("tasks").make_composite

        def run() -> dict[str, Any]:
            dataset = make_composite(seed, trials=episodes)
            unknown = sorted(set(forbid or []) - set(dataset.codebook))
            if unknown:
                raise ToolError(
                    f"unknown label(s) {unknown}; valid: {sorted(dataset.codebook)}"
                )
            invariants = [forbidden_labels("mcp_forbidden", forbid)] if forbid else []
            config = RFCConfig(
                dim=dataset.dim,
                steps=steps,
                max_depth=max_depth,
                seed=seed,
                reflect_every=reflect_every,
                params=OperatorParams(),
            )
            mind = ResonantFractalCognition(
                config,
                codebook=dataset.codebook,
                invariants=invariants,
            )
            rows, hits, percept = [], 0, None
            for evidence, truth in dataset.samples:
                percept = mind.perceive(evidence)
                correct = percept.label == truth
                hits += correct
                mind.learn(1.0 if correct else 0.0)
                rows.append(
                    {
                        "truth": truth,
                        "answer": percept.label,
                        "confidence": percept.confidence,
                        "coherence": percept.coherence,
                        "depth_used": percept.depth_used,
                    }
                )
            if percept is None:
                raise ToolError("dataset produced no samples")
            result: dict[str, Any] = {
                "accuracy": hits / len(rows),
                "hits": hits,
                "samples": len(rows),
                "labels": sorted(dataset.codebook),
                "forbidden": forbid or [],
                "last_percept": percept.as_dict(),
                "explanation": mind.explain(percept),
            }
            if include_episodes:
                result["episodes"] = rows
            if include_state:
                result["state"] = mind.state()
            return result

        return jsonable(await anyio.to_thread.run_sync(run))

    @mcp.tool(annotations=COMPUTE)
    async def run_orchestrator_cycle(
        values: list[float], config: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Run one RFAISystem cycle (fractal engine -> swarm coordinator -> quantum
        processor -> meta learner) on a numeric input and return every component output
        plus orchestrator status. `config` maps component name to
        {"enabled": bool, "config": {...}} (see describe_architecture for schemas)."""
        if not values:
            raise ToolError("values must contain at least one number")
        if len(values) > MAX_VALUES:
            raise ToolError(f"values may contain at most {MAX_VALUES} numbers")
        from src.rfai_system import RFAISystem

        def run() -> dict[str, Any]:
            system = RFAISystem(config)
            outputs = system.run_cycle(values)
            return {"outputs": outputs, "status": system.get_status()}

        return jsonable(await anyio.to_thread.run_sync(run))

    def _doc(rel_path: str) -> str:
        path = repo_root / rel_path
        return (
            path.read_text(errors="replace")
            if path.is_file()
            else f"{rel_path} missing"
        )

    @mcp.resource("rfm://readme", mime_type="text/markdown")
    def readme_resource() -> str:
        """Repository README."""
        return _doc("README.md")

    @mcp.resource("rfm://agents", mime_type="text/markdown")
    def agents_resource() -> str:
        """Agent system specification (AGENTS.md)."""
        return _doc("AGENTS.md")

    @mcp.resource("rfm://docs/rfc-architecture", mime_type="text/markdown")
    def rfc_architecture_resource() -> str:
        """Resonant Fractal Cognition design, results and limitations."""
        return _doc("docs/RFC_ARCHITECTURE.md")
