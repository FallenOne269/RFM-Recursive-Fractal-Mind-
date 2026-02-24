from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from ..dfe import DynamicFractalEncoder
from ..fractal_graph import FractalGraph
from ..rsa import RecursiveStructuralAdapter, RSAConfig
from ..semantic_goal import SemanticGoal
from ..semantic_smart_node import SemanticSmartNode
from ..smart_fractal_node import SmartFractalNode
from ..adaptation_signal import AdaptationSignal
from ..sim import metrics
from .envs import TaskHierarchyEnv, ObstacleAvoidanceEnv, ResourceAllocationEnv


def _init_stack(seed: int, max_depth: int, max_nodes: int):
    dfe = DynamicFractalEncoder()
    graph = FractalGraph(max_depth=max_depth)
    rsa = RecursiveStructuralAdapter(dfe, graph, RSAConfig(max_depth=max_depth, max_nodes=max_nodes))
    semantic = np.ones(4)
    fim = dfe.encode({"seed": seed}, semantic, context={})
    node = SemanticSmartNode(node_id="root", fim_id=fim.fim_id, complexity=fim.complexity, goal=SemanticGoal(semantic))
    graph.add_node(node)
    return dfe, graph, rsa


def run_scenario_A(steps: int = 50, seed: int = 0, max_depth: int = 3, max_nodes: int = 10) -> Dict:
    env = TaskHierarchyEnv(seed)
    env.reset()
    dfe, graph, rsa = _init_stack(seed, max_depth, max_nodes)
    signals = 0
    for _ in range(steps):
        obs = env.step()
        observed = {"progress": obs["progress"], "semantic_vector": np.ones(4)}
        target = {"progress": env.num_subtasks}
        sig = rsa.evaluate_and_adapt("root", observed, target, goal=graph.nodes["root"].goal)
        rsa.apply_signal("root", sig)
        signals += 1
        if obs.get("done"):
            break
    metrics_dict = {
        "task_completion_rate": metrics.task_completion_rate(env.progress, env.num_subtasks),
        "time_efficiency": metrics.time_efficiency(max(1, env.step_count), env.progress),
        "coordination_effectiveness": metrics.coordination_effectiveness(signals, len(graph.nodes)),
        "node_count": len(graph.nodes),
        "bifurcations": max(0, len(graph.nodes) - 1),
    }
    return metrics_dict


def run_scenario_B(steps: int = 50, seed: int = 0, max_depth: int = 3, max_nodes: int = 10) -> Dict:
    env = ObstacleAvoidanceEnv(rng_seed=seed)
    env.reset()
    dfe, graph, rsa = _init_stack(seed, max_depth, max_nodes)
    collisions = []
    for _ in range(steps):
        obs = env.step()
        collisions.append(obs["hit"])
        observed = {"semantic_vector": np.ones(4) * (1 - int(obs["hit"]))}
        target = {"avoid": 1.0}
        sig = AdaptationSignal(delta=-0.2 if obs["hit"] else 0.05, reason="collision" if obs["hit"] else "progress")
        rsa.apply_signal("root", sig)
    metrics_dict = {
        "oscillation_score": metrics.oscillation_score(collisions),
        "node_count": len(graph.nodes),
        "bifurcations": max(0, len(graph.nodes) - 1),
    }
    return metrics_dict


def run_scenario_C(steps: int = 50, seed: int = 0, max_depth: int = 3, max_nodes: int = 10) -> Dict:
    env = ResourceAllocationEnv(rng_seed=seed)
    env.reset()
    dfe, graph, rsa = _init_stack(seed, max_depth, max_nodes)
    efficiencies = []
    for _ in range(steps):
        obs = env.step()
        efficiencies.append(obs["efficiency"])
        observed = {"efficiency": obs["efficiency"], "semantic_vector": np.ones(4)}
        target = {"efficiency": 0.8}
        sig = rsa.evaluate_and_adapt("root", observed, target, goal=graph.nodes["root"].goal)
        rsa.apply_signal("root", sig)
    metrics_dict = {
        "time_efficiency": float(np.mean(efficiencies)),
        "node_count": len(graph.nodes),
        "bifurcations": max(0, len(graph.nodes) - 1),
    }
    return metrics_dict

