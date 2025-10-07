"""Swarm coordination subsystem managing specialised agents."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

import numpy as np

from ..core.base import BaseSubsystem
from ..core.registry import plugin_registry


@dataclass
class Agent:
    """Representation of a single autonomous agent within the swarm."""

    identifier: str
    specialization: str
    capabilities: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the agent into a serialisable dictionary."""
        payload = asdict(self)
        payload["memory"] = list(self.memory)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Agent":
        """Recreate an agent from saved state."""
        return cls(
            identifier=payload["identifier"],
            specialization=payload["specialization"],
            capabilities=list(payload.get("capabilities", [])),
            metrics=dict(payload.get("metrics", {})),
            memory=list(payload.get("memory", [])),
        )


class SwarmCoordinator(BaseSubsystem):
    """Coordinate the agent swarm for collaborative task execution."""

    DEFAULT_SPECIALISATIONS = {
        "pattern_recognition": ["fractal_pattern_analysis", "anomaly_detection"],
        "optimization": ["gradient_search", "evolutionary_search"],
        "memory_management": ["hierarchical_storage", "retrieval_optimisation"],
        "goal_planning": ["plan_generation", "constraint_satisfaction"],
        "resource_allocation": ["load_balancing", "capacity_planning"],
        "conflict_resolution": ["negotiation", "consensus_building"],
        "learning_coordination": ["meta_learning", "knowledge_transfer"],
        "emergent_behavior_detection": ["pattern_monitoring", "novelty_detection"],
    }

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.swarm_size: int = int(self.config.get("swarm_size", 12))
        self.specialisations: Dict[str, List[str]] = dict(
            self.DEFAULT_SPECIALISATIONS | self.config.get("specialisations", {})
        )
        seed = self.config.get("seed")
        self.rng = np.random.default_rng(seed)
        self.agents: List[Agent] = self._initialise_agents()
        self.history: List[Dict[str, Any]] = []

    def _initialise_agents(self) -> List[Agent]:
        """Create the configured number of agents."""
        agents: List[Agent] = []
        specialisations = list(self.specialisations.keys())
        for index in range(self.swarm_size):
            spec = specialisations[index % len(specialisations)]
            capabilities = list(self.specialisations[spec])
            metrics = {
                "task_completion_rate": float(self.rng.uniform(0.4, 0.8)),
                "collaboration_score": float(self.rng.uniform(0.4, 0.8)),
                "adaptability": float(self.rng.uniform(0.4, 0.8)),
            }
            agents.append(
                Agent(
                    identifier=f"agent_{index:03d}",
                    specialization=spec,
                    capabilities=capabilities,
                    metrics=metrics,
                )
            )
        return agents

    def _select_agents(self, task_type: str, count: int) -> List[Agent]:
        """Select a subset of agents specialised for the task type."""
        matching = [agent for agent in self.agents if agent.specialization == task_type]
        if len(matching) < count:
            matching.extend(agent for agent in self.agents if agent not in matching)
        return matching[:count]

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents to collaborate on the task."""
        task_type = task.get("type", "general")
        complexity = float(task.get("complexity", 0.5))
        required_agents = max(1, int(round(complexity * 4)))
        selected_agents = self._select_agents(task_type, required_agents)
        collaboration_scores = [
            agent.metrics.get("collaboration_score", 0.5) for agent in selected_agents
        ]
        base_score = float(np.mean(collaboration_scores)) if collaboration_scores else 0.0
        synergy = float(np.clip(base_score * (1.0 + complexity / 2.0), 0.0, 1.0))

        record = {
            "task_id": task.get("id", "unknown"),
            "selected_agents": [agent.identifier for agent in selected_agents],
            "collaboration_score": base_score,
            "synergy": synergy,
        }
        self.history.append(record)

        for agent in selected_agents:
            agent.metrics["collaboration_score"] = float(
                np.clip(
                    agent.metrics.get("collaboration_score", 0.5) * 0.95 + synergy * 0.05, 0.0, 1.0
                )
            )
            agent.memory.append({"task_id": record["task_id"], "outcome": synergy})
            if len(agent.memory) > 50:
                agent.memory.pop(0)

        return {
            "selected_agents": record["selected_agents"],
            "collaboration_score": base_score,
            "synergy": synergy,
            "swarm_success": float(np.clip(synergy * (1.0 + complexity), 0.0, 1.0)),
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialise the swarm state."""
        return {
            "config": {"swarm_size": self.swarm_size},
            "agents": [agent.to_dict() for agent in self.agents],
            "history": list(self.history[-20:]),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agents from saved state."""
        config = state.get("config", {})
        self.swarm_size = int(config.get("swarm_size", self.swarm_size))
        self.agents = [Agent.from_dict(payload) for payload in state.get("agents", [])]
        self.history = list(state.get("history", []))

    def get_status(self) -> Dict[str, Any]:
        """Return current swarm statistics."""
        avg_collaboration = (
            float(np.mean([agent.metrics.get("collaboration_score", 0.0) for agent in self.agents]))
            if self.agents
            else 0.0
        )
        return {
            "swarm_size": len(self.agents),
            "history_records": len(self.history),
            "average_collaboration": avg_collaboration,
        }


def _factory(config: Dict[str, Any]) -> SwarmCoordinator:
    """Factory used for plugin registration."""
    return SwarmCoordinator(config)


plugin_registry.register("swarm_coordinator", _factory)
