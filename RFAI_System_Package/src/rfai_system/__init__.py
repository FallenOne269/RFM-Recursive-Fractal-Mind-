"""Recursive Fractal Autonomous Intelligence (RFAI) system package."""

from .core.orchestrator import Orchestrator

# Backwards compatibility alias
RecursiveFractalAutonomousIntelligence = Orchestrator

__all__ = [
    "Orchestrator",
    "RecursiveFractalAutonomousIntelligence",
]
