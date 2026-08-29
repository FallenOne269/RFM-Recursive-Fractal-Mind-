"""Resonant Fractal Cognition (RFC).

A cognitive architecture in which belief is an interference pattern rather than
a stored value.  One scale-invariant operator drives every level of the system:
perception, symbol formation, and the system's reflection on itself are the same
computation applied to different evidence at different scales.

See ``docs/RFC_ARCHITECTURE.md`` for the design and its provenance in the
research collected in this repository.
"""

from .constraints import ConstraintField, Invariant, forbidden_direction, forbidden_labels
from .engine import Percept, RFCConfig, ResonantFractalCognition
from .field import Coalition, Hypothesis, ResonantField
from .lattice import LatticeConfig, Symbol, SymbolLattice
from .metacognition import MetaConfig, MetaPolicy, MetaReport, MetaResonator
from .operator import OperatorParams, ScaleInvariantOperator, StepReport, TUNABLE_PARAMETERS
from .scale_space import ScaleBand, dyadic_decompose, temporal_bands
from .telemetry import EpisodeRecord, Telemetry

__version__ = "0.1.0"

__all__ = [
    "Coalition",
    "ConstraintField",
    "EpisodeRecord",
    "Hypothesis",
    "Invariant",
    "LatticeConfig",
    "MetaConfig",
    "MetaPolicy",
    "MetaReport",
    "MetaResonator",
    "OperatorParams",
    "Percept",
    "RFCConfig",
    "ResonantField",
    "ResonantFractalCognition",
    "ScaleBand",
    "ScaleInvariantOperator",
    "StepReport",
    "Symbol",
    "SymbolLattice",
    "TUNABLE_PARAMETERS",
    "Telemetry",
    "dyadic_decompose",
    "forbidden_direction",
    "forbidden_labels",
    "temporal_bands",
]
