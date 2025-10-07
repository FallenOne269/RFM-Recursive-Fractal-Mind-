"""Contracts and structured interfaces for the RFAI system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional
import numpy as np


@dataclass(frozen=True)
class IODescription:
    """Describe the shape and semantics of subsystem inputs or outputs."""

    name: str
    description: str
    dtype: str
    shape: Optional[str] = None


@dataclass(frozen=True)
class ErrorBehavior:
    """Capture recoverable and fatal error expectations for a subsystem."""

    recoverable: List[str] = field(default_factory=list)
    fatal: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PerformanceExpectation:
    """Latency/throughput expectations for a subsystem."""

    target_latency_ms: float
    throughput_tasks_per_sec: float
    notes: str = ""


@dataclass(frozen=True)
class SubsystemContract:
    """Formal contract for a subsystem's interface and performance targets."""

    name: str
    version: str
    inputs: List[IODescription]
    outputs: List[IODescription]
    error_behavior: ErrorBehavior
    performance: PerformanceExpectation
    compatible_core_versions: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class PluginCompatibility:
    """Describe compatibility constraints for modular plugins."""

    core_versions: List[str]
    requires: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PluginMetadata:
    """Metadata tracked for each plugin to enforce compatibility."""

    name: str
    version: str
    compatibility: PluginCompatibility
    description: str = ""
    capabilities: List[str] = field(default_factory=list)


class PluginRegistry:
    """Simple in-memory registry to track plugin versions and compatibility."""

    def __init__(self, core_version: str):
        self.core_version = core_version
        self._plugins: Dict[str, PluginMetadata] = {}

    def register(self, metadata: PluginMetadata) -> None:
        if self.core_version not in metadata.compatibility.core_versions:
            raise ValueError(
                f"Plugin {metadata.name} v{metadata.version} is not compatible with core {self.core_version}"
            )
        conflicts = set(metadata.compatibility.conflicts)
        if conflicts.intersection(self._plugins.keys()):
            conflict = conflicts.intersection(self._plugins.keys()).pop()
            raise ValueError(
                f"Plugin {metadata.name} conflicts with already registered plugin {conflict}"
            )
        self._plugins[metadata.name] = metadata

    def unregister(self, name: str) -> None:
        self._plugins.pop(name, None)

    def get(self, name: str) -> Optional[PluginMetadata]:
        return self._plugins.get(name)

    def list_plugins(self) -> Dict[str, PluginMetadata]:
        return dict(self._plugins)


@dataclass
class TaskSpecification:
    """Structured representation of a task to be processed by the system."""

    task_id: str
    task_type: str
    complexity: float
    data: np.ndarray
    requirements: List[str] = field(default_factory=list)
    priority: float = 0.5

    def normalised_data(self, expected_size: int) -> np.ndarray:
        arr = np.asarray(self.data, dtype=float).flatten()
        if not np.isfinite(arr).all():
            raise ValueError("Task data must be finite")
        if arr.size > expected_size:
            arr = arr[:expected_size]
        elif arr.size < expected_size:
            arr = np.pad(arr, (0, expected_size - arr.size))
        return arr

    def to_payload(self) -> Dict[str, Any]:
        return {
            'id': self.task_id,
            'type': self.task_type,
            'complexity': self.complexity,
            'data': self.data.tolist(),
            'requirements': list(self.requirements),
            'priority': self.priority,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], expected_size: Optional[int] = None) -> "TaskSpecification":
        try:
            task_id = str(payload['id'])
            task_type = str(payload.get('type', 'general'))
            complexity = float(payload.get('complexity', 0.5))
            requirements = list(payload.get('requirements', []))
            priority = float(payload.get('priority', 0.5))
        except KeyError as exc:
            raise ValueError(f"Missing required task field: {exc}") from exc

        data = payload.get('data')
        if data is None:
            raise ValueError("Task payload must include 'data'")
        np_data = np.asarray(data, dtype=float).flatten()
        if expected_size is not None:
            if np_data.size > expected_size:
                np_data = np_data[:expected_size]
            elif np_data.size < expected_size:
                np_data = np.pad(np_data, (0, expected_size - np_data.size))
        return cls(
            task_id=task_id,
            task_type=task_type,
            complexity=max(0.0, min(1.0, complexity)),
            data=np_data,
            requirements=requirements,
            priority=max(0.0, min(1.0, priority)),
        )


@dataclass
class ProcessingReport:
    """Detailed output returned after processing a task."""

    task_id: str
    performance_score: float
    processing_time: float
    components: Dict[str, Any]
    fallback_reason: Optional[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'performance_score': float(self.performance_score),
            'processing_time': float(self.processing_time),
            'component_results': self.components,
            'fallback_reason': self.fallback_reason,
            'timestamp': self.timestamp,
        }


@dataclass
class BenchmarkMetrics:
    """Aggregate metrics gathered during benchmarking."""

    mean_score: float
    mean_latency: float
    throughput: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'mean_score': float(self.mean_score),
            'mean_latency': float(self.mean_latency),
            'throughput': float(self.throughput),
        }


@dataclass
class BenchmarkReport:
    """Comparison of baseline and current system performance."""

    baseline: BenchmarkMetrics
    current: BenchmarkMetrics
    deltas: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'baseline': self.baseline.to_dict(),
            'current': self.current.to_dict(),
            'deltas': {k: float(v) for k, v in self.deltas.items()},
        }


@dataclass
class ParameterSweepEntry:
    """Result for an individual parameter combination in a sweep."""

    parameters: Dict[str, Any]
    metrics: BenchmarkMetrics
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameters': self.parameters,
            'metrics': self.metrics.to_dict(),
            'notes': self.notes,
        }


@dataclass
class ParameterSweepReport:
    """Summary of a parameter sweep across configuration combinations."""

    entries: List[ParameterSweepEntry]
    synergy_candidates: List[Dict[str, Any]]
    degeneracy_risks: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entries': [entry.to_dict() for entry in self.entries],
            'synergy_candidates': self.synergy_candidates,
            'degeneracy_risks': self.degeneracy_risks,
        }


CONTRACT_REGISTRY: Dict[str, SubsystemContract] = {
    'task_ingestion': SubsystemContract(
        name='Task Ingestion',
        version='1.1.0',
        inputs=[
            IODescription('payload', 'Task payload with metadata and raw data vector', 'Mapping', '{id:str, data:array}'),
        ],
        outputs=[
            IODescription('task_specification', 'Validated and normalized TaskSpecification instance', 'TaskSpecification'),
        ],
        error_behavior=ErrorBehavior(
            recoverable=['Missing optional fields replaced with defaults'],
            fatal=['Missing required fields', 'Non-finite data values'],
        ),
        performance=PerformanceExpectation(
            target_latency_ms=1.5,
            throughput_tasks_per_sec=1000.0,
            notes='Validation is vectorized to remain sub-millisecond for default dimensions.',
        ),
        compatible_core_versions=['1.1.0'],
        notes='Provides a strict schema boundary for tasks entering the system.',
    ),
    'fractal_processing': SubsystemContract(
        name='Fractal Processing',
        version='1.1.0',
        inputs=[
            IODescription('vector', 'Normalized numeric vector sized to fractal base dimensions', 'np.ndarray', '(base_dimensions,)'),
            IODescription('level', 'Current recursion depth', 'int'),
        ],
        outputs=[
            IODescription('vector', 'Transformed representation after fractal traversal', 'np.ndarray', '(dimensions,)'),
        ],
        error_behavior=ErrorBehavior(
            recoverable=['Dimension mismatch triggers padding or truncation'],
            fatal=['Non-numeric inputs'],
        ),
        performance=PerformanceExpectation(
            target_latency_ms=5.0,
            throughput_tasks_per_sec=150.0,
            notes='Per-level recursion budgeted at <1ms to retain total <5ms for default depth 4.',
        ),
        compatible_core_versions=['1.1.0'],
        notes='Handles recursive traversal and aggregation of fractal modules.',
    ),
    'quantum_processing': SubsystemContract(
        name='Quantum Hybrid Processing',
        version='1.1.0',
        inputs=[
            IODescription('vector', 'Normalized input vector limited by qubit allocation', 'np.ndarray'),
        ],
        outputs=[
            IODescription('vector', 'Quantum-enhanced classical representation', 'np.ndarray'),
        ],
        error_behavior=ErrorBehavior(
            recoverable=['Automatic bypass when quantum features disabled'],
            fatal=['Negative qubit allocation configuration'],
        ),
        performance=PerformanceExpectation(
            target_latency_ms=2.0,
            throughput_tasks_per_sec=250.0,
            notes='Simulation bounded to deterministic operations to maintain tight latency.',
        ),
        compatible_core_versions=['1.1.0'],
        notes='Falls back to classical-only path when quantum disabled.',
    ),
    'swarm_coordination': SubsystemContract(
        name='Swarm Coordination',
        version='1.1.0',
        inputs=[
            IODescription('task', 'Task specification with complexity rating', 'TaskSpecification'),
        ],
        outputs=[
            IODescription('swarm_report', 'Agent-level success metrics', 'Mapping'),
        ],
        error_behavior=ErrorBehavior(
            recoverable=['Insufficient swarm size triggers reduced participant selection'],
            fatal=['Empty swarm after initialization'],
        ),
        performance=PerformanceExpectation(
            target_latency_ms=3.0,
            throughput_tasks_per_sec=200.0,
        ),
        compatible_core_versions=['1.1.0'],
        notes='Aggregates agent metrics with deterministic sampling for reproducibility in tests.',
    ),
    'meta_learning': SubsystemContract(
        name='Meta-Learning',
        version='1.1.0',
        inputs=[
            IODescription('system_state', 'Current adaptive metrics collected during task execution', 'Mapping'),
        ],
        outputs=[
            IODescription('optimization_report', 'Planned adjustments to improve performance', 'Mapping'),
        ],
        error_behavior=ErrorBehavior(
            recoverable=['Negative improvement values indicate exploration, not failure'],
            fatal=[],
        ),
        performance=PerformanceExpectation(
            target_latency_ms=1.0,
            throughput_tasks_per_sec=500.0,
        ),
        compatible_core_versions=['1.1.0'],
        notes='Optimization is lightweight and does not mutate state directly.',
    ),
    'state_management': SubsystemContract(
        name='State Management',
        version='1.1.0',
        inputs=[
            IODescription('snapshot_request', 'Identifier or label describing requested snapshot', 'str'),
        ],
        outputs=[
            IODescription('snapshot_id', 'Opaque identifier for stored snapshot', 'str'),
        ],
        error_behavior=ErrorBehavior(
            recoverable=['Unknown snapshot identifiers are reported and ignored'],
            fatal=['Filesystem write failures when persisting snapshots to disk'],
        ),
        performance=PerformanceExpectation(
            target_latency_ms=0.5,
            throughput_tasks_per_sec=2000.0,
            notes='Snapshots are stored in-memory using lightweight deep copies.',
        ),
        compatible_core_versions=['1.1.0'],
        notes='Supports checkpoint/rollback operations for experimentation.',
    ),
}
