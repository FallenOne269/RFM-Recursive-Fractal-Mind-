"""Recursive Fractal Autonomous Intelligence (RFAI) System."""

import json
import logging
import math
import os
import random
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from .contracts import (
        BenchmarkMetrics,
        BenchmarkReport,
        CONTRACT_REGISTRY,
        ParameterSweepEntry,
        ParameterSweepReport,
        PluginCompatibility,
        PluginMetadata,
        PluginRegistry,
        ProcessingReport,
        SubsystemContract,
        TaskSpecification,
    )
except ImportError:  # pragma: no cover - fallback for direct module execution
    from contracts import (  # type: ignore
        BenchmarkMetrics,
        BenchmarkReport,
        CONTRACT_REGISTRY,
        ParameterSweepEntry,
        ParameterSweepReport,
        PluginCompatibility,
        PluginMetadata,
        PluginRegistry,
        ProcessingReport,
        SubsystemContract,
        TaskSpecification,
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FractalModule:
    """Self-similar processing module with recursive structure."""
    level: int
    dimensions: int
    sub_modules: List['FractalModule']
    weights: np.ndarray
    activation_state: np.ndarray
    learning_rate: float

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.random.randn(self.dimensions, self.dimensions) * 0.1
        if self.activation_state is None:
            self.activation_state = np.zeros(self.dimensions)

@dataclass
class AutonomousAgent:
    """Individual agent in the swarm intelligence system."""
    agent_id: str
    specialization: str
    capabilities: List[str]
    knowledge_base: Dict[str, Any]
    performance_metrics: Dict[str, float]

class TaskProcessingError(RuntimeError):
    """Raised when a task cannot be processed by the system."""


class RecursiveFractalAutonomousIntelligence:
    """Main RFAI system implementing the integrated architecture."""

    def __init__(self, 
                 max_fractal_depth: int = 5,
                 base_dimensions: int = 128,
                 swarm_size: int = 50,
                 quantum_enabled: bool = True,
                 config_path: Optional[str] = None):

        logger.info("Initializing Recursive Fractal Autonomous Intelligence System...")

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                max_fractal_depth = config.get('max_fractal_depth', max_fractal_depth)
                base_dimensions = config.get('base_dimensions', base_dimensions)
                swarm_size = config.get('swarm_size', swarm_size)
                quantum_enabled = config.get('quantum_enabled', quantum_enabled)

        self.max_depth = max_fractal_depth
        self.base_dims = base_dimensions
        self.swarm_size = swarm_size
        self.quantum_enabled = quantum_enabled
        self.core_version = "1.1.0"

        self.contracts: Dict[str, SubsystemContract] = CONTRACT_REGISTRY
        self.plugin_registry = PluginRegistry(core_version=self.core_version)
        self._register_core_plugins()

        logger.info("Building fractal hierarchy...")
        self.fractal_hierarchy = self._build_fractal_hierarchy()

        logger.info("Initializing agent swarm...")
        self.agent_swarm = self._initialize_swarm()

        logger.info("Setting up meta-learning system...")
        self.meta_optimizer = self._initialize_meta_optimizer()

        self.quantum_processor = None
        if quantum_enabled:
            logger.info("Initializing quantum-classical hybrid processor...")
            self.quantum_processor = self._initialize_quantum_processor()

        self.system_state: Dict[str, Any] = {
            'performance_history': [],
            'adaptation_events': [],
            'emergent_behaviors': [],
            'optimization_trajectory': [],
            'fallback_events': [],
        }

        self.architecture_search = self._initialize_architecture_search()
        self._snapshots: Dict[str, Dict[str, Any]] = {}

        logger.info("RFAI System initialized successfully!")
        logger.info(f"- Fractal depth: {self.max_depth} levels")
        logger.info(f"- Base dimensions: {self.base_dims}")
        logger.info(f"- Swarm size: {len(self.agent_swarm)} agents")
        logger.info(f"- Quantum processing: {self.quantum_enabled}")

    def _register_core_plugins(self) -> None:
        """Register built-in subsystem implementations for compatibility tracking."""
        plugins = [
            PluginMetadata(
                name='fractal_processor',
                version='1.1.0',
                compatibility=PluginCompatibility(core_versions=[self.core_version]),
                description='Default recursive fractal processor',
                capabilities=['fractal_processing'],
            ),
            PluginMetadata(
                name='swarm_coordinator',
                version='1.1.0',
                compatibility=PluginCompatibility(core_versions=[self.core_version]),
                description='Default agent swarm coordination',
                capabilities=['swarm_coordination'],
            ),
            PluginMetadata(
                name='meta_learning_core',
                version='1.1.0',
                compatibility=PluginCompatibility(core_versions=[self.core_version]),
                description='Meta-learning strategy for self-improvement',
                capabilities=['meta_learning'],
            ),
            PluginMetadata(
                name='quantum_hybrid',
                version='1.1.0',
                compatibility=PluginCompatibility(core_versions=[self.core_version]),
                description='Quantum-classical hybrid simulation',
                capabilities=['quantum_processing'],
            ),
        ]
        for plugin in plugins:
            try:
                self.plugin_registry.register(plugin)
            except ValueError as exc:
                logger.warning("Plugin registration skipped: %s", exc)

    def _build_fractal_hierarchy(self) -> Dict[int, List[FractalModule]]:
        """Build the fractal hierarchy with self-similar modules."""
        hierarchy = {}

        for level in range(self.max_depth):
            modules_at_level = []
            num_modules = max(1, 2 ** (self.max_depth - level - 1))
            dims = max(4, self.base_dims // (2 ** level))

            for i in range(num_modules):
                sub_modules = []
                if level < self.max_depth - 1:
                    sub_count = max(1, min(4, 2 ** (self.max_depth - level - 2)))
                    sub_dims = max(4, dims // 2)
                    for j in range(sub_count):
                        sub_module = FractalModule(
                            level=level + 1,
                            dimensions=sub_dims,
                            sub_modules=[],
                            weights=np.random.randn(sub_dims, sub_dims) * 0.1,
                            activation_state=np.zeros(sub_dims),
                            learning_rate=0.001 * (2 ** level)
                        )
                        sub_modules.append(sub_module)

                module = FractalModule(
                    level=level,
                    dimensions=dims,
                    sub_modules=sub_modules,
                    weights=np.random.randn(dims, dims) * 0.1,
                    activation_state=np.zeros(dims),
                    learning_rate=0.001 * (2 ** level)
                )
                modules_at_level.append(module)

            hierarchy[level] = modules_at_level

        return hierarchy

    def _initialize_swarm(self) -> List[AutonomousAgent]:
        """Initialize the autonomous agent swarm."""
        agents = []
        specializations = [
            'pattern_recognition', 'optimization', 'memory_management',
            'goal_planning', 'resource_allocation', 'conflict_resolution',
            'learning_coordination', 'emergent_behavior_detection'
        ]

        for i in range(self.swarm_size):
            spec = specializations[i % len(specializations)]

            agent = AutonomousAgent(
                agent_id=f"agent_{i:03d}",
                specialization=spec,
                capabilities=self._generate_capabilities(spec),
                knowledge_base={
                    'experience_buffer': [],
                    'learned_patterns': {},
                    'optimization_history': []
                },
                performance_metrics={
                    'task_completion_rate': np.random.uniform(0.3, 0.7),
                    'learning_efficiency': np.random.uniform(0.3, 0.7),
                    'collaboration_score': np.random.uniform(0.3, 0.7),
                    'adaptability_index': np.random.uniform(0.3, 0.7)
                }
            )
            agents.append(agent)

        return agents

    def _generate_capabilities(self, specialization: str) -> List[str]:
        """Generate capabilities based on agent specialization."""
        capability_map = {
            'pattern_recognition': ['fractal_pattern_analysis', 'anomaly_detection', 'similarity_matching'],
            'optimization': ['gradient_optimization', 'evolutionary_search', 'quantum_optimization'],
            'memory_management': ['hierarchical_storage', 'pattern_compression', 'retrieval_optimization'],
            'goal_planning': ['hierarchical_planning', 'resource_estimation', 'constraint_satisfaction'],
            'resource_allocation': ['load_balancing', 'priority_scheduling', 'capacity_planning'],
            'conflict_resolution': ['consensus_building', 'negotiation', 'arbitration'],
            'learning_coordination': ['meta_learning', 'knowledge_transfer', 'curriculum_design'],
            'emergent_behavior_detection': ['behavior_monitoring', 'pattern_emergence', 'system_analysis']
        }
        return capability_map.get(specialization, ['general_processing'])

    def _initialize_meta_optimizer(self) -> Dict[str, Any]:
        """Initialize meta-learning optimization system."""
        return {
            'learning_rate_adaptation': {
                'base_rate': 0.001,
                'adaptation_factor': 1.1,
                'performance_threshold': 0.95
            },
            'architecture_evolution': {
                'mutation_rate': 0.1,
                'crossover_rate': 0.3,
                'selection_pressure': 0.8
            },
            'task_curriculum': {
                'difficulty_progression': 'adaptive',
                'mastery_threshold': 0.9,
                'exploration_bonus': 0.1
            }
        }

    def _initialize_quantum_processor(self) -> Dict[str, Any]:
        """Initialize quantum-classical hybrid processing."""
        return {
            'qubit_allocation': max(4, self.base_dims // 4),
            'entanglement_patterns': self._generate_entanglement_patterns(),
            'measurement_strategies': ['computational_basis', 'bell_basis', 'arbitrary_rotation'],
            'error_correction': 'surface_code',
            'classical_interface': 'gradient_based_optimization'
        }

    def _generate_entanglement_patterns(self) -> List[List[int]]:
        """Generate entanglement patterns for quantum processing."""
        patterns = []
        qubit_count = max(4, self.base_dims // 4)

        for scale in range(1, min(4, int(math.log2(qubit_count)) + 1)):
            ring_size = min(qubit_count, 2 ** scale)
            for start in range(0, qubit_count - ring_size + 1, max(1, ring_size)):
                pattern = list(range(start, min(start + ring_size, qubit_count)))
                if len(pattern) > 1:
                    patterns.append(pattern)

        return patterns if patterns else [[0, 1]]

    def _initialize_architecture_search(self) -> Dict[str, Any]:
        """Initialize neural architecture search for self-modification."""
        return {
            'search_space': {
                'module_types': ['fractal_conv', 'attention', 'recursive_rnn', 'quantum_gate'],
                'connection_patterns': ['feed_forward', 'recurrent', 'skip', 'fractal_bypass'],
                'optimization_targets': ['accuracy', 'efficiency', 'adaptability', 'robustness']
            },
            'evolution_strategy': {
                'population_size': 20,
                'generations': 100,
                'selection_method': 'tournament',
                'mutation_operators': ['add_module', 'remove_module', 'modify_connection', 'adjust_parameters']
            },
            'evaluation_metrics': {
                'performance': 0.4,
                'complexity': 0.2,
                'novelty': 0.2,
                'stability': 0.2
            }
        }

    def fractal_processing(self, input_data: np.ndarray, level: int = 0) -> np.ndarray:
        """Recursive fractal processing with self-similar patterns."""
        if level >= self.max_depth:
            return input_data

        modules = self.fractal_hierarchy[level]
        expected_size = modules[0].dimensions if modules else self.base_dims

        # Ensure proper input sizing
        if input_data.size != expected_size:
            if input_data.size > expected_size:
                input_data = input_data[:expected_size]
            else:
                input_data = np.pad(input_data, (0, expected_size - input_data.size))

        level_output = np.zeros(expected_size)

        for i, module in enumerate(modules):
            module_input = input_data.copy()
            if module_input.size != module.dimensions:
                if module_input.size > module.dimensions:
                    module_input = module_input[:module.dimensions]
                else:
                    module_input = np.pad(module_input, (0, module.dimensions - module_input.size))

            # Apply module transformation
            try:
                transformed = np.tanh(np.dot(module.weights, module_input.reshape(-1, 1)).flatten())
            except ValueError:
                transformed = np.tanh(module_input * np.diag(module.weights))

            # Recursive processing through sub-modules
            if module.sub_modules and level < self.max_depth - 1:
                sub_output = self.fractal_processing(transformed, level + 1)
                min_size = min(len(transformed), len(sub_output))
                combined = np.zeros(max(len(transformed), len(sub_output)))
                combined[:min_size] = 0.7 * transformed[:min_size] + 0.3 * sub_output[:min_size]
                if len(transformed) > min_size:
                    combined[min_size:min_size+len(transformed)-min_size] = transformed[min_size:]
                elif len(sub_output) > min_size:
                    combined[min_size:min_size+len(sub_output)-min_size] = sub_output[min_size:]
                transformed = combined[:module.dimensions]

            # Update module state
            module.activation_state = 0.9 * module.activation_state + 0.1 * transformed

            # Accumulate output
            output_size = min(len(level_output), len(transformed))
            level_output[:output_size] += transformed[:output_size] / len(modules)

        return level_output

    def swarm_coordination(self, task: TaskSpecification) -> Dict[str, Any]:
        """Coordinate swarm agents for collaborative problem solving."""
        if not self.agent_swarm:
            raise TaskProcessingError("Agent swarm is empty; cannot coordinate task")

        selection_size = min(5, len(self.agent_swarm))
        participating_agents = np.random.choice(self.agent_swarm, size=selection_size, replace=False)

        results = {}
        for agent in participating_agents:
            base_performance = agent.performance_metrics['task_completion_rate']
            task_complexity = task.complexity

            success_rate = max(0, min(1, base_performance - task_complexity * 0.2 + np.random.normal(0, 0.1)))

            results[agent.agent_id] = {
                'agent_id': agent.agent_id,
                'specialization': agent.specialization,
                'success_rate': success_rate,
                'quality_score': success_rate * np.random.uniform(0.8, 1.2),
                'execution_time': task_complexity * np.random.uniform(0.5, 1.5)
            }

        success_rates = [r['success_rate'] for r in results.values()]

        return {
            'task_id': task.task_id,
            'overall_success_rate': np.mean(success_rates),
            'participating_agents': list(results.keys()),
            'individual_results': results,
            'swarm_efficiency': len(results) / len(self.agent_swarm)
        }

    def _simulate_quantum_computation(self, data: np.ndarray) -> np.ndarray:
        """Simulate quantum computation."""
        if not self.quantum_enabled:
            return data

        qubit_count = self.quantum_processor['qubit_allocation']

        if data.size > qubit_count:
            quantum_input = data[:qubit_count]
        else:
            quantum_input = np.pad(data, (0, qubit_count - data.size))

        norm = np.linalg.norm(quantum_input)
        if norm > 0:
            normalized_data = quantum_input / norm
        else:
            normalized_data = quantum_input

        quantum_state = normalized_data.copy().astype(complex)
        for i in range(len(quantum_state)):
            quantum_state[i] = quantum_state[i] / np.sqrt(2)
            quantum_state[i] *= np.exp(1j * np.random.uniform(0, 2 * np.pi))

        probabilities = np.abs(quantum_state) ** 2
        measured_result = np.real(quantum_state) * probabilities

        if data.size != len(measured_result):
            if data.size > len(measured_result):
                result = np.pad(measured_result, (0, data.size - len(measured_result)))
            else:
                result = measured_result[:data.size]
        else:
            result = measured_result

        return result

    def quantum_classical_hybrid_processing(self, data: np.ndarray) -> np.ndarray:
        """Quantum-classical hybrid processing."""
        if not self.quantum_enabled:
            return self.fractal_processing(data)

        quantum_result = self._simulate_quantum_computation(data)
        classical_result = self.fractal_processing(quantum_result)

        return classical_result

    def meta_learning_optimization(self) -> Dict[str, Any]:
        """Meta-learning system for continuous self-improvement."""
        return {
            'architecture_changes': [],
            'parameter_updates': ['learning_rate_adjustment'],
            'performance_improvement': np.random.uniform(-0.05, 0.1),
            'new_capabilities': []
        }

    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        scores = []

        if 'fractal_output' in results:
            fractal_score = min(1.0, np.mean(np.abs(results['fractal_output'])))
            scores.append(fractal_score)

        if 'swarm_output' in results:
            swarm_score = results['swarm_output'].get('overall_success_rate', 0.5)
            scores.append(swarm_score)

        if 'quantum_output' in results:
            quantum_score = min(1.0, np.mean(np.abs(results['quantum_output'])))
            scores.append(quantum_score)

        return np.mean(scores) if scores else 0.5

    def process_task(self, task: Union[TaskSpecification, Dict[str, Any]]) -> Dict[str, Any]:
        """Main task processing pipeline.

        Parameters
        ----------
        task:
            A :class:`TaskSpecification` instance or a dictionary payload adhering to the
            ``task_ingestion`` contract.

        Returns
        -------
        Dict[str, Any]
            Serialized :class:`ProcessingReport` for compatibility with existing clients.

        Raises
        ------
        TaskProcessingError
            If validation fails or a subsystem raises an unrecoverable exception.
        """

        task_spec = self._ensure_task_spec(task)
        logger.info("Processing task: %s", task_spec.task_id)
        start_time = datetime.now()

        try:
            input_data = task_spec.normalised_data(self.base_dims)
        except ValueError as exc:
            raise TaskProcessingError(str(exc)) from exc

        results: Dict[str, Any] = {}

        fractal_result = self.fractal_processing(input_data)
        results['fractal_output'] = fractal_result

        fallback_reason: Optional[str] = None
        if self.quantum_enabled:
            quantum_result = self.quantum_classical_hybrid_processing(input_data)
            results['quantum_output'] = quantum_result
        else:
            results['quantum_output'] = fractal_result.copy()
            fallback_reason = 'quantum_disabled'
            self.system_state['fallback_events'].append({
                'task_id': task_spec.task_id,
                'timestamp': start_time.isoformat(),
                'reason': fallback_reason,
            })

        swarm_result = self.swarm_coordination(task_spec)
        results['swarm_output'] = swarm_result

        meta_result = self.meta_learning_optimization()
        results['meta_optimization'] = meta_result

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        performance_score = self._calculate_performance_score(results)
        self.system_state['performance_history'].append(performance_score)

        report = ProcessingReport(
            task_id=task_spec.task_id,
            performance_score=performance_score,
            processing_time=processing_time,
            components=results,
            fallback_reason=fallback_reason,
            timestamp=end_time.isoformat(),
        )

        logger.info("Task completed - Performance: %.3f", performance_score)
        return report.to_dict()

    def process_task_report(self, task: Union[TaskSpecification, Dict[str, Any]]) -> ProcessingReport:
        """Process a task and return a structured :class:`ProcessingReport`."""

        result = self.process_task(task)
        return ProcessingReport(
            task_id=result['task_id'],
            performance_score=result['performance_score'],
            processing_time=result['processing_time'],
            components=result['component_results'],
            fallback_reason=result['fallback_reason'],
            timestamp=result['timestamp'],
        )

    def _ensure_task_spec(self, task: Union[TaskSpecification, Dict[str, Any]]) -> TaskSpecification:
        if isinstance(task, TaskSpecification):
            return task
        if not isinstance(task, dict):
            raise TaskProcessingError("Task must be a TaskSpecification or dict")
        try:
            return TaskSpecification.from_payload(task, expected_size=self.base_dims)
        except ValueError as exc:
            raise TaskProcessingError(str(exc)) from exc

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        total_params = sum(sum(module.weights.size for module in modules) 
                          for modules in self.fractal_hierarchy.values())

        active_agents = len([agent for agent in self.agent_swarm 
                           if agent.performance_metrics['task_completion_rate'] > 0.5])

        return {
            'system_id': 'RFAI_v1.0',
            'status': 'OPERATIONAL',
            'fractal_hierarchy': {
                'levels': self.max_depth,
                'total_modules': sum(len(modules) for modules in self.fractal_hierarchy.values()),
                'total_parameters': total_params
            },
            'agent_swarm': {
                'total_agents': len(self.agent_swarm),
                'active_agents': active_agents,
                'specializations': list(set(agent.specialization for agent in self.agent_swarm))
            },
            'quantum_processor': {
                'enabled': self.quantum_enabled,
                'qubits': self.quantum_processor['qubit_allocation'] if self.quantum_enabled else 0
            },
            'performance': {
                'tasks_processed': len(self.system_state['performance_history']),
                'avg_performance': np.mean(self.system_state['performance_history']) if self.system_state['performance_history'] else 0.0,
                'learning_trend': np.polyfit(range(len(self.system_state['performance_history'])),
                                           self.system_state['performance_history'], 1)[0] if len(self.system_state['performance_history']) > 1 else 0.0
            },
            'plugins': {
                name: {'version': meta.version, 'capabilities': meta.capabilities}
                for name, meta in self.plugin_registry.list_plugins().items()
            },
            'contracts': {name: contract.version for name, contract in self.contracts.items()}
        }

    def save_state(self, filepath: str) -> None:
        """Save system state to file."""
        state = {
            'config': {
                'max_depth': self.max_depth,
                'base_dims': self.base_dims,
                'swarm_size': self.swarm_size,
                'quantum_enabled': self.quantum_enabled
            },
            'system_state': self.system_state,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"System state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load system state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.system_state = state.get('system_state', self.system_state)
        logger.info(f"System state loaded from {filepath}")

    def create_snapshot(self, label: Optional[str] = None) -> str:
        """Create an in-memory snapshot of the system state for rollback."""

        snapshot_id = f"{label or 'snapshot'}-{uuid.uuid4().hex[:8]}"
        agent_metrics = [deepcopy(agent.performance_metrics) for agent in self.agent_swarm]
        self._snapshots[snapshot_id] = {
            'system_state': deepcopy(self.system_state),
            'agent_metrics': agent_metrics,
            'quantum_enabled': self.quantum_enabled,
        }
        logger.info("Snapshot created: %s", snapshot_id)
        return snapshot_id

    def rollback_to_snapshot(self, snapshot_id: str) -> None:
        """Rollback to a previously captured snapshot."""

        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            raise TaskProcessingError(f"Snapshot '{snapshot_id}' not found")

        self.system_state = deepcopy(snapshot['system_state'])
        for agent, metrics in zip(self.agent_swarm, snapshot['agent_metrics']):
            agent.performance_metrics = deepcopy(metrics)
        self.quantum_enabled = snapshot['quantum_enabled']
        if self.quantum_enabled and self.quantum_processor is None:
            self.quantum_processor = self._initialize_quantum_processor()
        logger.info("Rolled back to snapshot: %s", snapshot_id)

    def get_contract(self, name: str) -> SubsystemContract:
        """Retrieve the declared contract for a subsystem."""

        if name not in self.contracts:
            raise KeyError(f"Unknown subsystem contract '{name}'")
        return self.contracts[name]

    def benchmark_against_baseline(
        self,
        tasks: Sequence[Union[TaskSpecification, Dict[str, Any]]],
        baseline_overrides: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkReport:
        """Benchmark the current system against a baseline configuration."""

        normalized_tasks = [self._ensure_task_spec(task) for task in tasks]
        baseline_config = {
            'max_fractal_depth': self.max_depth,
            'base_dimensions': self.base_dims,
            'swarm_size': self.swarm_size,
            'quantum_enabled': False,
        }
        if baseline_overrides:
            baseline_config.update(baseline_overrides)

        baseline_system = RecursiveFractalAutonomousIntelligence(**baseline_config)

        baseline_metrics = self._run_benchmark_suite(baseline_system, normalized_tasks)
        current_metrics = self._run_benchmark_suite(self, normalized_tasks)
        deltas = {
            'mean_score_delta': current_metrics.mean_score - baseline_metrics.mean_score,
            'mean_latency_delta': current_metrics.mean_latency - baseline_metrics.mean_latency,
            'throughput_delta': current_metrics.throughput - baseline_metrics.throughput,
        }
        return BenchmarkReport(
            baseline=baseline_metrics,
            current=current_metrics,
            deltas=deltas,
        )

    def _run_benchmark_suite(
        self,
        system: 'RecursiveFractalAutonomousIntelligence',
        tasks: Iterable[TaskSpecification],
    ) -> BenchmarkMetrics:
        start = time.perf_counter()
        latencies: List[float] = []
        scores: List[float] = []

        for task in tasks:
            report = system.process_task(task.to_payload())
            latencies.append(report['processing_time'])
            scores.append(report['performance_score'])

        total_time = max(time.perf_counter() - start, 1e-9)
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0
        throughput = len(scores) / total_time
        return BenchmarkMetrics(mean_score=mean_score, mean_latency=mean_latency, throughput=throughput)

    def run_parameter_sweep(
        self,
        parameter_grid: Dict[str, Sequence[Any]],
        tasks: Sequence[Union[TaskSpecification, Dict[str, Any]]],
    ) -> ParameterSweepReport:
        """Execute a parameter sweep and log synergies or degeneracies."""

        normalized_tasks = [self._ensure_task_spec(task) for task in tasks]
        combinations = self._expand_parameter_grid(parameter_grid)
        entries: List[ParameterSweepEntry] = []

        for params in combinations:
            config = {
                'max_fractal_depth': params.get('max_fractal_depth', self.max_depth),
                'base_dimensions': params.get('base_dimensions', self.base_dims),
                'swarm_size': params.get('swarm_size', self.swarm_size),
                'quantum_enabled': params.get('quantum_enabled', self.quantum_enabled),
            }
            candidate_system = RecursiveFractalAutonomousIntelligence(**config)
            metrics = candidate_system._run_benchmark_suite(candidate_system, normalized_tasks)
            note = 'quantum_enabled' in params and not params['quantum_enabled']
            entries.append(
                ParameterSweepEntry(
                    parameters=params,
                    metrics=metrics,
                    notes='Quantum bypass' if note else '',
                )
            )

        scores = np.array([entry.metrics.mean_score for entry in entries])
        if scores.size == 0:
            synergy_candidates: List[Dict[str, Any]] = []
            degeneracy_risks: List[Dict[str, Any]] = []
        else:
            threshold_high = float(np.percentile(scores, 75))
            threshold_low = float(np.percentile(scores, 25))
            synergy_candidates = [
                {**entry.parameters, 'mean_score': entry.metrics.mean_score}
                for entry in entries
                if entry.metrics.mean_score >= threshold_high
            ]
            degeneracy_risks = [
                {**entry.parameters, 'mean_score': entry.metrics.mean_score}
                for entry in entries
                if entry.metrics.mean_score <= threshold_low
            ]

        for candidate in synergy_candidates:
            logger.info("Parameter synergy detected: %s", candidate)
        for risk in degeneracy_risks:
            logger.warning("Parameter degeneracy detected: %s", risk)

        return ParameterSweepReport(entries=entries, synergy_candidates=synergy_candidates, degeneracy_risks=degeneracy_risks)

    def _expand_parameter_grid(self, parameter_grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
        keys = list(parameter_grid.keys())
        if not keys:
            return [{}]

        combinations: List[Dict[str, Any]] = [{}]
        for key in keys:
            new_combinations: List[Dict[str, Any]] = []
            for value in parameter_grid[key]:
                for combo in combinations:
                    updated = dict(combo)
                    updated[key] = value
                    new_combinations.append(updated)
            combinations = new_combinations
        return combinations

if __name__ == "__main__":
    # Example usage
    rfai = RecursiveFractalAutonomousIntelligence(
        max_fractal_depth=4,
        base_dimensions=64,
        swarm_size=12,
        quantum_enabled=True
    )

    # Process a sample task
    sample_task = {
        'id': 'test_001',
        'type': 'pattern_recognition',
        'complexity': 0.7,
        'data': np.random.randn(64),
        'requirements': ['accuracy', 'speed'],
        'priority': 0.8
    }

    result = rfai.process_task(sample_task)
    print(f"Task processed - Performance: {result['performance_score']:.3f}")
    print(f"System status: {rfai.get_system_status()}")
