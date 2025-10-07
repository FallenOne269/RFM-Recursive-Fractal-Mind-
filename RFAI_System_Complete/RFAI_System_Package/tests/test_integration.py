"""Integration tests that exercise cross-module functionality."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rfai_system import RecursiveFractalAutonomousIntelligence, TaskProcessingError  # noqa: E402
from contracts import TaskSpecification  # noqa: E402


def build_tasks(count: int, dims: int) -> list[TaskSpecification]:
    tasks = []
    for idx in range(count):
        payload = {
            'id': f'int_{idx:03d}',
            'type': 'integration',
            'complexity': 0.3 + 0.05 * idx,
            'data': np.random.randn(dims),
            'priority': 0.6,
        }
        tasks.append(TaskSpecification.from_payload(payload, expected_size=dims))
    return tasks


def test_end_to_end_pipeline_with_contracts():
    system = RecursiveFractalAutonomousIntelligence(
        max_fractal_depth=3,
        base_dimensions=16,
        swarm_size=5,
        quantum_enabled=True,
    )

    contract = system.get_contract('task_ingestion')
    assert contract.name == 'Task Ingestion'
    assert '1.1.0' in contract.compatible_core_versions

    tasks = build_tasks(3, dims=16)
    report = system.process_task_report(tasks[0])

    assert report.performance_score >= 0
    assert 'fractal_output' in report.components
    assert report.fallback_reason is None


def test_benchmark_and_parameter_sweep():
    system = RecursiveFractalAutonomousIntelligence(
        max_fractal_depth=3,
        base_dimensions=16,
        swarm_size=6,
        quantum_enabled=True,
    )
    tasks = build_tasks(4, dims=16)

    benchmark = system.benchmark_against_baseline(tasks)
    assert benchmark.current.mean_score >= 0
    assert 'mean_score_delta' in benchmark.deltas

    sweep = system.run_parameter_sweep(
        parameter_grid={'quantum_enabled': [True, False], 'swarm_size': [4, 6]},
        tasks=tasks,
    )
    assert len(sweep.entries) == 4
    assert sweep.synergy_candidates or sweep.degeneracy_risks


def test_snapshot_and_fallback_behaviour():
    system = RecursiveFractalAutonomousIntelligence(
        max_fractal_depth=2,
        base_dimensions=8,
        swarm_size=4,
        quantum_enabled=False,
    )

    snapshot_id = system.create_snapshot('pre_task')
    tasks = build_tasks(2, dims=8)
    result = system.process_task(tasks[0].to_payload())

    assert result['fallback_reason'] == 'quantum_disabled'
    assert np.allclose(result['component_results']['fractal_output'], result['component_results']['quantum_output'])

    system.rollback_to_snapshot(snapshot_id)
    assert not system.system_state['performance_history']

    with pytest.raises(TaskProcessingError):
        system.rollback_to_snapshot('unknown-snapshot')
