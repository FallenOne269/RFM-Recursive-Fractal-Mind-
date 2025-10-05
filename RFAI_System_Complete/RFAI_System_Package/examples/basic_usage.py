"""
Basic RFAI System Usage Example
==============================

This example demonstrates basic usage of the RFAI system.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rfai_system import RecursiveFractalAutonomousIntelligence

def main():
    print("RFAI System - Basic Usage Example")
    print("=" * 40)

    # Initialize RFAI system
    print("Initializing RFAI system...")
    rfai = RecursiveFractalAutonomousIntelligence(
        max_fractal_depth=4,
        base_dimensions=64,
        swarm_size=12,
        quantum_enabled=True
    )

    # Display system status
    status = rfai.get_system_status()
    print(f"System Status: {status['status']}")
    print(f"Fractal Levels: {status['fractal_hierarchy']['levels']}")
    print(f"Total Parameters: {status['fractal_hierarchy']['total_parameters']:,}")
    print(f"Active Agents: {status['agent_swarm']['active_agents']}/{status['agent_swarm']['total_agents']}")
    print(f"Quantum Qubits: {status['quantum_processor']['qubits']}")
    print()

    # Create sample tasks
    tasks = [
        {
            'id': 'pattern_001',
            'type': 'pattern_recognition',
            'complexity': 0.6,
            'data': np.sin(np.linspace(0, 4*np.pi, 64)) + np.random.randn(64) * 0.1,
            'requirements': ['accuracy', 'speed'],
            'priority': 0.8
        },
        {
            'id': 'optimization_001',
            'type': 'optimization',
            'complexity': 0.8,
            'data': np.random.exponential(1.0, 64),
            'requirements': ['global_optimum', 'convergence'],
            'priority': 0.9
        },
        {
            'id': 'memory_001',
            'type': 'memory_management',
            'complexity': 0.4,
            'data': np.random.beta(2, 5, 64),
            'requirements': ['efficiency', 'compression'],
            'priority': 0.7
        }
    ]

    # Process tasks
    print("Processing tasks...")
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"Task {i}: {task['id']} (complexity: {task['complexity']:.2f})")

        result = rfai.process_task(task)
        results.append(result)

        print(f"  Performance: {result['performance_score']:.3f}")
        print(f"  Processing time: {result['processing_time']:.4f}s")
        if 'swarm_output' in result['component_results']:
            swarm_success = result['component_results']['swarm_output']['overall_success_rate']
            print(f"  Swarm success: {swarm_success:.3f}")
        print()

    # Show performance evolution
    performance_history = rfai.system_state['performance_history']
    print("Performance Evolution:")
    for i, perf in enumerate(performance_history, 1):
        print(f"  Task {i}: {perf:.3f}")

    if len(performance_history) > 1:
        improvement = performance_history[-1] - performance_history[0]
        print(f"Overall improvement: {improvement:+.3f} ({improvement/performance_history[0]*100:+.1f}%)")

    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
