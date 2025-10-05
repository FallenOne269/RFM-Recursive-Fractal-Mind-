"""
Advanced RFAI System Usage Example
==================================

This example demonstrates advanced features including:
- Custom configuration
- System evolution simulation
- Performance monitoring
- State persistence
"""

import sys
import os
import numpy as np
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rfai_system import RecursiveFractalAutonomousIntelligence

def create_custom_config():
    """Create a custom configuration"""
    return {
        "max_fractal_depth": 5,
        "base_dimensions": 128,
        "swarm_size": 20,
        "quantum_enabled": True,
        "learning_settings": {
            "base_learning_rate": 0.002,
            "adaptation_factor": 1.2,
            "performance_threshold": 0.90
        }
    }

def run_evolution_simulation(rfai, iterations=10):
    """Run system evolution simulation"""
    print(f"Running {iterations} evolution iterations...")

    initial_performance = np.mean(rfai.system_state['performance_history']) if rfai.system_state['performance_history'] else 0.0

    for iteration in range(iterations):
        # Create dynamic task
        task = {
            'id': f'evolution_{iteration:03d}',
            'type': np.random.choice(['pattern_recognition', 'optimization', 'memory_management']),
            'complexity': 0.3 + 0.6 * (iteration / iterations),  # Increasing complexity
            'data': generate_complex_data(rfai.base_dims, iteration),
            'priority': np.random.uniform(0.6, 1.0)
        }

        result = rfai.process_task(task)

        if iteration % 2 == 0:
            print(f"  Iteration {iteration+1:2d}: Performance {result['performance_score']:.3f}, "
                  f"Complexity {task['complexity']:.2f}")

    final_performance = np.mean(rfai.system_state['performance_history'][-5:])
    improvement = final_performance - initial_performance if initial_performance > 0 else final_performance

    print(f"Evolution completed:")
    print(f"  Initial performance: {initial_performance:.3f}")
    print(f"  Final performance: {final_performance:.3f}")
    print(f"  Improvement: {improvement:+.3f} ({improvement/max(0.001, initial_performance)*100:+.1f}%)")

    return improvement

def generate_complex_data(size, seed):
    """Generate complex test data with patterns"""
    np.random.seed(seed)

    # Combine multiple patterns
    t = np.linspace(0, 4*np.pi, size)

    # Fractal-like pattern
    signal = np.zeros(size)
    for scale in range(1, 5):
        freq = 2 ** scale
        amplitude = 1.0 / scale
        signal += amplitude * np.sin(freq * t + np.random.uniform(0, np.pi))

    # Add noise
    signal += np.random.normal(0, 0.1, size)

    return signal

def analyze_agent_performance(rfai):
    """Analyze individual agent performance"""
    print("Agent Performance Analysis:")
    print("-" * 40)

    specializations = {}
    for agent in rfai.agent_swarm:
        spec = agent.specialization
        if spec not in specializations:
            specializations[spec] = []
        specializations[spec].append(agent.performance_metrics['task_completion_rate'])

    for spec, rates in specializations.items():
        avg_rate = np.mean(rates)
        print(f"  {spec:25}: {avg_rate:.3f} avg ({len(rates)} agents)")

def monitor_system_resources(rfai):
    """Monitor system resource usage"""
    total_params = sum(sum(module.weights.size for module in modules) 
                      for modules in rfai.fractal_hierarchy.values())

    active_modules = 0
    for modules in rfai.fractal_hierarchy.values():
        for module in modules:
            if np.mean(np.abs(module.activation_state)) > 0.01:
                active_modules += 1

    print("System Resource Usage:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Active modules: {active_modules}")
    print(f"  Memory efficiency: {active_modules/sum(len(modules) for modules in rfai.fractal_hierarchy.values()):.2%}")

def main():
    print("RFAI System - Advanced Usage Example")
    print("=" * 45)

    # Create custom configuration
    config = create_custom_config()
    print("Custom Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    print()

    # Initialize with custom config
    print("Initializing advanced RFAI system...")
    rfai = RecursiveFractalAutonomousIntelligence(**{k: v for k, v in config.items() if k != 'learning_settings'})

    # Run initial tasks to establish baseline
    print("Establishing baseline performance...")
    baseline_tasks = [
        {
            'id': f'baseline_{i:03d}',
            'type': 'pattern_recognition',
            'complexity': 0.5,
            'data': np.random.randn(config['base_dimensions']),
            'priority': 0.8
        }
        for i in range(3)
    ]

    for task in baseline_tasks:
        rfai.process_task(task)

    baseline_performance = np.mean(rfai.system_state['performance_history'])
    print(f"Baseline performance: {baseline_performance:.3f}")
    print()

    # Run evolution simulation
    improvement = run_evolution_simulation(rfai, 15)
    print()

    # Analyze system state
    analyze_agent_performance(rfai)
    print()

    monitor_system_resources(rfai)
    print()

    # Performance benchmarking
    print("Performance Benchmarking:")
    benchmark_start = time.time()

    benchmark_task = {
        'id': 'benchmark_001',
        'type': 'optimization',
        'complexity': 0.9,
        'data': np.random.randn(config['base_dimensions']),
        'priority': 1.0
    }

    result = rfai.process_task(benchmark_task)
    benchmark_end = time.time()

    print(f"  Benchmark task performance: {result['performance_score']:.3f}")
    print(f"  Benchmark processing time: {benchmark_end - benchmark_start:.4f}s")
    print(f"  Throughput: {1/(benchmark_end - benchmark_start):.1f} tasks/second")
    print()

    # Save system state
    state_file = 'advanced_rfai_state.json'
    rfai.save_state(state_file)
    print(f"System state saved to: {state_file}")

    # Final system status
    final_status = rfai.get_system_status()
    print("Final System Status:")
    print(f"  Tasks processed: {final_status['performance']['tasks_processed']}")
    print(f"  Average performance: {final_status['performance']['avg_performance']:.3f}")
    print(f"  Learning trend: {final_status['performance']['learning_trend']:.4f}")

    print("\nAdvanced example completed successfully!")

if __name__ == "__main__":
    main()
