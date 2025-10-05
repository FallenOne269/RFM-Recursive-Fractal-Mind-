"""
RFAI System Test Suite
=====================

Comprehensive tests for the Recursive Fractal Autonomous Intelligence system.
"""

import sys
import os
import numpy as np
import unittest
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rfai_system import RecursiveFractalAutonomousIntelligence

class TestRFAISystem(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.rfai = RecursiveFractalAutonomousIntelligence(
            max_fractal_depth=3,  # Smaller for testing
            base_dimensions=32,
            swarm_size=6,
            quantum_enabled=True
        )

        self.test_task = {
            'id': 'test_001',
            'type': 'pattern_recognition',
            'complexity': 0.5,
            'data': np.random.randn(32),
            'requirements': ['accuracy', 'speed'],
            'priority': 0.8
        }

    def test_system_initialization(self):
        """Test system initialization"""
        self.assertEqual(self.rfai.max_depth, 3)
        self.assertEqual(self.rfai.base_dims, 32)
        self.assertEqual(len(self.rfai.agent_swarm), 6)
        self.assertTrue(self.rfai.quantum_enabled)
        self.assertIsNotNone(self.rfai.fractal_hierarchy)

    def test_fractal_processing(self):
        """Test fractal processing functionality"""
        input_data = np.random.randn(32)
        result = self.rfai.fractal_processing(input_data)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 32)
        self.assertFalse(np.isnan(result).any())

    def test_swarm_coordination(self):
        """Test agent swarm coordination"""
        result = self.rfai.swarm_coordination(self.test_task)

        self.assertIn('overall_success_rate', result)
        self.assertIn('participating_agents', result)
        self.assertIn('swarm_efficiency', result)
        self.assertIsInstance(result['overall_success_rate'], float)
        self.assertGreaterEqual(result['overall_success_rate'], 0.0)
        self.assertLessEqual(result['overall_success_rate'], 1.0)

    def test_quantum_processing(self):
        """Test quantum-classical hybrid processing"""
        if self.rfai.quantum_enabled:
            input_data = np.random.randn(32)
            result = self.rfai.quantum_classical_hybrid_processing(input_data)

            self.assertIsInstance(result, np.ndarray)
            self.assertFalse(np.isnan(result).any())

    def test_task_processing(self):
        """Test complete task processing pipeline"""
        result = self.rfai.process_task(self.test_task)

        # Check result structure
        self.assertIn('task_id', result)
        self.assertIn('performance_score', result)
        self.assertIn('processing_time', result)
        self.assertIn('component_results', result)

        # Check performance score
        self.assertIsInstance(result['performance_score'], float)
        self.assertGreaterEqual(result['performance_score'], 0.0)
        self.assertLessEqual(result['performance_score'], 1.0)

        # Check processing time
        self.assertIsInstance(result['processing_time'], float)
        self.assertGreater(result['processing_time'], 0.0)

    def test_system_status(self):
        """Test system status reporting"""
        status = self.rfai.get_system_status()

        self.assertIn('system_id', status)
        self.assertIn('status', status)
        self.assertIn('fractal_hierarchy', status)
        self.assertIn('agent_swarm', status)
        self.assertIn('quantum_processor', status)
        self.assertIn('performance', status)

    def test_meta_learning(self):
        """Test meta-learning optimization"""
        result = self.rfai.meta_learning_optimization()

        self.assertIn('architecture_changes', result)
        self.assertIn('parameter_updates', result)
        self.assertIn('performance_improvement', result)
        self.assertIn('new_capabilities', result)

    def test_multiple_tasks(self):
        """Test processing multiple tasks"""
        tasks = [
            {
                'id': f'test_{i:03d}',
                'type': 'pattern_recognition',
                'complexity': np.random.uniform(0.3, 0.9),
                'data': np.random.randn(32),
                'priority': np.random.uniform(0.5, 1.0)
            }
            for i in range(5)
        ]

        results = []
        for task in tasks:
            result = self.rfai.process_task(task)
            results.append(result)

        self.assertEqual(len(results), 5)

        # Check for performance improvement trend
        performances = [r['performance_score'] for r in results]
        self.assertEqual(len(self.rfai.system_state['performance_history']), 5)

    def test_state_save_load(self):
        """Test system state saving and loading"""
        # Process a task to create some state
        self.rfai.process_task(self.test_task)

        # Save state
        state_file = 'test_state.json'
        self.rfai.save_state(state_file)
        self.assertTrue(os.path.exists(state_file))

        # Load state
        self.rfai.load_state(state_file)

        # Clean up
        os.remove(state_file)

class PerformanceTests(unittest.TestCase):
    """Performance and stress tests"""

    def setUp(self):
        self.rfai = RecursiveFractalAutonomousIntelligence(
            max_fractal_depth=4,
            base_dimensions=64,
            swarm_size=12,
            quantum_enabled=True
        )

    def test_processing_speed(self):
        """Test processing speed with different task complexities"""
        import time

        complexities = [0.1, 0.3, 0.5, 0.7, 0.9]
        processing_times = []

        for complexity in complexities:
            task = {
                'id': f'speed_test_{complexity}',
                'type': 'optimization',
                'complexity': complexity,
                'data': np.random.randn(64),
                'priority': 0.8
            }

            start_time = time.time()
            result = self.rfai.process_task(task)
            end_time = time.time()

            processing_times.append(end_time - start_time)

        # Check that processing times are reasonable
        for pt in processing_times:
            self.assertLess(pt, 1.0)  # Should process in less than 1 second

    def test_memory_usage(self):
        """Test system with large inputs"""
        large_task = {
            'id': 'memory_test',
            'type': 'memory_management',
            'complexity': 0.8,
            'data': np.random.randn(512),  # Larger input
            'priority': 0.9
        }

        # Should handle large inputs gracefully
        result = self.rfai.process_task(large_task)
        self.assertIsNotNone(result)
        self.assertIn('performance_score', result)

if __name__ == '__main__':
    print("Running RFAI System Tests...")
    print("=" * 50)

    # Run tests
    unittest.main(verbosity=2)
