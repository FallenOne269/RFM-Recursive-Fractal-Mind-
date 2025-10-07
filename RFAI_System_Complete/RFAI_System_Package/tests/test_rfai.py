"""RFAI System Test Suite."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_ROOT.parent

import sys
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from rfai_system import RecursiveFractalAutonomousIntelligence  # noqa: E402


class TestRFAISystem(unittest.TestCase):
    """Functional tests for the RFAI core."""

    def setUp(self) -> None:
        self.rfai = RecursiveFractalAutonomousIntelligence(
            max_fractal_depth=3,
            base_dimensions=32,
            swarm_size=6,
            quantum_enabled=True,
        )
        self.test_task = self._make_task(task_id="test_001", complexity=0.5)

    def _make_task(self, task_id: str, complexity: float) -> dict:
        return {
            "id": task_id,
            "type": "pattern_recognition",
            "complexity": complexity,
            "data": np.random.randn(32),
            "requirements": ["accuracy", "speed"],
            "priority": 0.8,
        }

    def test_system_initialization(self) -> None:
        self.assertEqual(self.rfai.max_depth, 3)
        self.assertEqual(self.rfai.base_dims, 32)
        self.assertEqual(len(self.rfai.agent_swarm), 6)
        self.assertTrue(self.rfai.quantum_enabled)
        self.assertIsNotNone(self.rfai.fractal_hierarchy)

    def test_fractal_processing_valid_input(self) -> None:
        input_data = np.random.randn(32)
        result = self.rfai.fractal_processing(input_data)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 32)
        self.assertFalse(np.isnan(result).any())

    def test_fractal_processing_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.rfai.fractal_processing(np.array([]))

    def test_fractal_processing_nan_input_raises(self) -> None:
        data = np.random.randn(32)
        data[5] = np.nan
        with self.assertRaises(ValueError):
            self.rfai.fractal_processing(data)

    def test_fractal_processing_inf_input_raises(self) -> None:
        data = np.random.randn(32)
        data[10] = np.inf
        with self.assertRaises(ValueError):
            self.rfai.fractal_processing(data)

    def test_fractal_processing_incorrect_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.rfai.fractal_processing(np.random.randn(10))
        with self.assertRaises(ValueError):
            self.rfai.fractal_processing(np.random.randn(64))

    def test_swarm_coordination(self) -> None:
        result = self.rfai.swarm_coordination(self.test_task)
        self.assertIn("overall_success_rate", result)
        self.assertIn("participating_agents", result)
        self.assertIn("swarm_efficiency", result)
        self.assertIsInstance(result["overall_success_rate"], float)
        self.assertGreaterEqual(result["overall_success_rate"], 0.0)
        self.assertLessEqual(result["overall_success_rate"], 1.0)

    def test_quantum_processing_enabled(self) -> None:
        input_data = np.random.randn(32)
        result = self.rfai.quantum_classical_hybrid_processing(input_data)
        self.assertIsInstance(result, np.ndarray)
        self.assertFalse(np.isnan(result).any())

    def test_quantum_processing_fallback(self) -> None:
        non_quantum = RecursiveFractalAutonomousIntelligence(
            max_fractal_depth=3,
            base_dimensions=32,
            swarm_size=6,
            quantum_enabled=False,
        )
        input_data = np.random.randn(32)
        expected = non_quantum.fractal_processing(input_data)
        result = non_quantum.quantum_classical_hybrid_processing(input_data)
        np.testing.assert_array_equal(result, expected)

    def test_task_processing(self) -> None:
        result = self.rfai.process_task(self.test_task)
        self.assertIn("task_id", result)
        self.assertIn("performance_score", result)
        self.assertIn("processing_time", result)
        self.assertIn("component_results", result)
        self.assertIsInstance(result["performance_score"], float)
        self.assertGreaterEqual(result["performance_score"], 0.0)
        self.assertLessEqual(result["performance_score"], 1.0)
        self.assertIsInstance(result["processing_time"], float)
        self.assertGreater(result["processing_time"], 0.0)

    def test_system_status(self) -> None:
        status = self.rfai.get_system_status()
        self.assertIn("system_id", status)
        self.assertIn("status", status)
        self.assertIn("fractal_hierarchy", status)
        self.assertIn("agent_swarm", status)
        self.assertIn("quantum_processor", status)
        self.assertIn("performance", status)

    def test_meta_learning(self) -> None:
        result = self.rfai.meta_learning_optimization()
        self.assertIn("architecture_changes", result)
        self.assertIn("parameter_updates", result)
        self.assertIn("performance_improvement", result)
        self.assertIn("new_capabilities", result)

    def test_multiple_tasks_update_history(self) -> None:
        self.rfai.process_task(self._make_task("task_a", 0.4))
        self.rfai.process_task(self._make_task("task_b", 0.6))
        self.rfai.process_task(self._make_task("task_c", 0.8))
        history = self.rfai.system_state["performance_history"]
        self.assertEqual(len(history), 3)

    def test_state_save_load(self) -> None:
        self.rfai.process_task(self.test_task)
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "test_state.json"
            self.rfai.save_state(state_path.as_posix())
            self.assertTrue(state_path.exists())
            loaded = self.rfai.load_state(state_path.as_posix())
            self.assertIn("system_state", loaded)

    def test_load_state_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.rfai.load_state("missing_state.json")

    def test_load_state_corrupted_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted = Path(tmpdir) / "corrupted.json"
            corrupted.write_text("{invalid json}", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                self.rfai.load_state(corrupted.as_posix())

    def test_extremely_large_input(self) -> None:
        huge_task = self._make_task("huge", 0.95)
        huge_task["data"] = np.random.randn(20000)
        result = self.rfai.process_task(huge_task)
        self.assertIsNotNone(result)
        self.assertIn("performance_score", result)

    def test_malformed_input_raises(self) -> None:
        malformed_task = self._make_task("malformed", 0.5)
        malformed_task["data"] = np.array(["a", None, {}, []], dtype=object)
        with self.assertRaises(ValueError):
            self.rfai.process_task(malformed_task)


class PerformanceTests(unittest.TestCase):
    """Performance and stress tests without loops or conditionals in assertions."""

    def setUp(self) -> None:
        self.rfai = RecursiveFractalAutonomousIntelligence(
            max_fractal_depth=4,
            base_dimensions=64,
            swarm_size=12,
            quantum_enabled=True,
        )

    def _assert_processing_time(self, complexity: float) -> None:
        task = {
            "id": f"speed_test_{complexity}",
            "type": "optimization",
            "complexity": complexity,
            "data": np.random.randn(64),
            "priority": 0.8,
        }
        result = self.rfai.process_task(task)
        self.assertIsNotNone(result)
        self.assertLess(result["processing_time"], 1.0)

    def test_processing_speed_low(self) -> None:
        self._assert_processing_time(0.1)

    def test_processing_speed_medium(self) -> None:
        self._assert_processing_time(0.5)

    def test_processing_speed_high(self) -> None:
        self._assert_processing_time(0.9)

    def test_memory_usage(self) -> None:
        large_task = {
            "id": "memory_test",
            "type": "memory_management",
            "complexity": 0.8,
            "data": np.random.randn(512),
            "priority": 0.9,
        }
        result = self.rfai.process_task(large_task)
        self.assertIsNotNone(result)
        self.assertIn("performance_score", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
