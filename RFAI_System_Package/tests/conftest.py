"""Pytest configuration for the RFAI system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import load_config


@pytest.fixture
def config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base_config = load_config()
    config_dict = {
        "fractal_engine": {
            "max_depth": base_config.fractal.max_depth,
            "base_dimensions": base_config.fractal.base_dimensions,
            "noise_scale": base_config.fractal.noise_scale,
        },
        "swarm_coordinator": {
            "swarm_size": base_config.swarm.swarm_size,
            "max_parallel_tasks": base_config.swarm.max_parallel_tasks,
        },
        "quantum_processor": {
            "enabled": base_config.quantum.enabled,
            "qubit_count": base_config.quantum.qubit_count,
            "sampling_runs": base_config.quantum.sampling_runs,
        },
        "meta_learner": {
            "base_learning_rate": base_config.meta.base_learning_rate,
            "adaptation_factor": base_config.meta.adaptation_factor,
            "performance_threshold": base_config.meta.performance_threshold,
        },
        "persistence": {
            "state_dir": str(tmp_path_factory.mktemp("state")),
            "version": base_config.persistence.version,
        },
    }
    config_file = tmp_path_factory.mktemp("config") / "config.json"
    config_file.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    return config_file
