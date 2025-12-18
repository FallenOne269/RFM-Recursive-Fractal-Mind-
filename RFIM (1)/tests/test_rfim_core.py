import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the RFIM core package is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rfim_core.amifs import AMIFS
from rfim_core.dfe import DynamicFractalEncoder
from rfim_core.rsa import RecursiveStructuralAdapter
from rfim_core.semantic_goal import SemanticGoal
from rfim_core.semantic_smart_node import SemanticSmartNode
from rfim_core.smart_fractal_node import SmartFractalNode


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


def test_amifs_default_transforms_are_deterministic(rng):
    amifs = AMIFS(num_functions=3, dimension=2)
    features = rng.normal(size=(3, 2))
    system_state = {"alpha": 0.1, "beta": 0.2}

    first = amifs.generate_attractor(5, features=features, system_state=system_state, iterations=8)
    second = amifs.generate_attractor(5, features=features, system_state=system_state, iterations=8)

    np.testing.assert_allclose(first, second)
    assert first.shape == (5, 2)


def test_amifs_custom_transforms_validation():
    amifs = AMIFS(num_functions=2, dimension=2)
    bad_transforms = [
        (np.eye(3), np.zeros(2)),
        (np.eye(2), np.zeros(3)),
    ]
    with pytest.raises(ValueError):
        amifs._build_transforms(bad_transforms)


def test_amifs_resonance_outputs(rng):
    amifs = AMIFS(num_functions=2, dimension=2)
    data = rng.normal(size=(4, 2))

    def extractor(arr):
        return arr * 0.5

    def state_generator(arr):
        return {"sum": float(arr.sum())}

    result = amifs.computational_resonance(data, extractor, state_generator, iterations=4)

    assert set(result) == {"attractor", "resonance", "state", "features"}
    assert result["features"].shape == (4, 2)
    assert result["state"] == {"sum": pytest.approx(float(data.sum()))}
    assert -1.0 <= result["resonance"] <= 1.0


def test_dfe_registration_and_encoding_determinism(rng):
    encoder = DynamicFractalEncoder(dimension=3)

    def extractor(data):
        return data[:, :3]

    encoder.register_feature_extractor("identity", extractor)
    data = rng.normal(size=(5, 3))

    encoded_first = encoder.encode(data, "identity")
    encoded_second = encoder.encode(data, "identity")

    np.testing.assert_allclose(encoded_first["fim"], encoded_second["fim"])
    stats = encoded_first["statistics"]
    assert stats.energy > 0
    assert stats.mean.shape == (3,)
    assert encoded_first["extractor"] == "identity"


def test_dfe_decode_shape_and_empty_guard():
    encoder = DynamicFractalEncoder(dimension=2)
    encoder.register_feature_extractor("sum", lambda arr: np.sum(arr, axis=1))
    data = np.ones((2, 2))
    encoded = encoder.encode(data, "sum")

    with pytest.raises(ValueError):
        encoder.decode([], shape=(2, 2))

    bad_encoded = [{"fim": np.ones((3, 3)), "statistics": encoded["statistics"]}]
    with pytest.raises(ValueError):
        encoder.decode(bad_encoded, shape=(1, 2))


def test_rsa_adaptation_progresses_with_regularisation():
    adapter = RecursiveStructuralAdapter(learning_rate=0.5, regularisation=0.1)
    initial = np.ones((2, 2))
    updated = np.full((2, 2), 2.0)

    first = adapter.adapt(initial)
    second = adapter.adapt(updated)

    np.testing.assert_allclose(first.structure, np.ones((2, 2)) + np.eye(2) * 0.1)
    expected_structure = first.structure + 0.5 * (updated - first.structure)
    expected_structure += np.eye(2) * 0.1
    np.testing.assert_allclose(second.structure, expected_structure)
    assert second.stability == pytest.approx(np.linalg.norm(updated - first.structure))


def test_semantic_node_recursion_and_reconstruction(rng):
    encoder = DynamicFractalEncoder(dimension=2)
    encoder.register_feature_extractor("pair", lambda arr: arr[:, :2])
    goal = SemanticGoal(name="stability_goal", success_criteria={"stability": lambda x: x < 1.0})
    adapter = RecursiveStructuralAdapter(learning_rate=0.4, regularisation=0.0)
    semantic_node = SemanticSmartNode(goal=goal, encoder=encoder, adapter=adapter, memory_capacity=4)

    root_node = SmartFractalNode(name="root", node=semantic_node, propagation_decay=0.8)
    child_node = SmartFractalNode(name="child", node=semantic_node, propagation_decay=0.5)
    root_node.add_child(child_node)

    data = rng.normal(size=(3, 2))
    signals = root_node.propagate(data, "pair")

    assert set(signals) == {"root", "child"}
    assert signals["root"].payload["goal"] == "stability_goal"
    assert len(semantic_node.memory._store) <= semantic_node.memory_capacity
    assert semantic_node.last_adaptation() is not None

    reconstructed = semantic_node.reconstruct((3,))
    assert reconstructed.shape == (3,)


def test_scenario_smoke_metrics_from_semantic_node(rng):
    encoder = DynamicFractalEncoder(dimension=2)
    encoder.register_feature_extractor("mean", lambda arr: arr.mean(axis=1))
    goal = SemanticGoal(name="metric_goal", description="track stability")
    adapter = RecursiveStructuralAdapter(learning_rate=0.3, regularisation=0.0)
    node = SemanticSmartNode(goal=goal, encoder=encoder, adapter=adapter, memory_capacity=2)

    data = rng.normal(size=(2, 2))
    signal = node.process(data, "mean")

    payload = signal.payload
    assert payload["goal"] == "metric_goal"
    assert set(payload).issuperset({"goal", "satisfied", "stability", "metadata"})


def test_cli_help_and_runner_output(tmp_path):
    cli_path = Path("FRM Model notes/scripts/rfim_cli.py").resolve()
    help_result = subprocess.run(
        [sys.executable, str(cli_path), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Run RFIM operations" in help_result.stdout

    run_result = subprocess.run(
        [sys.executable, str(cli_path), "--generate", "--encode"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Generating fractal attractor" in run_result.stdout
    assert "Encoding data using DFE" in run_result.stdout
