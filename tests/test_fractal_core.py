from rfai.fractal_core import FractalState, RecursiveFractalAlgorithm


def test_recursion_base_case():
    algo = RecursiveFractalAlgorithm(max_depth=2, transform_gain=1.0)
    state = FractalState({"value": 0.0})
    result = algo.recursive_process(state)
    assert result.data["depth"] == 2


def test_self_improve_changes_parameters():
    algo = RecursiveFractalAlgorithm(max_depth=2, transform_gain=1.0)
    initial_gain = algo.transform_gain
    algo.self_improve({"performance": 0.2})
    assert algo.transform_gain != initial_gain

