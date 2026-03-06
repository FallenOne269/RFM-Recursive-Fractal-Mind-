import numpy as np
from rfai.amifs import AMIFS, TransformationParameters, AdaptiveComponent


def test_affine_transform_correctness():
    params = TransformationParameters(scale=1.0, rotation=0.0, translation=(1.0, 1.0))
    amifs = AMIFS(params)
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    transformed = amifs.base_transformations[0].apply(points)
    assert np.allclose(transformed, points + 1.0)


def test_precision_control():
    params = TransformationParameters(precision="float32")
    amifs = AMIFS(params)
    points = np.zeros((2, 2))
    out = amifs.generate(points, steps=1)
    assert out.dtype == np.float32


def test_adaptive_component_cache():
    params = TransformationParameters()
    component = AdaptiveComponent(params)
    features = np.array([1.0, 2.0])
    ctx = np.array([0.5, 1.5])
    first = component.compute_delta(features, ctx)
    second = component.compute_delta(features, ctx)
    assert first == second

