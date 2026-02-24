import numpy as np
from rfai.dfe import DynamicFractalEncoder


def test_encode_decode_lifecycle():
    encoder = DynamicFractalEncoder()
    fim = encoder.encode({"value": 1}, np.ones(3), context={})
    decoded = encoder.decode(fim.fim_id)
    assert decoded["fim_id"] == fim.fim_id
    assert fim.fim_id in encoder.list()
    encoder.delete(fim.fim_id)
    assert fim.fim_id not in encoder.list()


def test_encode_many_deterministic():
    encoder = DynamicFractalEncoder()
    inputs = ["a", "b", "c"]
    result1 = encoder.encode_many(inputs, n_workers=1)
    result2 = encoder.encode_many(inputs, n_workers=1)
    ids1 = [f.fim_id for f in result1]
    ids2 = [f.fim_id for f in result2]
    assert ids1 != ids2  # unique ids
    assert len(result1) == len(inputs)

