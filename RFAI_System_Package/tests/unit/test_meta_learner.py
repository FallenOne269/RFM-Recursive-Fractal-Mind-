"""Unit tests for the meta learner."""

from __future__ import annotations

from meta_learner import MetaLearner
from utils import MetaLearnerConfig


def test_meta_learner_adjusts_learning_rate() -> None:
    meta = MetaLearner(MetaLearnerConfig(base_learning_rate=0.01, adaptation_factor=1.1, performance_threshold=0.5))
    output_low = meta.step(0.3)
    assert output_low.status == "accelerated"
    output_high = meta.step(0.9)
    assert output_high.status == "stabilized"
    assert output_high.updated_learning_rate < output_low.updated_learning_rate
