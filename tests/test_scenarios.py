from rfai.sim import scenarios


def test_scenarios_smoke():
    res_a = scenarios.run_scenario_A(steps=10, seed=1)
    res_b = scenarios.run_scenario_B(steps=10, seed=1)
    res_c = scenarios.run_scenario_C(steps=10, seed=1)
    assert "task_completion_rate" in res_a
    assert "oscillation_score" in res_b
    assert "time_efficiency" in res_c

