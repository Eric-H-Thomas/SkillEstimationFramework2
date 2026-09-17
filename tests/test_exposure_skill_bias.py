from types import SimpleNamespace

from HJEEDS.darts_exposure_skill_bias_sensitivity import make_count_assigner
from HJEEDS.models import AgentTruth


def _config():
    return SimpleNamespace(count_buckets=(1, 2, 3, 4, 5), agents_per_bucket=1)


def _truths():
    return [
        AgentTruth(agent_id=0, log_sigma_true=0.0, log_lambda_true=2.0),
        AgentTruth(agent_id=1, log_sigma_true=0.1, log_lambda_true=1.5),
        AgentTruth(agent_id=2, log_sigma_true=0.2, log_lambda_true=1.0),
        AgentTruth(agent_id=3, log_sigma_true=0.3, log_lambda_true=0.5),
        AgentTruth(agent_id=4, log_sigma_true=0.4, log_lambda_true=0.0),
    ]


def test_quality_aligned_assignment_gives_best_agents_more_observations():
    counts = make_count_assigner("quality_aligned")(_config(), _truths(), 123)

    assert counts == [5, 4, 3, 2, 1]


def test_quality_reversed_assignment_gives_best_agents_fewer_observations():
    counts = make_count_assigner("quality_reversed")(_config(), _truths(), 123)

    assert counts == [1, 2, 3, 4, 5]


def test_randomized_assignment_is_reproducible_and_preserves_buckets():
    assigner = make_count_assigner("randomized")
    first = assigner(_config(), _truths(), 123)
    second = assigner(_config(), _truths(), 123)

    assert first == second
    assert sorted(first) == [1, 2, 3, 4, 5]
    assert first != [5, 4, 3, 2, 1]
