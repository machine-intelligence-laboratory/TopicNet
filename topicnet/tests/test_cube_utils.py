import pytest

from topicnet.cooking_machine.cubes.controller_cube import ScoreControllerPerplexity, ControllerAgent
from .fixtures import DATA_AGENT_CONTROLLER_LEN_CHECK, DATA_REG_CONTROLLER_SORT_OF_DECREASING


@pytest.mark.parametrize('values, fraction, answer_true', DATA_REG_CONTROLLER_SORT_OF_DECREASING)
def test_get_two_values_diff(values, fraction, answer_true):
    score_controller = ScoreControllerPerplexity('test', fraction)
    is_out_of_control = score_controller.is_out_of_control(values)

    assert is_out_of_control.answer == answer_true


@pytest.mark.parametrize('agent_blueprint, answer_true', DATA_AGENT_CONTROLLER_LEN_CHECK)
def test_controllers_length(agent_blueprint, answer_true):
    agent = ControllerAgent(**agent_blueprint)

    assert len(agent.score_controllers) == answer_true
