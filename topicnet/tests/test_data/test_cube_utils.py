import pytest

from topicnet.cooking_machine.cubes.controller_cube import get_two_values_diff

data_2values = [
    (1, 2, 1),
    (0, 2, 2),
    (100, 115, 0.15)
]


@pytest.mark.parametrize('min_val, max_val, true_diff', data_2values)
def test_get_two_values_diff(min_val, max_val, true_diff):
    diff = get_two_values_diff(min_val, max_val)

    assert diff == true_diff
