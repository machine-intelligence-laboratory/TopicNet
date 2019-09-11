import numpy as np
import pandas as pd
import pytest
from ..viewers import spectrum
from scipy.spatial import distance
from ..cooking_machine.models.base_model import BaseModel

# to run all test
@pytest.fixture(scope="function")
def experiment_enviroment(request):
    """
    Creates environment for experiment.

    """

    problem_size_x1 = 10
    problem_size_y = 2
    matrix_left = np.random.rand(problem_size_x1, problem_size_y)
    matrix = distance.squareform(distance.pdist(matrix_left, 'jensenshannon'))
    np.fill_diagonal(matrix, 10 * np.max(matrix))

    return matrix


def test_triplet_generator():
    """ """
    left_answer = list(spectrum.generate_all_segments(6))
    right_answer = [[0, 2, 4]]

    np.testing.assert_array_equal(left_answer, right_answer)


def test_random_generator_len():
    """ """
    left_answer = len(spectrum.generate_index_candidates(10))
    right_answer = 3
    assert left_answer == right_answer


def test_random_generator_sort():
    """ """
    left_answer = spectrum.generate_index_candidates(10)
    assert np.all(np.diff(left_answer) > 0)


def test_swap_all_unique(experiment_enviroment):
    """
    Checks if swap works.

    """
    matrix = experiment_enviroment
    init = list(np.append(np.arange(10), [0]))
    seq = [0, 4, 8]
    tour = spectrum.make_three_opt_swap(init, matrix, seq)[0]
    assert set(range(10)) == set(tour)


def test_swap_same_len(experiment_enviroment):
    """ """
    matrix = experiment_enviroment
    init = list(np.append(np.arange(10), [0]))
    seq = [0, 4, 8]
    tour = spectrum.make_three_opt_swap(init, matrix, seq)[0]
    assert len(init) == len(tour)


def test_solve_tsp():
    """ """
    matrix = np.array([
        [0.0, 0.0],
        [0.0, 1],
        [0.0, -1],
        [5, 0.0],
        [-5, 0.0],
        [0.5, 0.5],
    ])
    distance_m = distance.squareform(distance.pdist(matrix, 'euclidean'))
    np.fill_diagonal(distance_m, 10 * np.max(distance_m))
    init = list(np.append(np.arange(6), [0]))

    right_answer = 21.432838377440824
    path = spectrum.get_three_opt_path(init, distance_m)
    left_answer = np.sum(distance_m[path[:-1], path[1:]])
    np.testing.assert_almost_equal(left_answer, right_answer, decimal=15)


def test_short_path():
    """ """
    matrix = np.array([
        [0.0, 0.0],
        [0.0, 1],
        [0.0, -1],
        [3, 0.0],
    ])
    distance_m = distance.squareform(distance.pdist(matrix, 'jensenshannon'))
    np.fill_diagonal(distance_m, 10 * np.max(distance_m))

    right_answer = spectrum.get_nearest_neighbour_init(matrix)
    with pytest.warns(UserWarning):
        left_answer = spectrum.get_annealed_spectrum(matrix, 1e3)[0]

    np.testing.assert_array_equal(left_answer, right_answer)


def test_viewer():
    class dummy_model(BaseModel):
        def get_phi(self, class_ids=None):
            matrix = pd.DataFrame([
                [0.0, 0.0],
                [0.0, 1],
                [0.0, -1],
                [5, 0.0],
                [-5, 0.0],
                [0.5, 0.5],
            ]).T
            return matrix
    dummy_model_instance = dummy_model()
    path = spectrum.TopicSpectrumViewer(
        model=dummy_model_instance,
        metric='euclidean',
        early_stopping=1000,
    ).view()

    distance_m = distance.squareform(
        distance.pdist(dummy_model_instance.get_phi().values.T, 'euclidean')
    )

    right_answer = 21.432838377440824
    left_answer = np.sum(distance_m[path[:-1], path[1:]])
    np.testing.assert_almost_equal(left_answer, right_answer, decimal=15)
