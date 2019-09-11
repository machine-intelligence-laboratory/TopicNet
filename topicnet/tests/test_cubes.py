import pytest

import shutil

import artm
import numpy as np

from ..cooking_machine.cubes.base_cube import retrieve_perplexity_score
from ..cooking_machine.cubes.greedy_weights_strategy import GreedyWeightedStrategy
from ..cooking_machine.cubes.perplexity_strategy import retrieve_score_for_strategy
from ..cooking_machine.cubes.perplexity_strategy import PerplexityStrategy

from ..cooking_machine.cubes import RegularizersModifierCube
from ..cooking_machine.cubes import ClassIdModifierCube
from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset


# to run all test
@pytest.fixture(scope="function")
def experiment_enviroment(request):
    """ """
    dataset = Dataset('tests/test_data/test_dataset.csv')
    dictionary = dataset.get_dictionary()

    model_artm = artm.ARTM(
        num_processors=1,
        num_topics=5, cache_theta=True,
        num_document_passes=1, dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary)],
    )

    tm = TopicModel(model_artm, model_id='new_id')
    experiment = Experiment(experiment_id="test", save_path="tests/experiments", topic_model=tm)

    def resource_teardown():
        """ """
        shutil.rmtree("tests/experiments")
        shutil.rmtree("tests/test_data/test_dataset_batches")

    request.addfinalizer(resource_teardown)

    return tm, dataset, experiment, dictionary


def test_initial_tm(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    assert tm.depth == 1
    assert tm.parent_model_id == '<' * 8 + 'start' + '>' * 8


def test_simple_experiment(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID = [0.1, 0.5, 1, 5, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test',
                                                                    class_ids='text'),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid"
    )

    tmodels = cube(tm, dataset)

    assert len(tmodels) == len(TAU_GRID)

    for i, one_model in enumerate(tmodels):
        assert one_model.regularizers['test'].tau == TAU_GRID[i]
        assert one_model.regularizers['test'].tau == TAU_GRID[i]


def test_simple_experiment_pair_strategy(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID = [0.1, 0.5, 1, 5, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test',
                                                                    class_ids='text'),
        "tau_grid": TAU_GRID
    }

    cube_pair = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="pair"
    )

    tmodels_pair = cube_pair(tm, dataset)

    assert len(tmodels_pair) == 5

    for i, one_model in enumerate(tmodels_pair):
        assert one_model.regularizers['test'].tau == TAU_GRID[i]


def test_double_steps_experiment(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test_first',
                                                                    class_ids='text'),
        "tau_grid": [0.1, 0.5, 1, 5, 10]
    }

    cube_first = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid"
    )

    tmodels_lvl2 = cube_first(tm, dataset)

    TAU_GRID_LVL2 = [0.1, 0.5, 1]

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparseThetaRegularizer(name='test_second'),
        "tau_grid": TAU_GRID_LVL2
    }

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid"
    )

    tmodels_lvl3 = cube_second(tmodels_lvl2[0], dataset)

    assert len(tmodels_lvl3) == len(TAU_GRID_LVL2)
    for i, one_model in enumerate(tmodels_lvl3):
        assert one_model.regularizers['test_second'].tau == TAU_GRID_LVL2[i]
        assert one_model.depth == 3


def test_two_regularizers_on_step_experiment(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID_FIRST = (1, 5, 10)
    TAU_GRID_SECOND = (-0.1, -0.5)

    regularizer_parameters = [
        {
            "regularizer": artm.SmoothSparsePhiRegularizer(name='test_first', tau=1),
            "tau_grid": TAU_GRID_FIRST
        },
        {
            "regularizer": artm.SmoothSparsePhiRegularizer(name='test_second', tau=1),
            "tau_grid": TAU_GRID_SECOND
        }
    ]

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid"
    )

    tmodels_lvl2 = cube(tm, dataset)

    TAU_FOR_CHECKING = [
        (1, -0.1), (1, -0.5), (5, -0.1), (5, -0.5), (10, -0.1), (10, -0.5)
    ]

    assert len(tmodels_lvl2) == len(TAU_FOR_CHECKING)
    for i, one_model in enumerate(tmodels_lvl2):
        taus = (
            one_model.regularizers['test_first'].tau,
            one_model.regularizers['test_second'].tau,
        )
        assert taus == TAU_FOR_CHECKING[i]

    for lvl2_model in tmodels_lvl2:
        assert lvl2_model.parent_model_id == tm.model_id
        assert lvl2_model.depth == 2


def test_two_regularizers_on_step_experiment_pair_grid(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID_FIRST = (1, 5, 10)
    TAU_GRID_SECOND = (-0.1, -0.5, -0.3)

    regularizer_parameters = [
        {
            "regularizer": artm.SmoothSparsePhiRegularizer(name='test_first', tau=1),
            "tau_grid": TAU_GRID_FIRST
        },
        {
            "regularizer": artm.SmoothSparsePhiRegularizer(name='test_second', tau=1),
            "tau_grid": TAU_GRID_SECOND
        }
    ]

    cube_pair = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="pair"
    )

    tmodels_lvl2_pair = cube_pair(tm, dataset)

    TAU_FOR_CHECKING = [
        (1, -0.1), (5, -0.5), (10, -0.3)
    ]

    assert len(tmodels_lvl2_pair) == 3
    for i, one_model in enumerate(tmodels_lvl2_pair):
        taus = (
            one_model.regularizers['test_first'].tau,
            one_model.regularizers['test_second'].tau,
        )
        assert taus == TAU_FOR_CHECKING[i]


def test_modefier_cube_on_two_steps_experiment(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID_FIRST = (1, 5, 10)
    TAU_GRID_SECOND = (-0.1, -0.5, -0.3)

    regularizer_parameters = [
        {
            "regularizer": artm.SmoothSparsePhiRegularizer(name='test_first', tau=1),
            "tau_grid": TAU_GRID_FIRST
        },
        {
            "regularizer": artm.SmoothSparsePhiRegularizer(name='test_second', tau=1),
            "tau_grid": TAU_GRID_SECOND
        }
    ]

    cube_pair = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="pair"
    )

    tmodels_lvl2_pair = cube_pair(tm, dataset)

    regularizer_parameters_second = {
        "name": 'test_second',
        "tau_grid": [-0.2, -0.4, -0.6, -0.7]
    }

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters_second,
        reg_search="pair"
    )

    tmodels_second = cube_second(tmodels_lvl2_pair[1], dataset)
    TAU_FOR_CHECKING = [
        (5, -0.2), (5, -0.4), (5, -0.6), (5, -0.7)
    ]

    assert len(tmodels_second) == 4
    for i, one_model in enumerate(tmodels_second):
        taus = (
            one_model.regularizers['test_first'].tau,
            one_model.regularizers['test_second'].tau,
        )
        assert taus == TAU_FOR_CHECKING[i]


def test_class_id_cube(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "@text": [1, 2, 3],
        "@ngramms": [1, 2],
    }

    cube = ClassIdModifierCube(
        num_iter=1,
        class_id_parameters=class_id_params,
        reg_search="grid"
    )

    tmodels_lvl2 = cube(tm, dataset)

    CLASS_IDS_FOR_CHECKING = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    assert len(tmodels_lvl2) == 6

    for i, one_model in enumerate(tmodels_lvl2):
        assert one_model.class_ids['@text'] == CLASS_IDS_FOR_CHECKING[i][0]
        assert one_model.class_ids['@ngramms'] == CLASS_IDS_FOR_CHECKING[i][1]


# most likely fail
def test_phi_matrix_after_class_ids_cube(experiment_enviroment):
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "@text": [0, 1, 2],
        "@ngramms": [1, 0, 2],
    }

    cube = ClassIdModifierCube(
        num_iter=5,
        class_id_parameters=class_id_params,
        reg_search="pair"
    )

    tmodels = cube(tm, dataset)
    for first_ind in range(len(tmodels)):
        for second_ind in range(first_ind + 1, len(tmodels)):
            phi_first = tmodels[first_ind].get_phi_dense()[0]
            phi_second = tmodels[second_ind].get_phi_dense()[0]
            if phi_first.shape == phi_second.shape:
                assert (np.prod(phi_first.shape) - np.sum(phi_first == phi_second)) > 0, \
                    'Phi matrixes are the same after class_ids_cube.'


def test_class_id_cube_strategy(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "@text": list(np.arange(0, 1.0, 0.25)),
        "@ngramms": list(np.arange(0, 2.05, 0.25)),
    }

    cube = ClassIdModifierCube(
        num_iter=1,
        class_id_parameters=class_id_params,
        reg_search="grid",
        strategy=GreedyWeightedStrategy(),
        tracked_score_function=retrieve_perplexity_score
    )

    tmodels_lvl2 = cube(tm, dataset)

    CLASS_IDS_FOR_CHECKING = [(1.0, 0.0), (1.0, 0.0), (0.8, 0.2), (0.667, 0.333),
                              (0.571, 0.429), (0.5, 0.5), (0.444, 0.556),
                              (0.4, 0.6), (0.364, 0.636), (0.333, 0.667)]
    assert len(tmodels_lvl2) == 10

    for i, one_model in enumerate(tmodels_lvl2):
        assert np.round(one_model.class_ids['@text'], 3) == CLASS_IDS_FOR_CHECKING[i][0]
        assert np.round(one_model.class_ids['@ngramms'], 3) == CLASS_IDS_FOR_CHECKING[i][1]

    assert cube.strategy.best_point == [['', '@text', 1.0], ['', '@ngramms', 0.0]]


def test_class_id_cube_strategy_elliptic_paraboloid(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "@text": list(np.arange(0, 1.0, 0.25)),
        "@ngramms": list(np.arange(0, 2.05, 0.25)),
    }

    def retrieve_elliptic_paraboloid_score(topic_model):
        """ """
        model = topic_model._model
        return -((model.class_ids["@text"]-0.6-model.class_ids["@ngramms"]) ** 2 +
                 (model.class_ids["@text"]-0.6+model.class_ids["@ngramms"]/2) ** 2)

    cube = ClassIdModifierCube(
        num_iter=1,
        class_id_parameters=class_id_params,
        reg_search="grid",
        strategy=GreedyWeightedStrategy(),
        tracked_score_function=retrieve_elliptic_paraboloid_score
    )

    tmodels_lvl2 = cube(tm, dataset)

    CLASS_IDS_FOR_CHECKING = [(1.0, 0.0), (1.0, 0.0), (0.8, 0.2), (0.667, 0.333),
                              (0.571, 0.429), (0.5, 0.5), (0.444, 0.556),
                              (0.4, 0.6), (0.364, 0.636), (0.333, 0.667)]
    assert len(tmodels_lvl2) == 10

    for i, one_model in enumerate(tmodels_lvl2):
        assert np.round(one_model.class_ids['@text'], 3) == CLASS_IDS_FOR_CHECKING[i][0]
        assert np.round(one_model.class_ids['@ngramms'], 3) == CLASS_IDS_FOR_CHECKING[i][1]

    assert cube.strategy.best_point == [['', '@text', 0.8], ['', '@ngramms', 0.2]]


def test_class_id_cube_strategy_rosenbrock(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "@text": list(np.arange(0, 1.0, 0.25)),
        "@ngramms": list(np.arange(0, 2.05, 0.25)),
    }

    def retrieve_rosenbrock_score(topic_model):
        """ """
        model = topic_model._model
        return -((1 - model.class_ids["@text"]) ** 2 +
                 100*(model.class_ids["@ngramms"] - model.class_ids["@text"]) ** 2)

    cube = ClassIdModifierCube(
        num_iter=1,
        class_id_parameters=class_id_params,
        reg_search="grid",
        strategy=GreedyWeightedStrategy(),
        tracked_score_function=retrieve_rosenbrock_score
    )

    tmodels_lvl2 = cube(tm, dataset)

    CLASS_IDS_FOR_CHECKING = [(1.0, 0.0), (1.0, 0.0), (0.8, 0.2), (0.667, 0.333),
                              (0.571, 0.429), (0.5, 0.5), (0.444, 0.556),
                              (0.4, 0.6), (0.364, 0.636), (0.333, 0.667)]
    assert len(tmodels_lvl2) == 10

    for i, one_model in enumerate(tmodels_lvl2):
        assert np.round(one_model.class_ids['@text'], 3) == CLASS_IDS_FOR_CHECKING[i][0]
        assert np.round(one_model.class_ids['@ngramms'], 3) == CLASS_IDS_FOR_CHECKING[i][1]

    assert cube.strategy.best_point == [['', '@text', 0.5], ['', '@ngramms', 0.5]]


def test_class_id_cube_strategy_3d_parabolic(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "@text": list(np.arange(0, 1.0, 0.25)),
        "@ngramms": list(np.arange(0, 1.05, 0.25)),
        "@str": list(np.arange(0, 1.05, 0.25))
    }

    def retrieve_parabolic_3d_score(topic_model):
        """ """
        model = topic_model._model
        return -((1 - model.class_ids["@text"]) ** 2 + (1 - model.class_ids["@ngramms"]) ** 2 +
                 (1 - model.class_ids["@str"]) ** 2)

    cube = ClassIdModifierCube(
        num_iter=1,
        class_id_parameters=class_id_params,
        reg_search="grid",
        strategy=GreedyWeightedStrategy(),
        tracked_score_function=retrieve_parabolic_3d_score,
        verbose=True
    )

    tmodels_lvl2 = cube(tm, dataset)

    CLASS_IDS_FOR_CHECKING = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.8, 0.2, 0.0),
                              (0.667, 0.333, 0.0), (0.571, 0.429, 0.0), (0.5, 0.5, 0.0),
                              (0.5, 0.5, 0.0), (0.444, 0.444, 0.111), (0.4, 0.4, 0.2),
                              (0.364, 0.364, 0.273), (0.333, 0.333, 0.333)]
    assert len(tmodels_lvl2) == 11

    for i, one_model in enumerate(tmodels_lvl2):
        assert np.round(one_model.class_ids['@text'], 3) == CLASS_IDS_FOR_CHECKING[i][0]
        assert np.round(one_model.class_ids['@ngramms'], 3) == CLASS_IDS_FOR_CHECKING[i][1]
        assert np.round(one_model.class_ids['@str'], 3) == CLASS_IDS_FOR_CHECKING[i][2]

    assert cube.strategy.best_point == [['', '@text', 0.3333333333333333],
                                        ['', '@ngramms', 0.3333333333333333],
                                        ['', '@str', 0.3333333333333333]]


def test_topic_model_score(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    num_values = len(tm.scores['PerplexityScore'])
    assert num_iterations == num_values


def test_perplexity_strategy_grid(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID = [0.1, 0.5, 1, 5, 50]
    regularizer_parameters = {
        "regularizer": artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=3,
        regularizer_parameters=regularizer_parameters,
        strategy=PerplexityStrategy(1, 5),
        tracked_score_function=retrieve_score_for_strategy('PerplexityScore'),
        reg_search="grid"
    )
    with pytest.warns(UserWarning):
        tmodels = cube(tm, dataset)

    SCORES = [3.756, 3.755, 3.751, 3.746, 3.558, 2.543]
    assert len(tmodels) == len(TAU_GRID) + 1  # for tau=0

    assert list(map(lambda score: np.round(score, 3), cube.strategy.score)) == SCORES

    assert cube.strategy.best_point[0][2] == 50


def test_perplexity_strategy_add(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    regularizer_parameters = {
        "regularizer": artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'),
        "tau_grid": []
    }

    cube = RegularizersModifierCube(
        num_iter=3,
        regularizer_parameters=regularizer_parameters,
        strategy=PerplexityStrategy(1, 1, 5),
        tracked_score_function=retrieve_score_for_strategy('PerplexityScore'),
        reg_search='add',
        verbose=True
    )
    with pytest.warns(UserWarning):
        tmodels = cube(tm, dataset)

    SCORES = [3.756, 3.746, 3.734, 3.602, 3.58, 3.558]
    print(list(map(lambda score: np.round(score, 3), cube.strategy.score)))
    assert len(tmodels) == 5 + 1  # for tau=0

    assert list(map(lambda score: np.round(score, 3), cube.strategy.score)) == SCORES

    assert cube.strategy.best_point[0][2] == 5


def test_perplexity_strategy_mul(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    regularizer_parameters = {
        "regularizer": artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'),
        "tau_grid": []
    }

    cube = RegularizersModifierCube(
        num_iter=3,
        regularizer_parameters=regularizer_parameters,
        strategy=PerplexityStrategy(1, 5, 25),
        tracked_score_function=retrieve_score_for_strategy('PerplexityScore'),
        reg_search='mul',
        verbose=True
    )
    with pytest.warns(UserWarning):
        tmodels = cube(tm, dataset)

    SCORES = [3.756, 3.746, 3.558, 4.645]
    print(list(map(lambda score: np.round(score, 3), cube.strategy.score)))
    assert len(tmodels) == 3 + 1  # for tau=0

    assert list(map(lambda score: np.round(score, 3), cube.strategy.score)) == SCORES

    assert cube.strategy.best_point[0][2] == 5
