import pytest

import shutil
import numpy as np

import artm

from ..cooking_machine.cubes.perplexity_strategy import retrieve_score_for_strategy
from ..cooking_machine.cubes import RegularizersModifierCube

from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset

ROOT_ID = '<' * 8 + 'start' + '>' * 8

# to run all test
@pytest.fixture(scope="function")
def two_experiment_enviroments(request):
    """ """
    dataset = Dataset('tests/test_data/test_dataset.csv')
    dictionary = dataset.get_dictionary()

    model_artm_1 = artm.ARTM(
        num_processors=1,
        num_topics=5, cache_theta=True,
        num_document_passes=1, dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary)],
    )

    model_artm_2 = artm.ARTM(
        num_processors=1,
        num_topics=5, cache_theta=True,
        num_document_passes=1, dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary)],
    )

    tm_1 = TopicModel(model_artm_1, model_id='new_id_1')
    tm_2 = TopicModel(model_artm_2, model_id='new_id_2')

    experiment_1 = Experiment(
        experiment_id="test_1", save_path="tests/experiments", topic_model=tm_1
    )
    experiment_2 = Experiment(
        experiment_id="test_2", save_path="tests/experiments", topic_model=tm_2
    )

    def resource_teardown():
        """ """
        shutil.rmtree("tests/experiments")
        shutil.rmtree("tests/test_data/test_dataset_batches")

    request.addfinalizer(resource_teardown)

    return tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary


def test_initial_save_load(two_experiment_enviroments):
    """ """
    tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary = two_experiment_enviroments

    experiment_1.set_criteria(0, 'test_criteria')
    experiment_1.save('tests/experiments/test_1')
    experiment = Experiment.load('tests/experiments/test_1')
    tm_id = ROOT_ID
    tm_load = experiment.models[tm_id]
    tm_save = experiment_1.models[tm_id]

    assert tm_save.depth == tm_load.depth
    assert tm_save.parent_model_id == tm_load.parent_model_id
    assert experiment.criteria[0] == ['test_criteria'], 'Experiment failed to load criteria'


def test_simple_experiment(two_experiment_enviroments):
    """ """
    tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary = two_experiment_enviroments

    experiment_1.save('tests/experiments/test_1')
    experiment = Experiment.load('tests/experiments/test_1')
    tm_id = ROOT_ID
    tm = experiment.models[tm_id]

    TAU_GRID = [0.1, 0.5, 1, 5, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test',
                                                                    class_ids='text'),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        tracked_score_function=retrieve_score_for_strategy('PerplexityScore'),
        reg_search="grid"
    )

    cube_2 = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        tracked_score_function=retrieve_score_for_strategy('PerplexityScore'),
        reg_search="grid"
    )

    tmodels = cube(tm, dataset)
    tmodels_2 = cube_2(tm_2, dataset)

    assert len(tmodels) == len(tmodels_2)
    assert cube.strategy.score == cube_2.strategy.score

    for models in zip(tmodels, tmodels_2):
        assert models[0].regularizers['test'].tau == models[1].regularizers['test'].tau
        assert np.array_equal(models[0].get_phi(), models[1].get_phi())
        assert models[0].depth == 2
        assert models[1].depth == 2


def test_double_steps_experiment(two_experiment_enviroments):
    """ """
    tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary = two_experiment_enviroments

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test_first',
                                                                    class_ids='text'),
        "tau_grid": [0.1, 0.5, 1, 5, 10]
    }

    cube_first_1 = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid"
    )

    cube_first_2 = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid"
    )

    tmodels_lvl2_1 = cube_first_1(tm_1, dataset)
    tmodels_lvl2_2 = cube_first_2(tm_2, dataset)

    experiment_1.save('tests/experiments/test_1')
    experiment = Experiment.load('tests/experiments/test_1')
    tmodels_lvl2 = experiment.get_models_by_depth(2)

    TAU_GRID_LVL2 = [0.1, 0.5, 1]

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparseThetaRegularizer(name='test_second'),
        "tau_grid": TAU_GRID_LVL2
    }

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        tracked_score_function=retrieve_score_for_strategy('PerplexityScore'),
        reg_search="grid"
    )

    cube_second_2 = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        tracked_score_function=retrieve_score_for_strategy('PerplexityScore'),
        reg_search="grid"
    )

    tmodels_lvl3 = cube_second(tmodels_lvl2[0], dataset)
    tmodels_lvl3_2 = cube_second_2(tmodels_lvl2_2[0], dataset)

    assert len(tmodels_lvl3) == len(tmodels_lvl3_2)
    assert cube_second.strategy.score == cube_second_2.strategy.score

    for models in zip(tmodels_lvl2_1, tmodels_lvl2):
        assert np.array_equal(models[0].get_phi(), models[1].get_phi())

    for models in zip(tmodels_lvl3, tmodels_lvl3_2):
        assert (
            models[0].regularizers['test_second'].tau == models[1].regularizers['test_second'].tau
        )
        assert np.array_equal(models[0].get_phi(), models[1].get_phi())
        assert models[0].depth == 3
        assert models[1].depth == 3
