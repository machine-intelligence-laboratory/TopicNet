import pytest
import warnings
import os
import shutil

import artm

from time import sleep
from ..cooking_machine.cubes import RegularizersModifierCube, CubeCreator
from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.models.example_score import ScoreExample
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1


def resource_teardown():
    """ """
    if os.path.exists("tests/experiments"):
        shutil.rmtree("tests/experiments")
    if os.path.exists("tests/test_data/test_dataset_batches"):
        shutil.rmtree("tests/test_data/test_dataset_batches")


def setup_function():
    resource_teardown()


def teardown_function():
    resource_teardown()


# to run all test
@pytest.fixture(scope="function")
def experiment_enviroment(request):
    """ """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
        dataset = Dataset('tests/test_data/test_dataset.csv')
        dictionary = dataset.get_dictionary()

    model_artm = artm.ARTM(
        num_processors=1,
        num_topics=5, cache_theta=True,
        num_document_passes=1, dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore', )],
    )
    model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
    ex_score = ScoreExample()
    tm = TopicModel(model_artm, model_id='new_id', custom_scores={'example_score': ex_score})
    # experiment starts without model
    experiment = Experiment(tm, experiment_id="test", save_path="tests/experiments")
    return tm, dataset, experiment, dictionary


def test_simple_experiment(experiment_enviroment):
    """ """
    # experiment with one level created by CubeCreator
    tm, dataset, experiment, dictionary = experiment_enviroment

    parameters = [
        {
            'name': 'num_topics',
            'values': [2, 3, 4]
        }, {
            'name': 'num_document_passes',
            'values': [1, 3, 5]
        }
    ]

    cube = CubeCreator(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        separate_thread=False,
    )
    dummies = cube(experiment.root, dataset)
    tmodels = [dummy.restore() for dummy in dummies]
    right_parameters = [(2, 1), (2, 3), (2, 5), (3, 1), (3, 3), (3, 5),
                        (4, 1), (4, 3), (4, 5)]
    assert len(tmodels) == 9

    for i, one_model in enumerate(tmodels):
        assert len(one_model.scores.keys()) == 4, 'Some scores are missing'
        assert 'PerplexityScore' in one_model.scores.keys(), 'PerplexityScore is missing'
        assert 'SparsityPhiScore' in one_model.scores.keys(), 'SparsityPhiScore is missing'
        assert 'SparsityThetaScore' in one_model.scores.keys(), 'SparsityThetaScore is missing'
        assert 'example_score' in one_model.scores.keys(), 'example_score is missing'
        assert one_model._model.num_topics == right_parameters[i][0]
        assert one_model._model.num_document_passes == right_parameters[i][1]
        assert len(one_model.scores['PerplexityScore']) > 0, 'Smth wrong with scores'
        assert len(one_model.scores['SparsityPhiScore']) > 0, 'Smth wrong with scores'
        assert len(one_model.scores['SparsityThetaScore']) > 0, 'Smth wrong with scores'


def test_two_cubes_experiment(experiment_enviroment):
    """ """
    # experiment with two levels: first one is CubeCreator,
    # second one is RegularizersModifierCube
    tm, dataset, experiment, dictionary = experiment_enviroment

    parameters = [{
            'name': 'num_topics',
            'values': [2, 3, 4]
        }, {
            'name': 'num_document_passes',
            'values': [1, 3, 5]
        }, {
            'name': 'dictionary',
            'values': [dictionary]
    }]

    cube = CubeCreator(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        separate_thread=False,
    )

    dummies = cube(experiment.root, dataset)
    tmodels = [dummy.restore() for dummy in dummies]

    TAU_GRID = [0.1, 0.5, 1, 5, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test',
                                                                    class_ids='text'),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=False,
    )
    dummies = cube(tmodels[2], dataset)
    tmodels = [dummy.restore() for dummy in dummies]

    assert len(tmodels) == 5

    for i, one_model in enumerate(tmodels):
        assert one_model.regularizers['test'].tau == TAU_GRID[i]
        assert one_model._model.num_topics == 2
        assert one_model._model.num_document_passes == 5

    assert len(experiment.models) == 15


sleep(1)


# @pytest.mark.xfail
@pytest.mark.parametrize('thread_flag', [True, False])
def test_three_cubes_hier_model(experiment_enviroment, thread_flag):
    """ """
    # experiment with two levels: first one is CubeCreator,
    # second one is RegularizersModifierCube, third one is CubeCreator for hier model

    tm, dataset, experiment, dictionary = experiment_enviroment

    parameters = [{
            'name': 'num_topics',
            'values': [2, 3, 4]
        }, {
            'name': 'num_document_passes',
            'values': [1, 3, 5]
        }, {
            'name': 'dictionary',
            'values': [dictionary]
    }]

    cube = CubeCreator(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        separate_thread=thread_flag,
    )

    dummies = cube(experiment.root, dataset)
    tmodels_first_level = [dummy.restore() for dummy in dummies]

    TAU_GRID = [0.1, 0.5, 1, 5, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test',
                                                                    class_ids='text'),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )

    dummies = cube(tmodels_first_level[2], dataset)
    tmodels_second_level = [dummy.restore() for dummy in dummies]

    parameters = [{
            'name': 'num_topics',
            'values': [2, 3, 4]
        }, {
            'name': 'num_document_passes',
            'values': [1, 3, 5]
        }
    ]

    cube = CubeCreator(
        num_iter=10,
        second_level=True,
        parameters=parameters,
        reg_search="grid",
        separate_thread=thread_flag,
    )

    dummies = cube(tmodels_second_level[3], dataset)
    tmodels_third_level = [dummy.restore() for dummy in dummies]

    for model in tmodels_third_level:
        # TODO: Failing SparsityPhiScore
        print(model.scores)
        assert len(model.scores['PerplexityScore']) > 0, 'Smth wrong with scores'
        assert len(model.scores['SparsityPhiScore']) > 0, 'Smth wrong with scores'
        assert len(model.scores['SparsityThetaScore']) > 0, 'Smth wrong with scores'


def test_scores_are_different_after_cube(experiment_enviroment):
    tm, dataset, experiment, dictionary = experiment_enviroment

    parameters = [
        {
            'name': 'num_topics',
            'values': [2, 3, 4]
        }
    ]

    cube = CubeCreator(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        separate_thread=False,
    )

    def check_scores(tmodels):
        for i in range(len(tmodels)):
            for j in range(i + 1, len(tmodels)):
                assert (
                    tmodels[i].custom_scores['example_score'] !=
                    tmodels[j].custom_scores['example_score']
                ), 'Custom score objects are the same for different models.'
                assert (
                    tmodels[i]._model.score_tracker['PerplexityScore'] !=
                    tmodels[j]._model.score_tracker['PerplexityScore']
                ), 'ARTM score object (PerplexityScore) is the same for different models.'

    dummies = cube(experiment.root, dataset)
    tmodels = [dummy.restore() for dummy in dummies]
    check_scores(tmodels)

    TAU_GRID = [0.1, 1, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        verbose=True,
        separate_thread=False,
    )

    dummies = cube(tmodels[1], dataset)
    tmodels = [dummy.restore() for dummy in dummies]
    check_scores(tmodels)
