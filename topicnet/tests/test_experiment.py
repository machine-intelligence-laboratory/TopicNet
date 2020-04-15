import pytest
import warnings

import os
import shutil
import numpy as np

import artm

from ..cooking_machine.cubes import RegularizersModifierCube

from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.experiment import Experiment, START
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1


MAIN_MODALITY = "@text"
NGRAM_MODALITY = "@ngramms"
EXTRA_MODALITY = "@str"
MULTIPROCESSING_FLAGS = [True, False]


def resource_teardown():
    """ """
    dataset = Dataset('tests/test_data/test_dataset.csv')

    if os.path.exists("tests/experiments"):
        shutil.rmtree("tests/experiments")
    if os.path.exists(dataset._internals_folder_path):
        shutil.rmtree(dataset._internals_folder_path)


def setup_function():
    resource_teardown()


def teardown_function():
    resource_teardown()

# to run all test
@pytest.fixture(scope="function")
def two_experiment_enviroments(request):
    """ """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
        dataset = Dataset('tests/test_data/test_dataset.csv')
        dictionary = dataset.get_dictionary()

    model_artm_1 = artm.ARTM(
        num_processors=1,
        num_topics=5, cache_theta=True,
        num_document_passes=1, dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore'),
                artm.SparsityPhiScore(name='SparsityPhiScore', class_id=MAIN_MODALITY)]
    )

    model_artm_2 = artm.ARTM(
        num_processors=1,
        num_topics=5, cache_theta=True,
        num_document_passes=1, dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore'),
                artm.SparsityPhiScore(name='SparsityPhiScore', class_id=MAIN_MODALITY)]
    )

    tm_1 = TopicModel(model_artm_1, model_id='new_id_1')
    tm_2 = TopicModel(model_artm_2, model_id='new_id_2')

    experiment_1 = Experiment(
        experiment_id="test_1", save_path="tests/experiments", topic_model=tm_1
    )
    experiment_2 = Experiment(
        experiment_id="test_2", save_path="tests/experiments", topic_model=tm_2
    )

    return tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary


def test_initial_save_load(two_experiment_enviroments):
    """ """
    print(os.getcwd())
    tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary = two_experiment_enviroments

    experiment_1.set_criteria(0, 'test_criteria')
    experiment_1.save_path = 'tests/experiments/test'
    experiment_1.save()
    experiment = Experiment.load('tests/experiments/test/test_1')
    tm_id = START
    tm_load = experiment.models[tm_id]
    tm_save = experiment_1.models[tm_id]

    assert tm_save.depth == tm_load.depth
    assert tm_save.parent_model_id == tm_load.parent_model_id
    assert experiment.criteria[0] == ['test_criteria'], 'Experiment failed to load criteria'


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_simple_experiment(two_experiment_enviroments, thread_flag):
    """ """
    tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary = two_experiment_enviroments

    experiment_1.save_path = 'tests/experiments/test'
    experiment_1.save()
    experiment = Experiment.load('tests/experiments/test/test_1')
    tm_id = START
    tm = experiment.models[tm_id]

    TAU_GRID = [0.1, 0.5, 1, 5, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test',
                                                                    class_ids=MAIN_MODALITY),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )

    cube_2 = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )
    dummies = cube(tm, dataset)
    tmodels = [dummy.restore() for dummy in dummies]
    dummies_2 = cube_2(tm_2, dataset)
    tmodels_2 = [dummy.restore() for dummy in dummies_2]

    assert len(tmodels) == len(tmodels_2)
    assert cube.strategy.score == cube_2.strategy.score

    for models in zip(tmodels, tmodels_2):
        assert models[0].regularizers['test'].tau == models[1].regularizers['test'].tau
        assert np.array_equal(models[0].get_phi(), models[1].get_phi())
        assert models[0].depth == 2
        assert models[1].depth == 2


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_double_steps_experiment(two_experiment_enviroments, thread_flag):
    """ """
    tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary = two_experiment_enviroments

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test_first',
                                                                    class_ids=MAIN_MODALITY),
        "tau_grid": [0.1, 0.5, 1, 5, 10]
    }

    cube_first_1 = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )

    cube_first_2 = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )
    dummies_first = cube_first_1(tm_1, dataset)
    tmodels_lvl2_1 = [dummy.restore() for dummy in dummies_first]
    dummies_second = cube_first_2(tm_2, dataset)
    tmodels_lvl2_2 = [dummy.restore() for dummy in dummies_second]

    experiment_1.save_path = 'tests/experiments/test'
    experiment_1.save()
    experiment = Experiment.load('tests/experiments/test_1')
    print('original experiment ', experiment_1.models.items())
    print('loaded experiment ', experiment.models.items())
    tmodels_lvl2 = experiment.get_models_by_depth(2)

    TAU_GRID_LVL2 = [0.1, 0.5, 1]

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparseThetaRegularizer(name='test_second'),
        "tau_grid": TAU_GRID_LVL2
    }

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )

    cube_second_2 = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )
    dummies_lvl3 = cube_second(tmodels_lvl2[0], dataset)
    tmodels_lvl3 = [dummy.restore() for dummy in dummies_lvl3]
    dummies_lvl3_2 = cube_second_2(tmodels_lvl2_2[0], dataset)
    tmodels_lvl3_2 = [dummy.restore() for dummy in dummies_lvl3_2]

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


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_describe(two_experiment_enviroments, thread_flag):
    """ """
    tm_1, experiment_1, tm_2, experiment_2, dataset, dictionary = two_experiment_enviroments

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test_first',
                                                                    class_ids=MAIN_MODALITY),
        "tau_grid": [0.1, 0.5, 1, 5, 10]
    }

    cube_first = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )

    _ = cube_first(tm_1, dataset)
    criterion = "PerplexityScore -> min COLLECT 2"
    tmodels_lvl1 = experiment_1.select(criterion)
    experiment_1.set_criteria(1, [criterion])

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        use_relative_coefficients=False,
        reg_search="grid",
        separate_thread=thread_flag,
    )

    _ = cube_second(tmodels_lvl1, dataset)
    criterion = "SparsityPhiScore -> max COLLECT 1"
    final_models = experiment_1.select(criterion)
    experiment_1.set_criteria(2, [criterion])

    final_model_name = final_models[0].model_id
    second_model_name = tmodels_lvl1[0].model_id
    assert "SparsityPhiScore" in experiment_1.describe_model(final_model_name)
    assert "PerplexityScore" in experiment_1.describe_model(second_model_name)
