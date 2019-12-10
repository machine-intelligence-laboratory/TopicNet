import pytest
import warnings

import os
import shutil
from time import sleep

import artm

from ..cooking_machine.cubes import RegularizersModifierCube
from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.models.example_score import ScoreExample
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1
from ..cooking_machine.config_parser import build_experiment_environment_from_yaml_config


def resource_teardown():
    """ """
    if os.path.exists("tests/experiments"):
        shutil.rmtree("tests/experiments")
    if os.path.exists("tests/test_data/test_dataset_batches"):
        shutil.rmtree("tests/test_data/test_dataset_batches")


def setup_function():
    sleep(1)
    resource_teardown()
    sleep(1)


def teardown_function():
    sleep(1)
    resource_teardown()
    sleep(1)


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
        scores=[artm.PerplexityScore(name='PerplexityScore')]
    )
    model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))

    ex_score = ScoreExample()
    tm = TopicModel(model_artm, model_id='new_id', custom_scores={'example_score': ex_score})

    experiment = Experiment(tm, experiment_id="test", save_path="tests/experiments")
    cube_settings = [{
        'CubeCreator':
        {
            'num_iter': 10,
            'parameters': [
                {
                    'name': 'seed',
                    'values': [82019, 322],
                },
            ],
            'reg_search': 'grid',
        },
        'selection': [
            'model.seed = 82019 and PerplexityScore -> min COLLECT 2',
        ]
    }, {
        'RegularizersModifierCube': {
            'num_iter': 10,
            'regularizer_parameters':
            {
                "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(),
                "tau_grid": [0.1, 0.5, 1, 5, 10]
            },
            'reg_search': 'grid',
            'use_relative_coefficients': False,
        },
        'selection': [
            'PerplexityScore -> max COLLECT 2',
        ]
    }]

    return tm, dataset, experiment, dictionary, cube_settings


def test_bad_empty_config(experiment_enviroment):
    with open("tests/test_data/bad_empty_config.yml", "r") as f:
        yaml_string = f.read()

    experiment, dataset = build_experiment_environment_from_yaml_config(
        yaml_string,
        experiment_id="Test_config",
        save_path="tests/experiments"
    )
    with pytest.warns(UserWarning, match='Unable to calculate special'):
        experiment.run(dataset)
    final_models = experiment.select()
    assert len(final_models) == 0


def test_bad_config(experiment_enviroment):
    with open("tests/test_data/bad_config.yml", "r") as f:
        yaml_string = f.read()

    with pytest.raises(ValueError, match='Unsupported stages value: SecondStageCube at line 53'):
        experiment, dataset = build_experiment_environment_from_yaml_config(
            yaml_string,
            experiment_id="Test_config",
            save_path="tests/experiments"
        )


def test_pipeline_from_config(experiment_enviroment):
    with open("tests/test_data/config.yml", "r") as f:
        yaml_string = f.read()

    experiment, dataset = build_experiment_environment_from_yaml_config(
        yaml_string,
        experiment_id="Test_config",
        save_path="tests/experiments"
    )
    with pytest.warns(UserWarning, match="Max progression length") as record:
        experiment.run(dataset)

    # check that only expected warnings were raised
    # due to models saving does not supported now (models files are open)
    for entry in record:
        msg = entry.message.args[0]
        known_exceptions = [
            "unclosed file", "Max progression length", "PyUnicode_AsEncodedObject() is deprecated"
        ]
        is_msg_expected = any(msg.startswith(warning_string) for warning_string in known_exceptions)
        if not is_msg_expected:
            assert msg is None

    final_models = experiment.select()
    assert len(final_models) > 0

    description = experiment.describe_model(final_models[0].model_id)
    for tm in experiment.models.values():
        depth = tm.depth - 1
        num_iters = len(tm.scores.get("PerplexityScore@all"))
        assert(depth * 5 == num_iters)

    print('depth: ', final_models[0].depth)
    print('criteria: ', experiment.criteria[final_models[0].depth - 1])
    print('len criteria : ', len(experiment.criteria))

    assert max(tm.depth for tm in experiment.models.values()) == len(experiment.cubes)
    assert max(tm.depth for tm in experiment.models.values()) == experiment.depth
    assert "seed" in description
    assert "PerplexityScore@all" in description
    assert "SparsityPhiScore" in description

    print('depth: ', final_models[0].depth)
    print('criteria: ', experiment.criteria[final_models[0].depth - 1])
    print("-----------")
    print(experiment.criteria)
    print(description)

    assert "PerplexityScore@all" in description


def test_config_with_blei_score(experiment_enviroment):
    with open("tests/test_data/config_blei.yml", "r") as f:
        yaml_string = f.read()

    experiment, dataset = build_experiment_environment_from_yaml_config(
        yaml_string,
        experiment_id="Test_config",
        save_path="tests/experiments"
    )
    sleep(2)
    experiment.run(dataset)

    final_models = experiment.select()
    assert len(final_models) > 0
    assert "BleiLaffertyScore" in final_models[0].scores
    assert final_models[0].scores["BleiLaffertyScore"][-1] > 0


def test_config_with_scores(experiment_enviroment):
    with open("tests/test_data/config_short.yml", "r") as f:
        yaml_string = f.read()

    experiment, dataset = build_experiment_environment_from_yaml_config(
        yaml_string,
        experiment_id="Test_config",
        save_path="tests/experiments"
    )
    experiment.run(dataset)

    final_models = experiment.select()
    assert len(final_models) > 0
    assert len(final_models[0].topic_names) == 5
    assert "ScoreExample" in final_models[0].scores
    assert "BTRS" in final_models[0].scores


def test_config_with_greedy_strategy(experiment_enviroment):
    sleep(1)
    with open("tests/test_data/config2.yml", "r") as f:
        yaml_string = f.read()

    experiment, dataset = build_experiment_environment_from_yaml_config(
        yaml_string,
        experiment_id="Test_config",
        save_path="tests/experiments"
    )
    sleep(1)
    experiment.run(dataset)

    final_models = experiment.select()
    assert len(final_models) > 0


def test_simple_pipeline(experiment_enviroment):
    tm, dataset, experiment, dictionary, cube_settings = experiment_enviroment

    experiment.build(cube_settings)
    with pytest.warns(UserWarning, match="Not enough models for"):
        final_models = experiment.run(dataset, verbose=False, nb_verbose=False)

    assert len(experiment.cubes) == 3, 'Incorrect number of cubes in the experiment.'
    assert len(experiment.criteria) == 3, 'Incorrect number of criteria in the experiment.'
    assert len(experiment.get_models_by_depth(level=2)) == 2, \
        'Incorrect number of models on the first level.'
    assert len(experiment.get_models_by_depth(level=3)) == 5, \
        'Incorrect number of models on the second level.'

    scores = [tm.scores["PerplexityScore"][-1] for tm in final_models]
    assert len(final_models) == 2, 'Incorrect number of final models.'
    assert len(set(scores)) == 2, 'Incorrect number of final models.'


def test_pipeline_with_new_cube_after(experiment_enviroment):
    tm, dataset, experiment, dictionary, cube_settings = experiment_enviroment

    experiment.build(cube_settings)
    # an ugly workaround for strange multithreading issue
    sleep(2)
    with pytest.warns(UserWarning, match="Not enough models for"):
        models = experiment.run(dataset, verbose=False, nb_verbose=False)
    models = list(models)

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters={
            "regularizer": artm.regularizers.SmoothSparseThetaRegularizer(name='second'),
            "tau_grid": [0.1, 0.5, 1]
        },
        reg_search="grid",
        use_relative_coefficients=False,
    )

    new_models = cube(models[-1], dataset)
    assert len(new_models) == 3, 'Incorrect number of final models.'
    assert len(experiment.cubes) == 4, 'Incorrect number of cubes in the experiment.'
    assert len(experiment.criteria) == 4, 'Incorrect number of criteria in the experiment.'
