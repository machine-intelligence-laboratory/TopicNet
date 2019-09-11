import pytest

import shutil

import artm

from ..cooking_machine.cubes import RegularizersModifierCube
from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.models.example_score import ScoreExample
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset


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
    model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))

    ex_score = ScoreExample()
    tm = TopicModel(model_artm, model_id='new_id', custom_scores={'example_score': ex_score})

    experiment = Experiment(experiment_id="test", save_path="tests/experiments")
    tm.experiment = experiment
    cube_settings = [{
        'CubeCreator':
        {
            'model': tm,
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
        },
        'selection': [
            'PerplexityScore -> max COLLECT 2',
        ]
    }]

    def resource_teardown():
        """ """
        shutil.rmtree("tests/experiments")
        shutil.rmtree("tests/test_data/test_dataset_batches")

    request.addfinalizer(resource_teardown)

    return tm, dataset, experiment, dictionary, cube_settings


def test_simple_pipeline(experiment_enviroment):
    tm, dataset, experiment, dictionary, cube_settings = experiment_enviroment

    experiment.build(cube_settings)
    final_models = experiment.run(dataset, verbose=False, nb_verbose=False)

    assert len(final_models) == 2, 'Incorrect number of final models.'
    assert len(experiment.cubes) == 3, 'Incorrect number of cubes in the experiment.'
    assert len(experiment.criteria) == 3, 'Incorrect number of criteria in the experiment.'
    assert len(experiment.get_models_by_depth(level=1)) == 2, \
        'Incorrect number of models on the first level.'
    assert len(experiment.get_models_by_depth(level=2)) == 5, \
        'Incorrect number of models on the second level.'


def test_pipeline_with_new_cube_after(experiment_enviroment):
    tm, dataset, experiment, dictionary, cube_settings = experiment_enviroment

    experiment.build(cube_settings)
    models = experiment.run(dataset, verbose=False, nb_verbose=False)
    models = list(models)

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters={
            "regularizer": artm.regularizers.SmoothSparseThetaRegularizer(name='second'),
            "tau_grid": [0.1, 0.5, 1]
        },
        reg_search="grid"
    )

    new_models = cube(models[-1], dataset)
    assert len(new_models) == 3, 'Incorrect number of final models.'
    assert len(experiment.cubes) == 4, 'Incorrect number of cubes in the experiment.'
    assert len(experiment.criteria) == 4, 'Incorrect number of criteria in the experiment.'
