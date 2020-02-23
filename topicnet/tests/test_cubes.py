import pytest
import warnings

import os
import shutil

import artm
import numpy as np

from ..cooking_machine.cubes.greedy_strategy import GreedyStrategy
from ..cooking_machine.cubes.perplexity_strategy import PerplexityStrategy

from ..cooking_machine.cubes import RegularizersModifierCube, CubeCreator
from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.models.topic_prior_regularizer import TopicPriorRegularizer
from ..cooking_machine.models.topic_prior_regularizer import TopicPriorSampledRegularizer
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1
from ..cooking_machine.rel_toolbox_lite import count_vocab_size, compute_regularizer_gimel

DATA_PATH = f'tests/test_data/test_dataset.csv'

MAIN_MODALITY = "@text"
NGRAM_MODALITY = "@ngramms"
EXTRA_MODALITY = "@str"

POSSIBLE_REGULARIZERS = [
    artm.regularizers.SmoothSparsePhiRegularizer(name='test_phi_sparse'),
    artm.regularizers.SmoothSparseThetaRegularizer(name='test_theta_sparse'),
    artm.regularizers.DecorrelatorPhiRegularizer(name='test_decor')
]
RENORMALIZE_FLAG = [False, True]
MULTIPROCESSING_FLAGS = [False]


def resource_teardown():
    """ """
    if os.path.exists("tests/experiments"):
        shutil.rmtree("tests/experiments")
        pass
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
        dataset = Dataset(DATA_PATH)
        dictionary = dataset.get_dictionary()

    model_artm = artm.ARTM(
        num_processors=1,
        num_topics=5,
        cache_theta=True,
        class_ids={MAIN_MODALITY: 1.0, NGRAM_MODALITY: 1.0},
        num_document_passes=1,
        dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore',)],
    )

    tm = TopicModel(model_artm, model_id='new_id')
    experiment = Experiment(experiment_id="test_cubes",
                            save_path="tests/experiments",
                            topic_model=tm)

    return tm, dataset, experiment, dictionary


def test_initial_tm(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    assert tm.depth == 1
    assert tm.parent_model_id is None
    assert len(tm.callbacks) == 0


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_simple_experiment(experiment_enviroment, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID = [0.1, 0.5, 1, 5, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test',
                                                                    class_ids=MAIN_MODALITY),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )
    dummies = cube(tm, dataset)

    tmodels = [dummy.restore() for dummy in dummies]

    assert len(tmodels) == len(TAU_GRID)
    for i, one_model in enumerate(tmodels):
        assert one_model.regularizers['test'].tau == TAU_GRID[i]


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_simple_experiment_pair_strategy(experiment_enviroment, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID = [0.1, 0.5, 1, 5, 10]
    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test',
                                                                    class_ids=MAIN_MODALITY),
        "tau_grid": TAU_GRID
    }

    cube_pair = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="pair",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )
    dummies = cube_pair(tm, dataset)

    for dummy in dummies:
        print(dummy._save_path)

    tmodels_pair = [dummy.restore() for dummy in dummies]
    print(dummies)

    assert len(tmodels_pair) == 5

    for i, one_model in enumerate(tmodels_pair):
        assert one_model.regularizers['test'].tau == TAU_GRID[i]


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_double_steps_experiment(experiment_enviroment, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(name='test_first',
                                                                    class_ids=MAIN_MODALITY),
        "tau_grid": [0.1, 0.5, 1, 5, 10]
    }

    cube_first = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )

    tmodels_lvl2 = cube_first(tm, dataset)

    TAU_GRID_LVL2 = [0.1, 0.5, 1, 5, 10]

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparseThetaRegularizer(name='test_second'),
        "tau_grid": TAU_GRID_LVL2
    }

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )
    dummies = cube_second(tmodels_lvl2[0], dataset)

    tmodels_lvl3 = [dummy.restore() for dummy in dummies]

    assert len(tmodels_lvl3) == len(TAU_GRID_LVL2)
    for i, one_model in enumerate(tmodels_lvl3):
        assert one_model.regularizers['test_second'].tau == TAU_GRID_LVL2[i]
        assert one_model.depth == 3


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
@pytest.mark.parametrize('artm_regularizer', POSSIBLE_REGULARIZERS)
def test_relative_coefficients(experiment_enviroment, artm_regularizer, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment
    modality_weights = tm.class_ids
    random_tau = np.random.rand() + 0.01
    TAU_GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0, ])

    regularizer_parameters = {
        "regularizer": artm_regularizer,
        "tau_grid": TAU_GRID * random_tau
    }

    cube_first = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )

    first_cube_tmodels = cube_first(tm, dataset)

    shutil.rmtree("tests/experiments")
    del(experiment)
    tm.experiment = None
    experiment = Experiment(  # noqa: F841
        experiment_id="test",
        save_path="tests/experiments",
        topic_model=tm)

    data_stats = count_vocab_size(dictionary, modality_weights)
    gimels = []
    if artm_regularizer.name == 'test_decor':
        gimels = TAU_GRID * random_tau
    else:
        for tau in TAU_GRID * random_tau:
            artm_regularizer._tau = tau
            gimels.append(compute_regularizer_gimel(
                data_stats,
                artm_regularizer,
                modality_weights,
                n_topics=len(tm.topic_names)
            ))

    regularizer_parameters = {
        "regularizer": artm_regularizer,
        "tau_grid": gimels
    }

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid",
        use_relative_coefficients=True,
        separate_thread=thread_flag
    )

    second_cube_models = cube_second(tm, dataset)

    assert len(first_cube_tmodels) == len(TAU_GRID) == len(second_cube_models)
    if artm_regularizer.name == 'test_decor':
        for one_model, second_model in zip(first_cube_tmodels, second_cube_models):
            assert one_model.scores['PerplexityScore'] != second_model.scores['PerplexityScore']
    else:
        assert np.all(gimels != TAU_GRID * random_tau)
        for one_model, second_model in zip(first_cube_tmodels, second_cube_models):
            assert one_model.scores['PerplexityScore'] == second_model.scores['PerplexityScore']


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_two_regularizers_on_step_experiment(experiment_enviroment, thread_flag):
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
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )
    dummies = cube(tm, dataset)
    tmodels_lvl2 = [dummy.restore() for dummy in dummies]

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


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_two_regularizers_on_step_experiment_pair_grid(experiment_enviroment, thread_flag):
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
        reg_search="pair",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )
    dummies = cube_pair(tm, dataset)

    tmodels_lvl2_pair = [dummy.restore() for dummy in dummies]

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


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_modifier_cube_on_two_steps_experiment(experiment_enviroment, thread_flag):
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
        reg_search="pair",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )

    tmodels_lvl2_pair = cube_pair(tm, dataset)

    regularizer_parameters_second = {
        "name": 'test_second',
        "tau_grid": [-0.2, -0.4, -0.6, -0.7]
    }

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters_second,
        reg_search="pair",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )
    dummies = cube_second(tmodels_lvl2_pair[1], dataset)
    tmodels_second = [dummy.restore() for dummy in dummies]
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


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_class_ids_cube(experiment_enviroment, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "class_ids" + MAIN_MODALITY: [1, 2, 3],
        "class_ids" + NGRAM_MODALITY: [1, 2],
    }

    cube = CubeCreator(
        num_iter=1,
        parameters=class_id_params,
        reg_search="grid",
        separate_thread=thread_flag
    )
    dummies = cube(tm, dataset)
    tmodels_lvl2 = [dummy.restore() for dummy in dummies]

    CLASS_IDS_FOR_CHECKING = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    assert len(tmodels_lvl2) == 6

    for i, one_model in enumerate(tmodels_lvl2):
        assert one_model.class_ids[MAIN_MODALITY] == CLASS_IDS_FOR_CHECKING[i][0]
        assert one_model.class_ids[NGRAM_MODALITY] == CLASS_IDS_FOR_CHECKING[i][1]


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
@pytest.mark.parametrize('renormalize', RENORMALIZE_FLAG)
def test_class_id_cube_strategy_elliptic_paraboloid(experiment_enviroment,
                                                    renormalize,
                                                    thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment
    class_id_params = {
        "class_ids" + MAIN_MODALITY: list(np.arange(0, 1.0, 0.25)),
        "class_ids" + NGRAM_MODALITY: list(np.arange(0, 2.05, 0.25)),
    }

    def retrieve_elliptic_paraboloid_score(topic_model):
        """ """
        model = topic_model._model
        return -((model.class_ids[MAIN_MODALITY]-0.6-model.class_ids[NGRAM_MODALITY]) ** 2 +
                 (model.class_ids[MAIN_MODALITY]-0.6+model.class_ids[NGRAM_MODALITY]/2) ** 2)

    cube = CubeCreator(
        num_iter=1,
        parameters=class_id_params,
        reg_search="grid",
        strategy=GreedyStrategy(renormalize),
        tracked_score_function=retrieve_elliptic_paraboloid_score,
        separate_thread=thread_flag
    )
    dummies = cube(tm, dataset)

    tmodels_lvl2 = [dummy.restore() for dummy in dummies]

    if not renormalize:
        assert len(tmodels_lvl2) == sum(len(m) for m in class_id_params.values())
    else:
        assert len(tmodels_lvl2) == 10

    if renormalize:
        CLASS_IDS_FOR_CHECKING = [(1.0, 0.0), (1.0, 0.0), (0.8, 0.2), (0.667, 0.333),
                                  (0.571, 0.429), (0.5, 0.5), (0.444, 0.556),
                                  (0.4, 0.6), (0.364, 0.636), (0.333, 0.667)]
        for i, one_model in enumerate(tmodels_lvl2):
            assert np.round(one_model.class_ids[MAIN_MODALITY], 3) == CLASS_IDS_FOR_CHECKING[i][0]
            assert np.round(one_model.class_ids[NGRAM_MODALITY], 3) == CLASS_IDS_FOR_CHECKING[i][1]
    else:
        one_model = tmodels_lvl2[len(class_id_params["class_ids" + MAIN_MODALITY])]
        assert np.round(one_model.class_ids[MAIN_MODALITY], 3) == 0.5
        assert np.round(one_model.class_ids[NGRAM_MODALITY], 3) == 0

    assert cube.strategy.best_score >= -0.09


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
@pytest.mark.parametrize('renormalize', RENORMALIZE_FLAG)
def test_class_id_cube_strategy_rosenbrock(experiment_enviroment, renormalize, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "class_ids" + MAIN_MODALITY: list(np.arange(0, 1.0, 0.25)),
        "class_ids" + NGRAM_MODALITY: list(np.arange(0, 2.05, 0.25)),
    }

    def retrieve_rosenbrock_score(topic_model):
        """ """
        model = topic_model._model
        return -((1 - model.class_ids[MAIN_MODALITY]) ** 2 +
                 100*(model.class_ids[NGRAM_MODALITY] - model.class_ids[MAIN_MODALITY]) ** 2)

    cube = CubeCreator(
        num_iter=1,
        parameters=class_id_params,
        reg_search="grid",
        strategy=GreedyStrategy(renormalize),
        tracked_score_function=retrieve_rosenbrock_score,
        separate_thread=thread_flag
    )
    dummies = cube(tm, dataset)
    tmodels_lvl2 = [dummy.restore() for dummy in dummies]

    if not renormalize:
        assert len(tmodels_lvl2) == sum(len(m) for m in class_id_params.values())
    else:
        assert len(tmodels_lvl2) == 10

    if not renormalize:
        one_model = tmodels_lvl2[len(class_id_params["class_ids" + MAIN_MODALITY])]
        assert np.round(one_model.class_ids[MAIN_MODALITY], 3) == 0.0
        assert np.round(one_model.class_ids[NGRAM_MODALITY], 3) == 0
    else:
        CLASS_IDS_FOR_CHECKING = [(1.0, 0.0), (1.0, 0.0), (0.8, 0.2), (0.667, 0.333),
                                  (0.571, 0.429), (0.5, 0.5), (0.444, 0.556),
                                  (0.4, 0.6), (0.364, 0.636), (0.333, 0.667)]
        for i, one_model in enumerate(tmodels_lvl2):
            assert np.round(one_model.class_ids[MAIN_MODALITY], 3) == CLASS_IDS_FOR_CHECKING[i][0]
            assert np.round(one_model.class_ids[NGRAM_MODALITY], 3) == CLASS_IDS_FOR_CHECKING[i][1]

    assert cube.strategy.best_score >= -0.09


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
@pytest.mark.parametrize('renormalize', RENORMALIZE_FLAG)
def test_class_id_cube_strategy_3d_parabolic(experiment_enviroment, renormalize, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    class_id_params = {
        "class_ids" + MAIN_MODALITY: list(np.arange(0, 1.0, 0.25)),
        "class_ids" + NGRAM_MODALITY: list(np.arange(0, 1.05, 0.25)),
        EXTRA_MODALITY: list(np.arange(0, 1.05, 0.25))
    }

    def retrieve_parabolic_3d_score(topic_model):
        """ """
        model = topic_model._model
        return -((1 - model.class_ids[MAIN_MODALITY]) ** 2 +
                 (1 - model.class_ids[NGRAM_MODALITY]) ** 2 +
                 (1 - model.class_ids[EXTRA_MODALITY]) ** 2)

    cube = CubeCreator(
        num_iter=1,
        parameters=class_id_params,
        reg_search="grid",
        strategy=GreedyStrategy(renormalize),
        tracked_score_function=retrieve_parabolic_3d_score,
        verbose=True,
        separate_thread=thread_flag
    )
    dummies = cube(tm, dataset)
    tmodels_lvl2 = [dummy.restore() for dummy in dummies]

    if not renormalize:
        assert len(tmodels_lvl2) == sum(len(m) for m in class_id_params.values())
    else:
        assert len(tmodels_lvl2) == 11

    if not renormalize:
        one_model = tmodels_lvl2[len(class_id_params["class_ids" + MAIN_MODALITY])]
        assert np.round(one_model.class_ids[MAIN_MODALITY], 3) == 0.75
        assert np.round(one_model.class_ids[NGRAM_MODALITY], 3) == 0
        assert np.round(one_model.class_ids[EXTRA_MODALITY], 3) == 0
    else:
        CLASS_IDS_FOR_CHECKING = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.8, 0.2, 0.0),
                                  (0.667, 0.333, 0.0), (0.571, 0.429, 0.0), (0.5, 0.5, 0.0),
                                  (0.5, 0.5, 0.0), (0.444, 0.444, 0.111), (0.4, 0.4, 0.2),
                                  (0.364, 0.364, 0.273), (0.333, 0.333, 0.333)]

        for i, one_model in enumerate(tmodels_lvl2):
            assert np.round(one_model.class_ids[MAIN_MODALITY], 3) == CLASS_IDS_FOR_CHECKING[i][0]
            assert np.round(one_model.class_ids[NGRAM_MODALITY], 3) == CLASS_IDS_FOR_CHECKING[i][1]
            assert np.round(one_model.class_ids[EXTRA_MODALITY], 3) == CLASS_IDS_FOR_CHECKING[i][2]

    if not renormalize:
        assert cube.strategy.best_point == [['', MAIN_MODALITY, 0.75],
                                            ['', NGRAM_MODALITY, 1.0],
                                            ['', EXTRA_MODALITY, 1.0]]
    else:
        assert cube.strategy.best_point == [['', MAIN_MODALITY, 0.3333333333333333],
                                            ['', NGRAM_MODALITY, 0.3333333333333333],
                                            ['', EXTRA_MODALITY, 0.3333333333333333]]


def test_topic_model_score(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    num_values = len(tm.scores['PerplexityScore'])
    assert num_iterations == num_values


def extract_visited_taus(tmodels):
    return [
        tm.regularizers["decorrelator_phi_regularizer"].tau
        for tm in tmodels
    ]


def extract_strategic_scores(cube):
    return list(map(lambda score: np.round(score, 3), cube.strategy.score))


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_perplexity_strategy_grid(experiment_enviroment, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    TAU_GRID = [0.1, 0.5, 1, 5, 50]
    regularizer_parameters = {
        "regularizer": artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer',
                                                       class_ids=MAIN_MODALITY),
        "tau_grid": TAU_GRID
    }

    cube = RegularizersModifierCube(
        num_iter=3,
        regularizer_parameters=regularizer_parameters,
        strategy=PerplexityStrategy(1, 5),
        tracked_score_function='PerplexityScore',
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=thread_flag
    )
    with pytest.warns(UserWarning, match='Grid would be used instead'):
        dummies = cube(tm, dataset)
        tmodels = [dummy.restore() for dummy in dummies]

    visited_taus = extract_visited_taus(tmodels)
    expected_taus = [0] + TAU_GRID
    assert visited_taus == expected_taus

    SCORES = [3.756, 3.756, 3.753, 3.75, 3.72, 2.887]
    real_scores = extract_strategic_scores(cube)
    if real_scores != SCORES:
        warnings.warn(f"real_scores == {real_scores}"
                      f"expected == {SCORES}")

    assert cube.strategy.best_point[0][2] == 50


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_perplexity_strategy_add(experiment_enviroment, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    regularizer_parameters = {
        "regularizer": artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer',
                                                       class_ids=MAIN_MODALITY),
        "tau_grid": []
    }

    cube = RegularizersModifierCube(
        num_iter=3,
        regularizer_parameters=regularizer_parameters,
        strategy=PerplexityStrategy(1, 1, max_len=5),
        tracked_score_function='PerplexityScore',
        reg_search='add',
        use_relative_coefficients=False,
        verbose=True,
        separate_thread=thread_flag
    )
    with pytest.warns(UserWarning, match="Max progression length"):
        dummies = cube(tm, dataset)
        tmodels = [dummy.restore() for dummy in dummies]

    visited_taus = extract_visited_taus(tmodels)
    expected_taus = [0, 1, 2, 3, 4, 5]
    assert visited_taus == expected_taus

    SCORES = [3.756, 3.75, 3.743, 3.736, 3.728, 3.72]
    real_scores = extract_strategic_scores(cube)
    if real_scores != SCORES:
        warnings.warn(f"real_scores == {real_scores}"
                      f"expected == {SCORES}")

    assert cube.strategy.best_point[0][2] == 5


@pytest.mark.parametrize('thread_flag', MULTIPROCESSING_FLAGS)
def test_perplexity_strategy_mul(experiment_enviroment, thread_flag):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    regularizer_parameters = {
        "regularizer": artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer',
                                                       class_ids=MAIN_MODALITY),
        "tau_grid": []
    }

    cube = RegularizersModifierCube(
        num_iter=20,
        regularizer_parameters=regularizer_parameters,
        strategy=PerplexityStrategy(0.001, 10, 25, threshold=1.0),
        tracked_score_function='PerplexityScore',
        reg_search='mul',
        use_relative_coefficients=False,
        verbose=True,
        separate_thread=thread_flag
    )

    with pytest.warns(UserWarning, match="Perplexity is too high for threshold"):
        dummies = cube(tm, dataset)
        tmodels = [dummy.restore() for dummy in dummies]

    visited_taus = extract_visited_taus(tmodels)
    expected_taus = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
    assert visited_taus == expected_taus

    SCORES = [3.756, 3.75, 3.72, 6.043]
    real_scores = extract_strategic_scores(cube)
    if real_scores != SCORES:
        warnings.warn(f"real_scores == {real_scores}"
                      f"expected == {SCORES}")

    assert cube.strategy.best_point[0][2] == 1.0


def test_phi_matrix_after_lda_regularizer(experiment_enviroment):
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
        dataset = Dataset(DATA_PATH)
        dictionary = dataset.get_dictionary()
        batch_vectorizer = dataset.get_batch_vectorizer()

    topic_prior_reg = TopicPriorRegularizer(
        name='topic_prior', tau=5,
        num_topics=5, beta=[10, 1, 100, 2, 1000]
    )

    model_artm_1 = artm.ARTM(
        num_processors=1,
        num_topics=5,
        cache_theta=True,
        class_ids={MAIN_MODALITY: 1.0, NGRAM_MODALITY: 1.0},
        num_document_passes=1,
        dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore',)],
    )
    model_artm_2 = artm.ARTM(
        num_processors=1,
        num_topics=5,
        cache_theta=True,
        class_ids={MAIN_MODALITY: 1.0, NGRAM_MODALITY: 1.0},
        num_document_passes=1,
        dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore',)],
    )

    tm_1 = TopicModel(
        model_artm_1, model_id='new_id_1',
        custom_regularizers={topic_prior_reg.name: topic_prior_reg}
    )
    tm_2 = TopicModel(model_artm_2, model_id='new_id_2')

    tm_1._fit(batch_vectorizer, 10)
    tm_2._fit(batch_vectorizer, 10)

    phi_first = tm_1.get_phi()
    phi_second = tm_2.get_phi()

    assert any(phi_first != phi_second), 'Phi matrices are the same after regularization.'


def test_phi_matrix_after_lda_sampled_regularizer(experiment_enviroment):
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
        dataset = Dataset(DATA_PATH)
        dictionary = dataset.get_dictionary()
        batch_vectorizer = dataset.get_batch_vectorizer()

    topic_prior_reg = TopicPriorSampledRegularizer(
        name='topic_prior', tau=5,
        num_topics=5, beta_prior=[10, 1, 100, 2, 1000]
    )

    model_artm_1 = artm.ARTM(
        num_processors=1,
        num_topics=5,
        cache_theta=True,
        class_ids={MAIN_MODALITY: 1.0, NGRAM_MODALITY: 1.0},
        num_document_passes=1,
        dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore',)],
    )
    model_artm_2 = artm.ARTM(
        num_processors=1,
        num_topics=5,
        cache_theta=True,
        class_ids={MAIN_MODALITY: 1.0, NGRAM_MODALITY: 1.0},
        num_document_passes=1,
        dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore',)],
    )

    tm_1 = TopicModel(
        model_artm_1, model_id='new_id_1',
        custom_regularizers={topic_prior_reg.name: topic_prior_reg}
    )
    tm_2 = TopicModel(model_artm_2, model_id='new_id_2')

    tm_1._fit(batch_vectorizer, 10)
    tm_2._fit(batch_vectorizer, 10)

    phi_first = tm_1.get_phi()
    phi_second = tm_2.get_phi()

    assert any(phi_first != phi_second), 'Phi matrices are the same after regularization.'
