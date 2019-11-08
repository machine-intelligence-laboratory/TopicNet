import pytest
import warnings

import os
import shutil
from time import sleep

import artm
import numpy as np

from ..cooking_machine.cubes.controller_cube import RegularizationControllerCube, W_HALT_CONTROL
from ..cooking_machine.cubes import RegularizersModifierCube
from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1
from ..cooking_machine.model_constructor import add_standard_scores

MAIN_MODALITY = "@text"
NGRAM_MODALITY = "@ngramms"
EXTRA_MODALITY = "@str"

POSSIBLE_REGULARIZERS = [
    artm.regularizers.SmoothSparsePhiRegularizer(name='test_phi_sparse'),
    artm.regularizers.SmoothSparseThetaRegularizer(name='test_theta_sparse'),
    artm.regularizers.DecorrelatorPhiRegularizer(name='test_decor')
]


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


def approx_equal(x, y):
    return (abs(x - y) < 0.01)
# ===============================


def generate_sparse_regularizers(
        specific_topic_names, background_topic_names,
        class_ids_for_bcg_smoothing=MAIN_MODALITY,
        specific_words_classes=MAIN_MODALITY):
    """
    Creates an array of pre-configured regularizers
    using specified coefficients
    """
    gimel_smooth_specific = 3e-10
    gimel_smooth_bcg = 0.3
    regularizers = [
        artm.SmoothSparsePhiRegularizer(
            tau=gimel_smooth_specific,
            name='smooth_phi_specific',
            topic_names=specific_topic_names,
            class_ids=specific_words_classes
        ),
        artm.SmoothSparseThetaRegularizer(
            tau=gimel_smooth_specific,
            name='smooth_theta_specific',
            topic_names=specific_topic_names
        ),
        artm.SmoothSparseThetaRegularizer(
            tau=gimel_smooth_bcg,
            name='smooth_theta_background',
            topic_names=background_topic_names
        ),
        artm.SmoothSparsePhiRegularizer(
            tau=gimel_smooth_bcg,
            name='smooth_phi_background',
            topic_names=background_topic_names,
            class_ids=class_ids_for_bcg_smoothing
        ),
    ]
    return regularizers


def generate_decorrelators(
        specific_topic_names_lvl1, background_topic_names_lvl1,
        words_class_ids=MAIN_MODALITY,
        class_ids_for_bcg_decorrelation=MAIN_MODALITY,
        ngramms_modalities_for_decor=NGRAM_MODALITY):
    """
    Creates an array of pre-configured regularizers
    using specified coefficients
    """
    decorrelator_tau_ngramms = 5*1e-3
    decorrelator_tau_words_specific = 5*1e-2
    decorrelator_tau_words_bcg = 5*1e-3

    regularizers = [
        artm.DecorrelatorPhiRegularizer(
            gamma=0,
            tau=decorrelator_tau_words_specific,
            name='decorrelation',
            topic_names=specific_topic_names_lvl1,
            class_ids=words_class_ids,
        ),
        artm.DecorrelatorPhiRegularizer(
            tau=decorrelator_tau_words_bcg,
            name='decorrelation_background',
            topic_names=background_topic_names_lvl1,
            class_ids=words_class_ids,
        ),
        artm.DecorrelatorPhiRegularizer(
            tau=decorrelator_tau_ngramms,
            name='decorrelation_ngramms',
            topic_names=specific_topic_names_lvl1,
            class_ids=ngramms_modalities_for_decor
        ),
        artm.DecorrelatorPhiRegularizer(
            tau=decorrelator_tau_ngramms,
            name='decorrelation_ngramms_background',
            topic_names=background_topic_names_lvl1,
            class_ids=class_ids_for_bcg_decorrelation
        )
    ]
    return regularizers


shrink_iterations = 10
grow_iterations = 25

shrink_constant = np.power(1 / 10, 1 / shrink_iterations)
grow_constant = np.power(10, 1 / grow_iterations)
LAMBDA_STRING = "lambda initial_tau, prev_tau, cur_iter, user_value:"
DEF_STRING = "func(initial_tau, prev_tau, cur_iter, user_value):"


# ===============================


# to run all test
@pytest.fixture(scope="function")
def experiment_enviroment(request):
    """ """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
        dataset = Dataset(f'tests/test_data/test_dataset.csv')
        dictionary = dataset.get_dictionary()

    model_artm = artm.ARTM(
        num_processors=1,
        num_topics=5,
        cache_theta=True,
        class_ids={MAIN_MODALITY: 1.0, NGRAM_MODALITY: 1.0},
        num_document_passes=1,
        dictionary=dictionary,
    )

    specific_topic_names = [t for t in model_artm.topic_names if "background" not in t]
    background_topic_names = [t for t in model_artm.topic_names if "background" in t]

    sparse_regs = generate_sparse_regularizers(
        specific_topic_names, background_topic_names,
    )
    decor_regs = generate_decorrelators(
            specific_topic_names, background_topic_names,
    )
    for regularizer in sparse_regs + decor_regs:
        model_artm.regularizers.add(regularizer, overwrite=True)

    add_standard_scores(
        model_artm, dictionary,
        main_modality=MAIN_MODALITY,
        all_modalities=[MAIN_MODALITY, NGRAM_MODALITY]
    )

    tm = TopicModel(model_artm, model_id='new_id')
    experiment = Experiment(experiment_id="test_controller",
                            save_path="tests/experiments", topic_model=tm)

    return tm, dataset, experiment, dictionary


def _retrieve_tau_history(result_tm, agent):
    last_tau = result_tm.regularizers[agent.reg_name].tau
    history = list(agent.tau_history) + [last_tau]
    return history


def test_simple_experiment_with_controller(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    parameters = [
        {
            "reg_name": "decorrelation",
            "score_to_track": "PerplexityScore" + MAIN_MODALITY,
            "tau_converter": "prev_tau * user_value",
            "user_value_grid": [2]
        }, {
            "reg_name": "decorrelation_background",
            "score_to_track": "PerplexityScore" + MAIN_MODALITY,
            "tau_converter": "prev_tau + user_value",
            "user_value_grid": [0, 0.001, grow_constant]
        }, {
            "reg_name": "smooth_phi_specific",
            "score_to_track": None,  # never stop working
            "tau_converter": "prev_tau * user_value",
            "user_value_grid": [shrink_constant]
        }, {
            "reg_name": "smooth_theta_specific",
            "score_to_track": None,  # never stop working
            "tau_converter": "prev_tau * user_value",
            "user_value_grid": [shrink_constant]
        }
    ]

    cube = RegularizationControllerCube(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        use_relative_coefficients=True,
        separate_thread=False
    )

    with pytest.warns(UserWarning, match=W_HALT_CONTROL[:10]):
        cube(tm, dataset)
    tmodels = experiment.select()

    assert len(tmodels) == 3

    for result_tm in tmodels:
        for agent in result_tm.callbacks:
            history = _retrieve_tau_history(result_tm, agent)
            max_tau_value = max(history)
            safe_tau_value = max_tau_value / 2

            assert len(history) == 11
            for prev_val, val in zip(history[:-1], history[1:]):
                if "decorrelation" == agent.reg_name:
                    assert (
                        approx_equal(val, prev_val * 2) or
                        approx_equal(val, max_tau_value) or
                        approx_equal(val, safe_tau_value)
                    )
                if "smooth" in agent.reg_name:
                    assert(prev_val > val)


def test_flicker_with_controller(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    parameters = [
        {
            "reg_name": "decorrelation",
            "score_to_track": "PerplexityScore" + MAIN_MODALITY,
            "tau_converter": "initial_tau * (cur_iter % 2)",
            "user_value_grid": [0.3]
        }, {
            "reg_name": "smooth_theta_specific",
            "score_to_track": None,  # never stop working
            "tau_converter": "initial_tau * ((cur_iter + 1) % 2)",
            "user_value_grid": [0.3]
        }
    ]

    cube = RegularizationControllerCube(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        use_relative_coefficients=True,
        separate_thread=False
    )

    tmodels = [dummy.restore() for dummy in cube(tm, dataset)]

    assert len(tmodels) == 1

    for result_tm in tmodels:
        agent1 = result_tm.callbacks[0]
        history1 = _retrieve_tau_history(result_tm, agent1)[1:]
        agent2 = result_tm.callbacks[1]
        history2 = _retrieve_tau_history(result_tm, agent2)[1:]
        print(history1)
        print(history2)

        for t1, t2 in zip(history1, history2):
            assert t1 * t2 == 0
            assert t1 + t2 > 0


def test_flicker_with_controller_lambdas(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    def func(initial_tau, prev_tau, cur_iter, user_value):
        return 0 if cur_iter % 2 else initial_tau

    def very_complex_func(initial_tau, prev_tau, cur_iter, user_value):
        relu_grower = user_value * (cur_iter - 8) if cur_iter > 8 else 0
        return 0 if cur_iter % 2 else relu_grower

    parameters = [
        {
            "reg_name": "decorrelation",
            "score_to_track": "PerplexityScore" + MAIN_MODALITY,
            "tau_converter": (
                lambda initial_tau, prev_tau, cur_iter, user_value: initial_tau * (cur_iter % 2)
            ),
            "user_value_grid": [0.3]
        }, {
            "reg_name": "smooth_theta_background",
            "score_to_track": None,  # never stop working
            "tau_converter": very_complex_func,
            "user_value_grid": [0.3]
        }, {
            "reg_name": "smooth_theta_specific",
            "score_to_track": None,  # never stop working
            "tau_converter": func,
            "user_value_grid": [0.3]
        }
    ]

    cube = RegularizationControllerCube(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        use_relative_coefficients=True,
        separate_thread=False
    )

    tmodels = [dummy.restore() for dummy in cube(tm, dataset)]

    for one_cube_part in cube.get_jsonable_from_parameters():
        source = one_cube_part["tau_converter"]
        print(source)
        assert LAMBDA_STRING in source or DEF_STRING in source
        assert " % 2" in source

    assert len(tmodels) == 1

    for result_tm in tmodels:
        agent1 = result_tm.callbacks[0]
        history1 = _retrieve_tau_history(result_tm, agent1)[1:]
        agent2 = result_tm.callbacks[2]
        history2 = _retrieve_tau_history(result_tm, agent2)[1:]
        print(history1)
        print(history2)

        for t1, t2 in zip(history1, history2):
            assert t1 * t2 == 0
            assert t1 + t2 > 0


def test_description_for_insanely_complicated_lambdas(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    parameters = [
        {
            "reg_name": "decorrelation",
            "score_to_track": "PerplexityScore" + MAIN_MODALITY,
            "tau_converter": (
                lambda initial_tau, prev_tau, cur_iter, user_value:
                initial_tau * ((cur_iter + 1) % 2)
            ),
            "user_value_grid": [0.3]
        }, {
            "reg_name": "smooth_theta_specific",
            "score_to_track": None,  # never stop working
            "tau_converter": (
                lambda initial_tau, prev_tau, cur_iter, user_value:
                (user_value * (cur_iter - 8) if cur_iter > 8 else 0)
                if cur_iter % 2 == 0 else
                initial_tau
            ),
            "user_value_grid": [0.3]
        }
    ]

    cube = RegularizationControllerCube(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        use_relative_coefficients=True,
        separate_thread=False
    )

    tmodels = cube(tm, dataset)
    for one_cube_part in cube.get_jsonable_from_parameters():
        source = one_cube_part["tau_converter"]
        print(source)
        assert LAMBDA_STRING in source
        # TODO: this is known issue
        # assert " % 2" in source

    assert len(tmodels) == 1


def test_inline_relative_regularizers(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    reg = artm.regularizers.SmoothSparsePhiRegularizer(name='test', tau=0.2,
                                                       class_ids=MAIN_MODALITY)
    parameters = [
        {
            "regularizer": reg,
            "score_to_track": None,
            "tau_converter": "prev_tau",
            "user_value_grid": [0.3]
        }
    ]

    cube = RegularizationControllerCube(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        use_relative_coefficients=True,
        separate_thread=False
    )

    tmodels = [dummy.restore() for dummy in cube(tm, dataset)]

    assert len(tmodels) == 1

    for tm in tmodels:
        assert "test" in tm.regularizers._data
        assert tm.regularizers["test"].tau != 0.2, "tau was not replaced with relative coefficient"
        agent = tm.callbacks[0]
        history = _retrieve_tau_history(tm, agent)[1:]
        assert len(set(history)) == 1, "tau shouldn't been changed"


def test_inline_regularizers(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    reg = artm.regularizers.SmoothSparsePhiRegularizer(name='test', tau=1, class_ids=MAIN_MODALITY)
    parameters = [
        {
            "regularizer": reg,
            "score_to_track": None,
            "tau_converter": "prev_tau * 2",
            "user_value_grid": [0.3]
        }
    ]

    cube = RegularizationControllerCube(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=False
    )

    tmodels = [dummy.restore() for dummy in cube(tm, dataset)]

    assert len(tmodels) == 1

    for tm in tmodels:
        assert "test" in tm.regularizers._data
        assert tm.regularizers["test"].tau > 1


@pytest.mark.parametrize('num_iters', [5, float("inf")])
def test_max_iters(experiment_enviroment, num_iters):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    reg = artm.regularizers.SmoothSparsePhiRegularizer(name='test', tau=1, class_ids=MAIN_MODALITY)
    parameters = [
        {
            "regularizer": reg,
            "score_to_track": None,
            "tau_converter": "prev_tau * 2",
            "user_value_grid": [0.3],
            "max_iters": num_iters
        }
    ]

    cube = RegularizationControllerCube(
        num_iter=10,
        parameters=parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=False
    )

    cube(tm, dataset)
    resulting_models = experiment.select()
    for one_model in resulting_models:
        expected_tau = 2 ** min(num_iters, one_model._synchronizations_processed)
        assert len(one_model.callbacks) > 0
        for agent in one_model.callbacks:
            history = _retrieve_tau_history(one_model, agent)
            print(history)
            print(agent.is_working)
        assert one_model.regularizers['test'].tau == expected_tau

    TAU_GRID_LVL2 = [0.1, 0.5]

    regularizer_parameters = {
        "regularizer": artm.regularizers.SmoothSparseThetaRegularizer(name='test_second'),
        "tau_grid": TAU_GRID_LVL2
    }

    cube_second = RegularizersModifierCube(
        num_iter=10,
        regularizer_parameters=regularizer_parameters,
        reg_search="grid",
        use_relative_coefficients=False,
        separate_thread=False
    )

    dummies = cube_second(experiment.select()[0], dataset)
    tmodels_lvl3 = [dummy.restore() for dummy in dummies]

    assert len(tmodels_lvl3) == len(TAU_GRID_LVL2)
    for one_model in tmodels_lvl3:
        expected_tau = 2 ** min(num_iters, one_model._synchronizations_processed)
        assert len(one_model.callbacks) > 0
        for agent in one_model.callbacks:
            history = _retrieve_tau_history(one_model, agent)
            print(history)
            print(agent.is_working)
        assert one_model.regularizers['test'].tau == expected_tau
