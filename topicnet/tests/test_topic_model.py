import datetime
import pytest
import warnings
import shutil
import time


import artm

from ..cooking_machine.models.dummy_topic_model import DummyTopicModel
from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1
from ..cooking_machine.models.example_score import ScoreExample
from ..cooking_machine.models.blei_lafferty_score import BleiLaffertyScore
from ..cooking_machine.models import BaseScore

ARTM_NINE = artm.version().split(".")[1] == "9"
MAIN_MODALITY = "@text"
NGRAM_MODALITY = "@ngramms"
EXTRA_MODALITY = "@str"


# to run all test
@pytest.fixture(scope="function")
def experiment_enviroment(request):
    """ """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
        dataset = Dataset('tests/test_data/test_dataset.csv')
        dictionary = dataset.get_dictionary()

    model_artm = artm.ARTM(
        num_topics=5,
        class_ids={MAIN_MODALITY: 1.0, NGRAM_MODALITY: 1.0, EXTRA_MODALITY: 1.0},
        num_document_passes=1, dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore', )],
        theta_columns_naming='title',
    )
    custom_scores = {'mean_kernel_size': ScoreExample()}

    tm = TopicModel(model_artm, model_id='new_id', custom_scores=custom_scores)
    experiment = Experiment(experiment_id="test", save_path="tests/experiments", topic_model=tm)

    def resource_teardown():
        """ """
        shutil.rmtree("tests/experiments")
        shutil.rmtree(dataset._internals_folder_path)

    request.addfinalizer(resource_teardown)

    return tm, dataset, experiment, dictionary


def test_topic_model_has_artm_attr(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    assert tm.num_document_passes == tm._model.num_document_passes


def test_topic_model_dont_generate_attrs(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    with pytest.raises(AttributeError):
        print(tm.psyduck_number)


def test_topic_model_score(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment

    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    num_values = len(tm.scores['PerplexityScore'])
    assert num_iterations == num_values


def test_topic_model_have_custom_score(experiment_enviroment):
    """ """
    tm, dataset, experiment, dictionary = experiment_enviroment
    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    assert len(tm.scores['mean_kernel_size']) == num_iterations


def test_topic_model_theta_is_ok(experiment_enviroment):
    tm, dataset, experiment, dictionary = experiment_enviroment
    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    with pytest.raises(ValueError, match="To get theta a dataset is required"):
        theta = tm.get_theta()
    theta = tm.get_theta(dataset=dataset)
    columns = list(theta.columns)
    correct_columns = list(dataset._data.index)
    assert sorted(columns) == sorted(correct_columns)


def test_topic_model_phi_is_ok(experiment_enviroment):
    tm, dataset, experiment, dictionary = experiment_enviroment
    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    phi = tm.get_phi().reset_index()
    columns = list(phi.columns)
    correct_columns = ['modality', 'token', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4']

    assert sorted(columns) == sorted(correct_columns)

    possible_modalities = dataset.get_possible_modalities() | {"@default_class"}
    assert all(
       mod in possible_modalities
       for mod in phi.modality.unique()
    )


def test_fancy_fit_is_ok(experiment_enviroment):
    tm, dataset, experiment, dictionary = experiment_enviroment
    model_artm = artm.ARTM(
        num_topics=5,
        num_document_passes=1, dictionary=dictionary,
        scores=[artm.PerplexityScore(name='PerplexityScore')],
        theta_columns_naming='title',
        class_ids={MAIN_MODALITY: 1, NGRAM_MODALITY: 1, EXTRA_MODALITY: 1, '@psyduck': 42},
        regularizers=[
            artm.SmoothSparseThetaRegularizer(name='smooth_theta', tau=10.0),
        ]
    )
    custom_scores = {'mean_kernel_size': ScoreExample()}

    tm = TopicModel(model_artm, model_id='absolutely_new_id', custom_scores=custom_scores)

    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    params = tm.get_jsonable_from_parameters()
    assert "smooth_theta" in params["regularizers"]
    PATH = "tests/experiments/save_standalone/"
    tm.save(PATH)
    tm2 = TopicModel.load(PATH)
    assert (tm.get_phi() == tm2.get_phi()).all().all()


def test_serialization_is_ok(experiment_enviroment):
    tm, dataset, experiment, dictionary = experiment_enviroment
    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    PATH = "tests/experiments/save/"
    tm.save(PATH)
    tm2 = TopicModel.load(PATH)
    assert (tm.get_phi() == tm2.get_phi()).all().all()


def test_topic_model_fancy_phi_are_ok(experiment_enviroment):
    tm, dataset, experiment, dictionary = experiment_enviroment
    num_iterations = 10
    tm._fit(dataset.get_batch_vectorizer(), num_iterations)
    if ARTM_NINE:
        tm.get_phi_sparse()
        tm.get_phi_dense()


@pytest.mark.parametrize("my_kwargs,exception_expected,error_msg", [
    ({}, ValueError,
     "Either num_topics or topic_names parameter should be set"),
    ({'num_topics': 10, 'nyarly': "Black Pharaoh"},
     TypeError, "got an unexpected keyword argument"),
    ({'num_topics': 10, 'class_ids': {"@word": "big"}},
     TypeError, " but expected one of: int, long, float"),
    ({'num_topics': 10, 'theta_name': ["lol", "lmao"]},
     TypeError, "has type list, but expected one"),
    ({'num_topics': 10, 'dictionary': "invalid dictionary name lol"},
     ValueError, "ARTM failed with following: InvalidOperationException")
])
# we allow model to deal with invalid entry parameters for now
@pytest.mark.xfail
def test_tm_with_bad_kwargs(my_kwargs, exception_expected, error_msg):
    with pytest.raises(exception_expected, match=error_msg):
        _ = TopicModel(**my_kwargs)


def test_tm_with_blei_laff_score(experiment_enviroment):
    tm, dataset, experiment, dictionary = experiment_enviroment
    tm.custom_scores['blei'] = BleiLaffertyScore()
    num_iter = 3
    tm._fit(dataset.get_batch_vectorizer(), num_iterations=num_iter)
    assert len(tm.scores['blei']) == num_iter
    assert BleiLaffertyScore().call(tm) != 0.0


def test_scores_add(experiment_enviroment):
    topic_model, dataset, experiment, dictionary = experiment_enviroment

    custom_score_name = 'blei'
    topic_model.scores.add(BleiLaffertyScore(name=custom_score_name))

    artm_score_name = 'perp'
    topic_model.scores.add(artm.scores.PerplexityScore(name=artm_score_name))

    num_iterations = 3
    topic_model._fit(dataset.get_batch_vectorizer(), num_iterations=num_iterations)

    assert len(topic_model.scores[custom_score_name]) == num_iterations
    assert len(topic_model.scores[artm_score_name]) == num_iterations


def test_to_dummy_and_back_with_scores(experiment_enviroment):
    topic_model, dataset, experiment, dictionary = experiment_enviroment

    custom_score_name = 'blei'
    topic_model.scores.add(BleiLaffertyScore(name=custom_score_name))

    artm_score_name = 'perp'
    topic_model.scores.add(artm.scores.PerplexityScore(name=artm_score_name))

    num_iterations = 3
    topic_model._fit(dataset.get_batch_vectorizer(), num_iterations=num_iterations)

    topic_model.save()
    save_path = topic_model.model_default_save_path

    del topic_model

    dummy = DummyTopicModel.load(save_path)

    # Dummy model keeps all score values
    assert len(dummy.scores[custom_score_name]) == num_iterations
    assert len(dummy.scores[artm_score_name]) == num_iterations
    assert not hasattr(dummy.scores, '_score_caches')

    restored_topic_model = dummy.restore(dataset)

    assert hasattr(restored_topic_model, '_scores_wrapper')
    assert hasattr(restored_topic_model._scores_wrapper, '_score_caches')
    assert restored_topic_model._scores_wrapper._score_caches is not None

    assert len(restored_topic_model.scores[custom_score_name]) == num_iterations
    assert len(restored_topic_model.scores[artm_score_name]) == num_iterations
    assert restored_topic_model.scores[custom_score_name] == dummy.scores[custom_score_name]
    assert restored_topic_model.scores[artm_score_name] == dummy.scores[artm_score_name]


@pytest.mark.parametrize(
    'should_compute',
    [False, True, None, lambda i: False, lambda i: True, lambda i: i == 2]
)
def test_should_compute(experiment_enviroment, should_compute):
    topic_model, dataset, experiment, dictionary = experiment_enviroment

    score_name = 'blei'
    topic_model.scores.add(
        BleiLaffertyScore(
            name=score_name,
            should_compute=should_compute,
        )
    )
    score_should_compute = topic_model.custom_scores[score_name]._should_compute

    num_iters = 20
    last_iter = num_iters - 1

    score_num_iters = sum(
        1 * [score_should_compute(i) for i in range(num_iters)]
    )

    if not score_should_compute(last_iter):
        score_num_iters = score_num_iters + 1

    topic_model._fit(dataset.get_batch_vectorizer(), num_iterations=num_iters)
    model_scores = topic_model.scores

    assert len(model_scores[score_name]) == score_num_iters


def test_compute_on_custom_iterations(experiment_enviroment):
    topic_model, dataset, experiment, dictionary = experiment_enviroment

    score_name_a = 'blei'
    score_should_compute_a = lambda iter: iter == 5  # noqa E731
    topic_model.scores.add(
        BleiLaffertyScore(
            name=score_name_a,
            should_compute=score_should_compute_a,
        )
    )

    score_name_b = 'perp'
    score_should_compute_b = lambda iter: iter % 2 == 0  # noqa E731
    topic_model.scores.add(
        BleiLaffertyScore(
            name=score_name_b,
            should_compute=score_should_compute_b,
        )
    )

    num_iters = 20
    last_iter = num_iters - 1

    score_num_iters_a = sum(
        1 * [score_should_compute_a(i) for i in range(num_iters)]
    )
    score_num_iters_b = sum(
        1 * [score_should_compute_b(i) for i in range(num_iters)]
    )

    if not score_should_compute_a(last_iter):
        score_num_iters_a = score_num_iters_a + 1
    if not score_should_compute_b(last_iter):
        score_num_iters_b = score_num_iters_b + 1

    topic_model._fit(dataset.get_batch_vectorizer(), num_iterations=num_iters)
    model_scores = topic_model.scores

    assert len(model_scores[score_name_a]) == score_num_iters_a
    assert len(model_scores[score_name_b]) == score_num_iters_b


def test_precomputed(experiment_enviroment):
    topic_model, dataset, experiment, dictionary = experiment_enviroment

    class SlowScore(BaseScore):
        def __init__(self, name):
            super().__init__(name=name)

            self._data_key = 'some_precomputed_data_key'

        def call(self, model: TopicModel, **kwargs):
            precomputed_data = kwargs.get(
                BaseScore._PRECOMPUTED_DATA_PARAMETER_NAME, dict()
            )

            if self._data_key in precomputed_data:
                return 0

            time.sleep(1)

            precomputed_data[self._data_key] = 0

            return 1

    topic_model.scores.add(SlowScore(name='slow_score'))

    num_iters = 5
    start_time = datetime.datetime.now()

    topic_model._fit(dataset.get_batch_vectorizer(), num_iterations=num_iters)

    middle_time = datetime.datetime.now()
    time_for_one_score = (middle_time - start_time).total_seconds()

    topic_model.scores.add(SlowScore(name='score_b'))
    topic_model._fit(dataset.get_batch_vectorizer(), num_iterations=num_iters)

    end_time = datetime.datetime.now()
    time_for_two_scores = (end_time - middle_time).total_seconds()

    assert time_for_two_scores / time_for_one_score < 1.1


def test_score_with_no_precomputed_for_compatibility(experiment_enviroment):
    topic_model, dataset, experiment, dictionary = experiment_enviroment

    class ScoreWithNoKwargs(BaseScore):
        return_value = 3

        def __init__(self, name):
            super().__init__(name=name)

        def call(self, model: TopicModel):
            time.sleep(0.01)

            return self.return_value

    score_name = 'score_without_precomputed'
    topic_model.scores.add(ScoreWithNoKwargs(name=score_name))

    num_iters = 5
    topic_model._fit(dataset.get_batch_vectorizer(), num_iterations=num_iters)

    assert len(topic_model.scores[score_name]) == num_iters
    assert all(v == ScoreWithNoKwargs.return_value for v in topic_model.scores[score_name])
