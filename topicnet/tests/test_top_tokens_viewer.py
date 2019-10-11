import artm
import collections
import numpy as np
import shutil

import pytest
import warnings

from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1
from ..viewers import top_tokens_viewer


NUM_TOPICS = 5
TOPIC_NAMES = ['topic_{}'.format(i) for i in range(NUM_TOPICS)]
CLASS_IDS = ['@ngramms']  # TODO: @text didn't work for some reason
NUM_DOCUMENT_PASSES = 1
NUM_ITERATIONS = 10
TOKENS_SCORING_METHODS = ['top', 'phi', 'blei', 'likelihood', 'ptw']  # TODO: make tests for tfidf
NUM_TOP_TOKENS = 2


class TestTopTokensViewer:
    """ """
    topic_model = None

    @classmethod
    def setup_class(cls):
        """ """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
            dataset = Dataset('tests/test_data/test_dataset.csv')
            raw_data = []
            with open('tests/test_data/test_vw.txt') as file:
                for line in file:
                    raw_data += [line.split(' ')]
            dictionary = dataset.get_dictionary()
            batch_vectorizer = dataset.get_batch_vectorizer()

        model_artm = artm.ARTM(
            num_topics=NUM_TOPICS,
            class_ids=dict.fromkeys(CLASS_IDS, 1.0),
            topic_names=TOPIC_NAMES,
            cache_theta=True,
            num_document_passes=NUM_DOCUMENT_PASSES,
            dictionary=dictionary,
            scores=[artm.PerplexityScore(name='PerplexityScore')],)

        cls.topic_model = TopicModel(model_artm, model_id='model_id')
        cls.topic_model._fit(batch_vectorizer, num_iterations=NUM_ITERATIONS)
        cls.raw_data = raw_data

    @classmethod
    def teardown_class(cls):
        """ """
        shutil.rmtree("tests/test_data/test_dataset_batches")

    @classmethod
    def return_raw(cls):
        """ """
        return cls.raw_data

    @classmethod
    def get_top_tokens_viewer(cls, method='top', num_top_tokens=NUM_TOP_TOKENS):
        """ """
        return top_tokens_viewer.TopTokensViewer(
            model=TestTopTokensViewer.topic_model,
            class_ids=CLASS_IDS,
            method=method,
            num_top_tokens=num_top_tokens)

    @pytest.mark.parametrize("scoring_method", TOKENS_SCORING_METHODS)
    def test_check_output_format(self, scoring_method):
        """ """
        viewer = TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
        raw_data = TestTopTokensViewer.return_raw()
        topics_modalities = viewer.view(raw_data=raw_data)

        assert isinstance(topics_modalities, dict),\
            'Result of view() is of type "{}", expected "dict"'.format(type(topics_modalities))
        assert all(isinstance(modalities_tokens, dict)
                   for modalities_tokens in topics_modalities.values()),\
            'Not all values of view() result dict: "{}" -- are of type "dict"'.format(
                list(topics_modalities.values()))
        assert all(isinstance(tokens_scores, dict)
                   for modalities_tokens in topics_modalities.values()
                   for tokens_scores in modalities_tokens.values()),\
            'Expected 3-levels dict as a result. ' \
            'Not all items on the third level of view() output "{}" are dicts'.format(
                topics_modalities)

    @pytest.mark.parametrize("scoring_method", TOKENS_SCORING_METHODS)
    def test_check_number_of_topics(self, scoring_method):
        """ """
        viewer = TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
        raw_data = TestTopTokensViewer.return_raw()
        topics_modalities = viewer.view(raw_data=raw_data)
        topics_names = list(topics_modalities.keys())

        assert len(topics_names) == NUM_TOPICS,\
            'Wrong number of topics: "{}". ' \
            'Expected: "{}"'.format(len(topics_names), NUM_TOPICS)

    @pytest.mark.parametrize("scoring_method", TOKENS_SCORING_METHODS)
    def test_check_topics_names(self, scoring_method):
        """ """
        viewer = TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
        raw_data = TestTopTokensViewer.return_raw()
        topics_modalities = viewer.view(raw_data=raw_data)
        topics_names = list(topics_modalities.keys())

        assert set(topics_names) == set(TOPIC_NAMES),\
            'Wrong topic names: "{}". ' \
            'Expected: "{}"'.format(topics_names, TOPIC_NAMES)

    @pytest.mark.parametrize("scoring_method", TOKENS_SCORING_METHODS)
    def test_check_number_of_modalities(self, scoring_method):
        """ """
        viewer = TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
        raw_data = TestTopTokensViewer.return_raw()
        topics_modalities = viewer.view(raw_data=raw_data)

        for topic_name, modalities_tokens in topics_modalities.items():
            modalities_names = list(modalities_tokens.keys())
            assert len(modalities_names) == len(CLASS_IDS),\
                'Wrong number of modalities for topic "{}": "{}". ' \
                'Expected "{}"'.format(topic_name, len(modalities_names), len(CLASS_IDS))

    @pytest.mark.parametrize("scoring_method", TOKENS_SCORING_METHODS)
    def test_check_modalities_names(self, scoring_method):
        """ """
        viewer = TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
        raw_data = TestTopTokensViewer.return_raw()
        topics_modalities = viewer.view(raw_data=raw_data)

        for topic_name, modalities_tokens in topics_modalities.items():
            modalities_names = list(modalities_tokens.keys())
            assert set(modalities_names) == set(CLASS_IDS),\
                'Wrong modalities names for topic "{}": "{}". ' \
                'Expected "{}"'.format(topic_name, modalities_names, CLASS_IDS)

    @pytest.mark.parametrize("scoring_method", TOKENS_SCORING_METHODS)
    def test_check_number_of_top_tokens(self, scoring_method):
        """ """
        viewer = TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
        raw_data = TestTopTokensViewer.return_raw()
        topics_modalities = viewer.view(raw_data=raw_data)
        modalities_vocabularies = get_vocabulary_by_modality(TestTopTokensViewer.topic_model)

        for topic_name, modalities_tokens in topics_modalities.items():
            for modality_name, tokens_scores in modalities_tokens.items():
                tokens = list(tokens_scores.keys())
                assert len(tokens) == NUM_TOP_TOKENS,\
                    'Modality "{}" in topic "{}" has "{}" tokens. ' \
                    'Expected "{}"'.format(
                        modality_name, topic_name,
                        len(tokens), len(modalities_vocabularies[modality_name]))

    @pytest.mark.parametrize("scoring_method", TOKENS_SCORING_METHODS)
    def test_check_tokens_from_model_modality_dictionary(self, scoring_method):
        """ """
        viewer = TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
        raw_data = TestTopTokensViewer.return_raw()
        topics_modalities = viewer.view(raw_data=raw_data)
        modalities_vocabularies = get_vocabulary_by_modality(TestTopTokensViewer.topic_model)

        for topic_name, modalities_tokens in topics_modalities.items():
            for modality_name, tokens_scores in modalities_tokens.items():
                tokens = list(tokens_scores.keys())
                assert set(tokens) <= set(modalities_vocabularies[modality_name]), \
                    'Not all tokens of modality "{}" in topic "{}" are from the corresponding ' \
                    'modality vocabulary: "{}" vs "{}"'.format(
                        modality_name, topic_name, tokens, modalities_vocabularies[modality_name])

    @pytest.mark.parametrize("scoring_method", TOKENS_SCORING_METHODS)
    def test_check_top_tokens_ordered_by_score_in_descending_order(self, scoring_method):
        """ """
        viewer = TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
        raw_data = TestTopTokensViewer.return_raw()
        topics_modalities = viewer.view(raw_data=raw_data)

        for topic_name, modalities_tokens in topics_modalities.items():
            for modality_name, tokens_scores in modalities_tokens.items():
                scores = list(tokens_scores.values())
                assert scores == sorted(scores)[::-1], \
                    'Modality "{}" in topic "{}" has wrong order of tokens: "{}"'.format(
                        modality_name, topic_name, tokens_scores)

    # TODO: check the meaning of tokens scores (if there is any): positive/negative, range, ...

    def test_check_scoring_methods_top_and_phi_return_the_same(self):
        """ """
        viewer_phi = TestTopTokensViewer.get_top_tokens_viewer('phi')
        viewer_top = TestTopTokensViewer.get_top_tokens_viewer('top')
        topics_modalities_phi = viewer_phi.view()
        topics_modalities_top = viewer_top.view()

        assert topics_modalities_top == topics_modalities_phi,\
            'Expected the results of view() with "phi" and "top" methods to be equal, ' \
            'but they are not: "{}" vs "{}"'.format(
                topics_modalities_phi, topics_modalities_top)

    def test_check_scoring_methods_differ(self):
        """ """
        scoring_methods = list(
            set(TOKENS_SCORING_METHODS).difference(set(['top'])))  # same as "phi"
        raw_data = TestTopTokensViewer.return_raw()
        viewers = {
            scoring_method: TestTopTokensViewer.get_top_tokens_viewer(scoring_method)
            for scoring_method in scoring_methods}
        scoring_methods_topics = {
            scoring_method: viewer.view(raw_data=raw_data)
            for scoring_method, viewer in viewers.items()}

        topics_modalities = list(scoring_methods_topics.values())[0]
        topics = list(topics_modalities.keys())
        topic_whose_tokens_scores_are_to_be_compared = topics[
            np.random.choice(range(0, len(topics)))]

        modalities_tokens = topics_modalities[topic_whose_tokens_scores_are_to_be_compared]
        modalities = list(modalities_tokens.keys())
        modality_whose_tokens_scores_are_to_be_compared = modalities[
            np.random.choice(range(0, len(modalities)))]

        tokens_scores = {
            scoring_method: scoring_methods_topics[
                scoring_method][
                topic_whose_tokens_scores_are_to_be_compared][
                modality_whose_tokens_scores_are_to_be_compared]
            for scoring_method in scoring_methods}
        tokens_scores_values = np.array([
            [score for score in tokens_scores[scoring_method].values()]
            for scoring_method in scoring_methods])
        unique_tokens_scores_values = np.unique(tokens_scores_values, axis=0)

        num_scoring_methods = len(scoring_methods)
        num_unique_tokens_scores_sequences = len(unique_tokens_scores_values)

        assert num_unique_tokens_scores_sequences == num_scoring_methods,\
            'Some scoring methods "{}" gave same tokens\' values for topic "{}" ' \
            'and modality "{}": "{}". ' \
            'Unique values sequences count: "{}" -- not the same ' \
            'as the scoring methods count: "{}"'.format(
                scoring_methods,
                topic_whose_tokens_scores_are_to_be_compared,
                modality_whose_tokens_scores_are_to_be_compared,
                tokens_scores,
                num_unique_tokens_scores_sequences,
                num_scoring_methods)

    def test_check_not_possible_to_pass_wrong_scoring_method(self):
        """ """
        with pytest.raises(ValueError):
            viewer = TestTopTokensViewer.get_top_tokens_viewer(method='UNKNOWN_METHOD')
            print(viewer.view())

    def test_check_warning_if_require_more_top_tokens_than_available(self):
        """ """
        model_vocabulary_size = TestTopTokensViewer.topic_model.get_phi().shape[0]
        excessive_num_top_tokens = model_vocabulary_size + 1

        viewer = TestTopTokensViewer.get_top_tokens_viewer(
            num_top_tokens=excessive_num_top_tokens)
        raw_data = TestTopTokensViewer.return_raw()

        with pytest.warns(UserWarning):
            viewer.view(raw_data=raw_data)


def get_vocabulary_by_modality(topic_model):
    """ """
    model_vocabulary = list(topic_model.get_phi().index)  # TODO: add such method to topic_model?
    modalities_vocabularies = collections.defaultdict(list)

    for modality_name, token in model_vocabulary:
        modalities_vocabularies[modality_name].append(token)

    return dict(modalities_vocabularies)
