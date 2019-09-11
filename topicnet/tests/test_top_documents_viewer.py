import artm
import numpy as np
import shutil

import pytest

from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.dataset import Dataset
from ..viewers import top_documents_viewer


NUM_TOPICS = 5
NUM_DOCUMENT_PASSES = 1
NUM_ITERATIONS = 10


class TestTopDocumentsViewer:
    """ """
    topic_model = None
    theta = None
    top_documents_viewer = None

    @classmethod
    def setup_class(cls):
        """ """
        dataset = Dataset('tests/test_data/test_dataset.csv')
        dictionary = dataset.get_dictionary()
        batch_vectorizer = dataset.get_batch_vectorizer()

        model_artm = artm.ARTM(
            num_topics=NUM_TOPICS,
            cache_theta=True,
            num_document_passes=NUM_DOCUMENT_PASSES,
            dictionary=dictionary,
            scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary)],)

        cls.topic_model = TopicModel(model_artm, model_id='model_id')
        cls.topic_model._fit(batch_vectorizer, num_iterations=NUM_ITERATIONS)
        cls.theta = cls.topic_model.get_theta(dataset=dataset)

        cls.top_documents_viewer = top_documents_viewer.TopDocumentsViewer(model=cls.topic_model)

    @classmethod
    def teardown_class(cls):
        """ """
        shutil.rmtree("tests/test_data/test_dataset_batches")

    def test_check_output_format(self):
        """ """
        topics_documents = TestTopDocumentsViewer.top_documents_viewer.view()

        assert isinstance(topics_documents, list), 'Result of view() not of type "list"'
        assert all(isinstance(topic_documents, list) for topic_documents in topics_documents),\
            'Some elements in the result list of view() not of type "list"'

    def test_check_output_content(self):
        """ """
        num_documents = TestTopDocumentsViewer.theta.shape[1]
        documents_indices = list(range(num_documents))

        topics_documents_from_viewer = TestTopDocumentsViewer.top_documents_viewer.view()
        documents_from_viewer = merge_lists(topics_documents_from_viewer)

        assert sorted(documents_from_viewer) == documents_indices,\
            'Viewer returned as documents "{0}".' \
            'But expected to get documents\' indices from "0" to "{1}"'.format(
                documents_from_viewer, num_documents - 1)

    def test_check_precomputed_distances_parameter_workable(self):
        """ """
        index_of_topic_to_be_nearest_to_all_documents = 0

        distances_all_one_except_to_one_topic = np.ones_like(TestTopDocumentsViewer.theta.values)
        distances_all_one_except_to_one_topic[:, index_of_topic_to_be_nearest_to_all_documents] = 0
        documents_viewer = top_documents_viewer.TopDocumentsViewer(
            model=TestTopDocumentsViewer.topic_model,
            precomputed_distances=distances_all_one_except_to_one_topic)

        topics_documents = documents_viewer.view()
        num_documents_in_nearest_topic = len(
            topics_documents[index_of_topic_to_be_nearest_to_all_documents])
        num_documents = TestTopDocumentsViewer.theta.shape[1]

        assert num_documents_in_nearest_topic == num_documents,\
            'Expected to see all documents in one topic.' \
            'But the topic has "{}" documents instead of "{}"'.format(
                num_documents_in_nearest_topic, num_documents)

    @pytest.mark.parametrize("max_num_top_documents", [0, 1])
    def test_check_max_top_documents_number_parameter_workable(self, max_num_top_documents):
        """ """
        documents_viewer = top_documents_viewer.TopDocumentsViewer(
            model=TestTopDocumentsViewer.topic_model,
            max_top_number=max_num_top_documents)

        topics_documents = documents_viewer.view()

        assert all(len(topic_documents) <= max_num_top_documents
                   for topic_documents in topics_documents),\
            'Not all top documents lists from "{}" have less elements than required "{}"'.format(
                topics_documents, max_num_top_documents)

    def test_check_object_clusters_parameter_workable(self):
        """ """
        num_documents = TestTopDocumentsViewer.theta.shape[1]
        cluster_label_to_be_same_for_all_documents = 0
        cluster_labels = list(
            cluster_label_to_be_same_for_all_documents for _ in range(num_documents))

        documents_viewer = top_documents_viewer.TopDocumentsViewer(
            model=TestTopDocumentsViewer.topic_model,
            object_clusters=cluster_labels)

        topics_documents = documents_viewer.view()
        num_documents_with_given_cluster_label = len(
            topics_documents[cluster_label_to_be_same_for_all_documents])

        assert num_documents_with_given_cluster_label == num_documents,\
            'Marked all documents with label "{}".' \
            'Expected to see all "{}" documents in that topic,' \
            'but there are only "{}" documents'.format(
                cluster_label_to_be_same_for_all_documents, num_documents,
                num_documents_with_given_cluster_label)

    @pytest.mark.parametrize("illegal_cluster_label", [-1, NUM_TOPICS])
    def test_check_object_clusters_parameter_validates_range_of_input_labels(
            self, illegal_cluster_label):
        """ """
        num_documents = TestTopDocumentsViewer.theta.shape[1]
        cluster_labels = list(0 for _ in range(num_documents))

        cluster_labels[0] = illegal_cluster_label

        with pytest.raises(ValueError):
            _ = top_documents_viewer.TopDocumentsViewer(
                model=TestTopDocumentsViewer.topic_model,
                object_clusters=cluster_labels).view()


def merge_lists(iterable_of_lists):
    """ """
    result = []

    for i in iterable_of_lists:
        result += i

    return result
