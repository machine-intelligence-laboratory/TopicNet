import artm
import numpy as np
import shutil

import pytest
import warnings

from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1
from ..viewers import top_documents_viewer


NUM_TOPICS = 5
NUM_DOCUMENT_PASSES = 1
NUM_ITERATIONS = 10


class TestTopDocumentsViewer:
    """ """
    topic_model = None
    theta = None
    top_documents_viewer = None
    dataset = None

    @classmethod
    def setup_class(cls):
        """ """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
            cls.dataset = Dataset('tests/test_data/test_dataset.csv')
            dictionary = cls.dataset.get_dictionary()
            batch_vectorizer = cls.dataset.get_batch_vectorizer()

        model_artm = artm.ARTM(
            num_topics=NUM_TOPICS,
            cache_theta=True,
            num_document_passes=NUM_DOCUMENT_PASSES,
            dictionary=dictionary,
            scores=[artm.PerplexityScore(name='PerplexityScore')],)

        cls.topic_model = TopicModel(model_artm, model_id='model_id')
        cls.topic_model._fit(batch_vectorizer, num_iterations=NUM_ITERATIONS)
        cls.theta = cls.topic_model.get_theta(dataset=cls.dataset)

        cls.top_documents_viewer = top_documents_viewer.TopDocumentsViewer(model=cls.topic_model)

    @classmethod
    def teardown_class(cls):
        """ """
        shutil.rmtree(cls.dataset._internals_folder_path)

    def test_check_output_format(self):
        """ """
        viewer_output = TestTopDocumentsViewer.top_documents_viewer.view()
        list_of_topics = list(viewer_output.keys())

        assert isinstance(viewer_output, dict), 'Result of view() not of type "list"'
        assert all(isinstance(viewer_output[topic], dict) for topic in list_of_topics),\
            'Some elements in the result list of view() not of type "list"'

    def test_check_output_content(self):
        """ """
        num_documents = TestTopDocumentsViewer.theta.shape[1]
        documents_indices = list(range(num_documents))

        viewer_output = TestTopDocumentsViewer.top_documents_viewer.view()
        documents_from_viewer = list(
            (viewer_output[key].keys())
            for key in viewer_output.keys()
        )
        flattened_output = [
            doc_id for topic_docs in documents_from_viewer
            for doc_id in topic_docs
        ]
        assert sorted(flattened_output) == documents_indices,\
            'Viewer returned as documents "{0}".' \
            'But expected to get documents\' indices from "0" to "{1}"'.format(
                flattened_output, num_documents - 1)

    def test_check_precomputed_distances_parameter_workable(self):
        """ """
        index_of_topic_to_be_nearest_to_all_documents = 0
        name_of_topic_to_be_nearest_to_all_document = 'topic_0' 

        distances_all_one_except_to_one_topic = np.ones_like(TestTopDocumentsViewer.theta.values)
        distances_all_one_except_to_one_topic[:, index_of_topic_to_be_nearest_to_all_documents] = 0
        documents_viewer = top_documents_viewer.TopDocumentsViewer(
            model=TestTopDocumentsViewer.topic_model,
            precomputed_distances=distances_all_one_except_to_one_topic)

        topics_documents = documents_viewer.view()
        num_documents_in_nearest_topic = len(
            topics_documents[name_of_topic_to_be_nearest_to_all_document])
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

        viewer_output = documents_viewer.view()

        assert all(len(value) <= max_num_top_documents
                   for _, value in viewer_output.items()),\
            'Not all top documents lists from "{}" have less elements than required "{}"'.format(
                topics_documents, max_num_top_documents)

    def test_check_object_clusters_parameter_workable(self):
        """ """
        num_documents = TestTopDocumentsViewer.theta.shape[1]
        cluster_label_to_be_same_for_all_documents = 0
        cluster_name_to_be_same_for_all_documents = 'topic_0'
        cluster_labels = list(
            cluster_label_to_be_same_for_all_documents for _ in range(num_documents))

        documents_viewer = top_documents_viewer.TopDocumentsViewer(
            model=TestTopDocumentsViewer.topic_model,
            object_clusters=cluster_labels)

        topics_documents = documents_viewer.view()
        num_documents_with_given_cluster_label = len(
            topics_documents[cluster_name_to_be_same_for_all_documents])

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
