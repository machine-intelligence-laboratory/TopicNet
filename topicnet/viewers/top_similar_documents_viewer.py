from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist as sp_cdist
import warnings

from .base_viewer import BaseViewer
from ..cooking_machine import BaseDataset


# If change, also modify docstring for view()
METRICS_NAMES = [
    'jensenshannon', 'euclidean', 'cosine', 'correlation'
]


ERROR_DUPLICATE_DOCUMENTS_IDS = """\
Some documents' IDs in dataset are the same: \
number of unique IDs and total number of documents not equal: "{0}" vs. "{1}". \
Need unique IDs in order to identify documents.\
"""

ERROR_TYPE_METRIC = """\
Parameter "metric" should be "str" or "callable". \
The argument given is of type "{0}"\
"""

ERROR_TYPE_NUM_TOP_SIMILAR = """\
Parameter "num_top_similar" should be "int". \
The argument given is of type "{0}"\
"""

ERROR_TYPE_KEEP_SIMILAR_BY_WORDS = """\
Parameter "keep_similar_by_words" should be "bool". \
The argument given is of type "{0}"\
"""

WARNING_UNDEFINED_FREQUENCY_IN_VW = """\
Some words in Vowpal Wabbit text were skipped \
because they didn\'t have frequency after colon sign ":"\
"""

WARNING_FEWER_THAN_REQUESTED = """\
Only "{0}" documents available{1}. \
This is smaller than the requested number of top similar documents "{2}". \
So output is going to contain all "{0}" documents, but sorted by distance\
"""

WARNING_TOO_MANY_REQUESTED = """\
Requested number of top similar documents "{0}" \
is bigger than total number of documents in the dataset "{1}"\
"""


class TopSimilarDocumentsViewer(BaseViewer):
    def __init__(self, model, dataset):
        """Viewer which uses topic model to find documents similar to given one

        Parameters
        ----------
        model : BaseModel
            Topic model
        dataset : BaseDataset
            Dataset with information about documents
        """
        super().__init__(model=model)

        if not isinstance(dataset, BaseDataset):
            raise TypeError('Parameter "dataset" should derive from BaseDataset')

        self._dataset = dataset
        self._theta = self.model.get_theta(dataset=self._dataset)

        self._documents_ids = list(self._theta.columns)

        if len(self._documents_ids) == 0:
            warnings.warn('No documents in given dataset', UserWarning)
        elif len(set(self._documents_ids)) != len(self._documents_ids):
            raise ValueError(ERROR_DUPLICATE_DOCUMENTS_IDS.format(
                len(set(self._documents_ids)), len(self._documents_ids)))

    def view(self,
             document_id,
             metric='jensenshannon',
             num_top_similar=5,
             keep_similar_by_words=True):
        """Shows documents similar to given one by distribution of topics

        Parameters
        ----------
        document_id
            ID of the document in `dataset`
        metric : str or callable
            Distance measure which is to be used to measure how documents differ from each other
            If str -- should be one of 'jensenshannon', 'euclidean', 'cosine', 'correlation' --
                as in scipy.spatial.distance.cdist
            If callable -- should map two vectors to numeric value
        num_top_similar : int
            How many top similar documents' IDs to show
        keep_similar_by_words : bool
            Whether or not to keep in the output those documents
            that are similar to the given one by their constituent words and words' frequencies

        Returns
        -------
        tuple(list, list)
            Top similar words, and corresponding distances to given document
        """
        self._check_view_parameters_valid(
            document_id=document_id,
            metric=metric,
            num_top_similar=num_top_similar,
            keep_similar_by_words=keep_similar_by_words)

        num_top_similar = min(num_top_similar, len(self._documents_ids))
        document_index = self._documents_ids.index(document_id)

        similar_documents_indices, distances = self._view(
            document_index=document_index,
            metric=metric,
            num_top_similar=num_top_similar,
            keep_similar_by_words=keep_similar_by_words)

        documents_ids = [self._documents_ids[doc_index] for doc_index in similar_documents_indices]

        return documents_ids, distances

    def _view(self,
              document_index,
              metric,
              num_top_similar,
              keep_similar_by_words):

        documents_indices = [i for i, _ in enumerate(self._documents_ids) if i != document_index]
        distances = self._get_documents_distances(documents_indices, document_index, metric)

        documents_indices, distances = \
            TopSimilarDocumentsViewer._sort_elements_by_corresponding_values(
                documents_indices, distances)

        if keep_similar_by_words or len(documents_indices) == 0:
            documents_indices_to_exclude = []
        else:
            documents_indices_to_exclude = \
                self._get_documents_with_similar_words_frequencies_indices(
                    documents_indices, document_index, num_top_similar)

        if len(documents_indices) == len(documents_indices_to_exclude):
            return self._empty_view
        elif len(documents_indices) - len(documents_indices_to_exclude) < num_top_similar:
            warnings.warn(
                WARNING_FEWER_THAN_REQUESTED.format(
                    len(documents_indices_to_exclude),
                    (' after throwing out documents similar just by words'
                     if not keep_similar_by_words else ''),
                    num_top_similar),
                RuntimeWarning
            )

        documents_indices, distances =\
            TopSimilarDocumentsViewer._filter_elements_and_values(
                documents_indices, distances, documents_indices_to_exclude)

        similar_documents_indices = documents_indices[:num_top_similar]
        similar_documents_distances = distances[:num_top_similar]

        return similar_documents_indices, similar_documents_distances

    @staticmethod
    def _sort_elements_by_corresponding_values(elements, values, ascending=True):
        def unzip(zipped):
            # Transforms [(a, A), (b, B), ...] to [a, b, ...], [A, B, ...]
            return list(zip(*zipped))

        elements_values = sorted(zip(elements, values), key=lambda kv: kv[1])

        if not ascending:
            elements_values = elements_values[::-1]

        return unzip(elements_values)

    @staticmethod
    def _filter_elements_and_values(elements, values, elements_to_exclude):
        elements_to_exclude = set(elements_to_exclude)
        indices_to_exclude = set([i for i, e in enumerate(elements) if e in elements_to_exclude])

        result_elements = [e for i, e in enumerate(elements) if i not in indices_to_exclude]
        result_values = [v for i, v in enumerate(values) if i not in indices_to_exclude]

        assert len(result_elements) == len(result_values)

        return result_elements, result_values

    @staticmethod
    def _are_words_frequencies_similar(words_frequencies_a, words_frequencies_b):
        # TODO: method seems very ... heuristic
        # maybe need some research to find the best way to compare words frequencies
        word_frequency_pairs_a = sorted(words_frequencies_a.items(), key=lambda kv: kv[1])
        word_frequency_pairs_b = sorted(words_frequencies_b.items(), key=lambda kv: kv[1])

        num_top_words_to_consider = 100
        jaccard_coefficient = TopSimilarDocumentsViewer._get_jaccard_coefficient(
            word_frequency_pairs_a[:num_top_words_to_consider],
            word_frequency_pairs_b[:num_top_words_to_consider])
        jaccard_coefficient_threshold_to_be_similar = 0.6

        return jaccard_coefficient >= jaccard_coefficient_threshold_to_be_similar

    @staticmethod
    def _get_jaccard_coefficient(word_frequency_pairs_a, word_frequency_pairs_b):
        def get_values_sum(dictionary, default=0.0):
            return sum(dictionary.values() or [default])

        def get_normalized_values(key_value_pairs):
            tiny = 1e-7
            denominator = sum(kv[1] for kv in key_value_pairs) or tiny

            return {k: v / denominator for k, v in key_value_pairs}

        # May help in case documents differ in length significantly
        frequencies_a = get_normalized_values(word_frequency_pairs_a)
        frequencies_b = get_normalized_values(word_frequency_pairs_b)

        words_a, words_b = set(frequencies_a), set(frequencies_b)

        intersection = {
            e: min(frequencies_a[e], frequencies_b[e])
            for e in words_a & words_b
        }
        union = {
            e: max(frequencies_a.get(e, 0), frequencies_b.get(e, 0))
            for e in words_a | words_b
        }

        if len(union) == 0:
            return 0.0

        return get_values_sum(intersection) / get_values_sum(union)

    @staticmethod
    def _extract_words_frequencies(vw_text):
        # Just gather frequencies of words of all modalities
        # TODO: use Dataset for this?

        def is_modality_name(vw_word):
            return vw_word.startswith('|')

        words_frequencies = defaultdict(int)
        has_words_with_undefined_frequencies = False

        for vw_word in vw_text.split():
            if is_modality_name(vw_word):
                continue

            if ':' in vw_word:
                word, frequency = vw_word.split(':')

                if len(frequency) == 0:
                    has_words_with_undefined_frequencies = True
                    continue

                # to allow frequencies as float's but assure that now all are int-s
                frequency = int(round(float(frequency)))
            else:
                word = vw_word
                frequency = 1

            words_frequencies[word] += frequency

        if has_words_with_undefined_frequencies:
            warnings.warn(WARNING_UNDEFINED_FREQUENCY_IN_VW, UserWarning)

        return words_frequencies

    @property
    def _empty_view(self):
        empty_top_similar_documents_list = list()
        empty_distances_list = list()

        return empty_top_similar_documents_list, empty_distances_list

    def _check_view_parameters_valid(
            self, document_id, metric, num_top_similar, keep_similar_by_words):

        if document_id not in self._documents_ids:
            raise ValueError('No document with such id "{}" in dataset'.format(document_id))

        if isinstance(metric, str):
            TopSimilarDocumentsViewer._check_str_metric_valid(metric)
        elif callable(metric):
            TopSimilarDocumentsViewer._check_callable_metric_valid(metric)
        else:
            raise TypeError(ERROR_TYPE_METRIC.format(type(metric)))

        if not isinstance(num_top_similar, int):
            raise TypeError(ERROR_TYPE_NUM_TOP_SIMILAR.format(type(num_top_similar)))
        elif num_top_similar < 0:
            raise ValueError('Parameter "num_top_similar" should be greater than zero')
        elif num_top_similar == 0:
            return self._empty_view
        elif num_top_similar > len(self._documents_ids):
            warnings.warn(
                WARNING_TOO_MANY_REQUESTED.format(
                    num_top_similar, len(self._documents_ids)),
                UserWarning
            )

        if not isinstance(keep_similar_by_words, bool):
            raise TypeError(ERROR_TYPE_KEEP_SIMILAR_BY_WORDS.format(type(keep_similar_by_words)))

    @staticmethod
    def _check_str_metric_valid(metric):
        if metric not in METRICS_NAMES:
            raise ValueError('Unknown metric name "{}", expected one of "{}"'.format(
                metric, ' '.join(METRICS_NAMES)))

    @staticmethod
    def _check_callable_metric_valid(metric):
        try:
            metric(np.array([0]), np.array([0]))
        except TypeError:  # more or less arguments or they are of wrong type for operation
            raise ValueError('Invalid "callable" metric')

    def _get_documents_distances(
            self,
            documents_indices_to_measure_distance_from,
            document_index_to_measure_distance_to,
            metric):

        theta_submatrix = self._theta.iloc[:, documents_indices_to_measure_distance_from]
        documents_vectors = theta_submatrix.T.values

        assert documents_vectors.ndim == 2

        theta_column = self._theta.iloc[:, document_index_to_measure_distance_to]
        document_vector = theta_column.T.values

        assert document_vector.ndim == 1

        document_vector = document_vector.reshape(1, -1)

        assert document_vector.ndim == 2
        assert document_vector.shape[0] == 1
        assert document_vector.shape[1] == documents_vectors.shape[1]

        answer = sp_cdist(documents_vectors, document_vector, metric)

        return answer.flatten()

    def _get_documents_with_similar_words_frequencies_indices(
            self, documents_indices, document_index_to_compare_with,
            num_dissimilar_documents_to_stop_searching):

        # Method is not going to find all similar documents
        # It terminates when enough dissimilar documents are encountered

        similar_documents_indices = []
        num_encountered_dissimilar_documents = 0
        words_frequencies_to_compare_with = \
            self._get_words_frequencies(document_index_to_compare_with)

        for i, doc_index in enumerate(documents_indices):
            if num_encountered_dissimilar_documents == num_dissimilar_documents_to_stop_searching:
                break

            if TopSimilarDocumentsViewer._are_words_frequencies_similar(
                    self._get_words_frequencies(i),
                    words_frequencies_to_compare_with):
                similar_documents_indices.append(doc_index)
            else:
                num_encountered_dissimilar_documents += 1

        return similar_documents_indices

    def _get_words_frequencies(self, document_index):
        vw_text = self._get_vw_text(document_index)

        return TopSimilarDocumentsViewer._extract_words_frequencies(vw_text)

    def _get_vw_text(self, document_index):
        dataset = self._dataset.get_dataset()

        return dataset.iloc[document_index, dataset.columns.get_loc('vw_text')]


def _run_view(viewer, document_id, keep_similar_by_words=True):
    print(
        '> similar_documents, distances = viewer.view('
        'document_id={}{})'.format(
            document_id,
            ', keep_similar_by_word=False' if not keep_similar_by_words else ''))

    similar_documents, distances = viewer.view(
        document_id=document_id, keep_similar_by_words=keep_similar_by_words)

    print('similar_documents:', similar_documents)
    print('distances:', ['{:.4f}'.format(d) for d in distances])
    print()


def _main():
    print('Starting TopSimilarDocumentsViewer\'s view() demonstration!', end='\n\n')

    import artm
    import os

    from cooking_machine.dataset import Dataset
    from cooking_machine.models.topic_model import TopicModel
    from viewers.top_similar_documents_viewer import TopSimilarDocumentsViewer

    current_folder = os.path.dirname(os.path.abspath(__file__))
    dataset = Dataset(os.path.join(current_folder, '../tests/test_data/test_dataset.csv'))

    num_topics = 3
    artm_model = artm.ARTM(
        topic_names=['topic_{}'.format(i) for i in range(num_topics)],
        theta_columns_naming='id',
        show_progress_bars=False,
        cache_theta=True)
    artm_model.initialize(dataset.get_dictionary())

    model = TopicModel(artm_model)
    num_iterations = 10
    model._fit(
        dataset_trainable=dataset.get_batch_vectorizer(),
        num_iterations=num_iterations)

    viewer = TopSimilarDocumentsViewer(
        model=model,
        dataset=dataset)

    # One may look if in notebook
    # artm_model.get_theta()
    # dataset.get_dataset()

    print('Documents\' ids:', viewer._documents_ids, end='\n\n')

    _run_view(viewer, document_id="doc_2")
    _run_view(viewer, document_id="doc_5")
    _run_view(viewer, document_id="doc_8")
    _run_view(viewer, document_id="doc_5", keep_similar_by_words=False)


# python -m viewers.top_similar_documents_viewer
if __name__ == '__main__':
    _main()
