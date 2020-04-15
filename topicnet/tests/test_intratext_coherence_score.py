from collections import defaultdict
from functools import reduce
from itertools import product
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile
from typing import List, Dict


from ..cooking_machine.config_parser import build_experiment_environment_from_yaml_config
from ..cooking_machine.dataset import Dataset, DEFAULT_ARTM_MODALITY
from ..cooking_machine.models.base_model import BaseModel
from ..cooking_machine.models.frozen_score import FrozenScore
from ..cooking_machine.models.intratext_coherence_score import (
    IntratextCoherenceScore,
    TextType,
    ComputationMethod,
    WordTopicRelatednessType,
    SpecificityEstimationMethod
)


NUM_TOP_WORDS = 10
BIG_SEGMENT_LENGTH = 8
SMALL_SEGMENT_LENGTHS = [1, 2, 4]
SMALL_SEGMENT_LENGTH_PROBABILITIES = [0.3, 0.45, 0.25]
DOCUMENT_LENGTH = 100
TOP_WORD_PROBABILITY_TIMES_BIGGER = 4
PHI_FILE_NAME = 'phi.csv'
DATASET_FILE_NAME = 'dataset.csv'


class MockModel(BaseModel):
    def __init__(self, phi):
        self._phi = phi

    def get_phi(self):
        return self._phi.copy()


class TestIntratextCoherenceScore:
    topics = ['topic_1', 'topic_2', 'topic_3']
    documents = ['doc_1', 'doc_2', 'doc_3']
    topic_documents = {
        'topic_1': ['doc_1', 'doc_2'],
        'topic_2': ['doc_3'],
        'topic_3': []
    }
    best_topic = 'topic_1'
    out_of_documents_topic = 'topic_3'
    document_topics = {
        'doc_1': ['topic_1', 'topic_2'],
        'doc_2': ['topic_1'],
        'doc_3': ['topic_1', 'topic_2']
    }
    top_words = {
        topic: [f'{topic}_word_{i}' for i in range(1, NUM_TOP_WORDS + 1)]
        for topic in topics
    }
    vocabulary = list(reduce(lambda res, cur: res + cur, top_words.values(), []))
    out_of_topics_word = 'unknown_word'

    data_folder_path = None
    model = None
    dataset = None
    dataset_file_path = None

    @classmethod
    def setup_class(cls):
        cls.model = MockModel(cls.create_phi())

        document_words = cls.create_documents()
        dataset_table = cls.create_dataset_table(document_words)

        cls.data_folder_path = tempfile.mkdtemp()

        cls.dataset_file_path = os.path.join(cls.data_folder_path, DATASET_FILE_NAME)
        dataset_table.to_csv(cls.dataset_file_path)

        cls.dataset = Dataset(cls.dataset_file_path)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.data_folder_path)

    @classmethod
    def create_phi(cls) -> pd.DataFrame:
        phi = pd.DataFrame(
            index=[(DEFAULT_ARTM_MODALITY, w) for w in cls.vocabulary],
            columns=cls.topics,
            data=np.random.random((len(cls.vocabulary), len(cls.topics)))
        )

        for t in cls.topics:
            phi.loc[[(DEFAULT_ARTM_MODALITY, w)
                     for w in cls.top_words[t]], t] = 1.0

            phi.loc[[(DEFAULT_ARTM_MODALITY, w)
                     for w in cls.vocabulary
                     if w not in cls.top_words[t]], t] = 1.0 / TOP_WORD_PROBABILITY_TIMES_BIGGER

        phi[cls.out_of_documents_topic] = 1.0  # so as next line works fine
        phi[:] = phi.values / np.sum(phi.values, axis=0, keepdims=True)
        phi[cls.out_of_documents_topic] = 0.0  # and now exclude all the words from the topic

        phi.index = pd.MultiIndex.from_tuples(
            phi.index, names=('modality', 'token'))  # TODO: copy-paste from TopicModel

        return phi

    @classmethod
    def create_documents(cls) -> Dict[str, List[str]]:
        def get_segments(
                topic: str, other_topics: List[str], top_words: Dict[str, List[str]]
        ) -> List[List[str]]:

            num_words = 0
            segments = []
            is_main_topic = True
            is_out_of_topics_word_included = False

            while num_words < DOCUMENT_LENGTH:
                if len(other_topics) == 0:
                    is_main_topic = True

                if is_main_topic:
                    current_topic = topic
                    current_segment_length = BIG_SEGMENT_LENGTH

                else:
                    current_topic = np.random.choice(other_topics)
                    current_segment_length = np.random.choice(
                        SMALL_SEGMENT_LENGTHS,
                        p=SMALL_SEGMENT_LENGTH_PROBABILITIES
                    )

                segment = np.random.choice(
                    top_words[current_topic],
                    current_segment_length
                )
                segment = segment.tolist()

                if not is_out_of_topics_word_included:
                    segment += [cls.out_of_topics_word]
                    is_out_of_topics_word_included = True

                is_main_topic = not is_main_topic

                num_words += len(segment)
                segments.append(segment)

            return segments

        document_words = defaultdict(list)

        for t, docs in cls.topic_documents.items():
            all_other_topics = list(set(cls.topic_documents.keys()).difference([t]))

            for doc in docs:
                other_topics = list(set(all_other_topics).intersection(
                    cls.document_topics[doc]
                ))

                document_words[doc] = list(reduce(
                    lambda res, cur: res + cur,
                    get_segments(t, other_topics, cls.top_words),
                    []
                ))

        return document_words

    @classmethod
    def create_dataset_table(cls, document_words: Dict[str, List[str]]):
        return pd.DataFrame(
            index=cls.documents,
            columns=['id', 'raw_text', 'vw_text'],
            data=[
                [doc, cls.get_raw_text(doc, document_words), cls.get_vw_text(doc, document_words)]
                for doc in cls.documents
            ]
        )

    @classmethod
    def get_raw_text(cls, doc: str, document_words: Dict[str, List[str]]) -> str:
        return ' '.join(document_words[doc])

    @classmethod
    def get_vw_text(cls, doc: str, document_words: Dict[str, List[str]]) -> str:
        return doc + ' ' + ' '.join(document_words[doc])

    def smoke_check_compute_coherence(
            self,
            text_type,
            computation_method,
            word_topic_relatedness,
            specificity_estimation):

        score = IntratextCoherenceScore(
            self.dataset,
            text_type=text_type,
            computation_method=computation_method,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        coherences = score.compute(self.model)
        coherence_values = list(coherences.values())
        maximum_coherence = max(c for c in coherence_values if c is not None)

        assert coherences[self.best_topic] == maximum_coherence,\
            'Topic that expected to be best doesn\'t have max coherence'

        assert coherences[self.out_of_documents_topic] is None,\
            'Topic that is not in any document has coherence other than None'

    def check_call(
            self,
            text_type,
            computation_method,
            word_topic_relatedness,
            specificity_estimation,
            documents=None):

        score = IntratextCoherenceScore(
            self.dataset,
            documents=documents,
            text_type=text_type,
            computation_method=computation_method,
            word_topic_relatedness=word_topic_relatedness,
            specificity_estimation=specificity_estimation
        )

        value = score.call(self.model)

        assert isinstance(value, float), f'Wrong score value type {type(value)}'

    @pytest.mark.parametrize(
        'text_type, computation_method, word_topic_relatedness, specificity_estimation',
        list(product(
            [TextType.VW_TEXT, TextType.RAW_TEXT],
            [ComputationMethod.SEGMENT_LENGTH, ComputationMethod.SEGMENT_WEIGHT,
             ComputationMethod.SUM_OVER_WINDOW],
            [WordTopicRelatednessType.PWT, WordTopicRelatednessType.PTW],
            [SpecificityEstimationMethod.NONE, SpecificityEstimationMethod.MAXIMUM,
             SpecificityEstimationMethod.AVERAGE]
        ))
    )
    def test_smoke_compute_coherence(
            self, text_type, computation_method, word_topic_relatedness, specificity_estimation):

        self.smoke_check_compute_coherence(
            text_type, computation_method, word_topic_relatedness, specificity_estimation
        )

    @pytest.mark.parametrize(
        'text_type, computation_method, word_topic_relatedness, specificity_estimation',
        list(product(
            [TextType.VW_TEXT, TextType.RAW_TEXT],
            [ComputationMethod.SEGMENT_LENGTH, ComputationMethod.SEGMENT_WEIGHT,
             ComputationMethod.SUM_OVER_WINDOW],
            [WordTopicRelatednessType.PWT, WordTopicRelatednessType.PTW],
            [SpecificityEstimationMethod.NONE, SpecificityEstimationMethod.MAXIMUM,
             SpecificityEstimationMethod.AVERAGE]
        ))
    )
    def test_smoke_call(
            self, text_type, computation_method, word_topic_relatedness, specificity_estimation):

        self.check_call(
            text_type, computation_method, word_topic_relatedness, specificity_estimation
        )

    def test_freeze(self):
        score = IntratextCoherenceScore(
            self.dataset,
            documents=self.documents,
            text_type=TextType.VW_TEXT,
            computation_method=ComputationMethod.SEGMENT_LENGTH,
            word_topic_relatedness=WordTopicRelatednessType.PWT,
            specificity_estimation=SpecificityEstimationMethod.NONE
        )

        frozen_score = FrozenScore(score.value, score)

        for attribute_name in [
                'value',
                '_text_type',
                '_computation_method',
                '_word_topic_relatedness',
                '_specificity_estimation_method',
                '_max_num_out_of_topic_words',
                '_window']:

            assert hasattr(frozen_score, attribute_name)
            assert getattr(frozen_score, attribute_name) == getattr(score, attribute_name)

    @pytest.mark.parametrize(
        'text_type, computation_method, word_topic_relatedness, specificity_estimation',
        list(product(
            [TextType.VW_TEXT, TextType.RAW_TEXT],
            [ComputationMethod.SEGMENT_LENGTH, ComputationMethod.SEGMENT_WEIGHT,
             ComputationMethod.SUM_OVER_WINDOW],
            [WordTopicRelatednessType.PWT, WordTopicRelatednessType.PTW],
            [SpecificityEstimationMethod.NONE, SpecificityEstimationMethod.MAXIMUM,
             SpecificityEstimationMethod.AVERAGE]
        ))
    )
    @pytest.mark.parametrize(
        'what_documents',
        ['first', 'all', 'empty', 'none']
    )
    def test_call_with_specified_documents(
            self, text_type, computation_method, word_topic_relatedness, specificity_estimation,
            what_documents):

        if what_documents == 'first':
            documents = [self.documents[0]]
        elif what_documents == 'all':
            documents = self.documents
        elif what_documents == 'empty':
            documents = list()
        elif what_documents == 'none':
            documents = None
        else:
            raise ValueError(f'{what_documents}')

        self.check_call(
            text_type,
            computation_method,
            word_topic_relatedness,
            specificity_estimation,
            documents
        )

    @pytest.mark.parametrize('keep_dataset', [False, True])
    @pytest.mark.parametrize('low_memory', [False, True])
    def test_recipe(self, keep_dataset, low_memory):
        recipe_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'cooking_machine',
            'recipes',
            'intratext_coherence_maximization.yml',
        )

        recipe_config_string = open(recipe_file_path, 'r').read()
        recipe_config_string = recipe_config_string.format(
            modality_names=[DEFAULT_ARTM_MODALITY],
            main_modality=DEFAULT_ARTM_MODALITY,
            dataset_path=self.dataset_file_path,
            keep_dataset_in_memory=True,
            keep_dataset=keep_dataset,
            documents_fraction=1.0,
            specific_topics=self.topics[:-1],
            background_topics=self.topics[-1:],
            one_stage_num_iter=2,
            verbose=False,
        )

        experiment, dataset = build_experiment_environment_from_yaml_config(
            recipe_config_string,
            experiment_id=(
                f'experiment_maximize_intratext'
                f'__{keep_dataset}'
                f'__{low_memory}'
            ),
            save_path=self.data_folder_path,
        )
        experiment._low_memory = low_memory  # TODO: add some better test for low_memory?

        experiment.run(dataset)

        score_name = 'IntratextCoherenceScore'

        best_model = None
        levels = range(1, len(experiment.cubes) + 1)

        # TODO: probably need such method in Experiment?
        #  (i.e. "select best of all", not only on level=level)
        for level in levels:
            best_model_candidates = experiment.select(
                f'{score_name} -> max',
                level=level
            )

            if len(best_model_candidates) == 0:
                continue

            best_model_candidate = best_model_candidates[0]

            if (best_model is None or
                    best_model.scores[score_name][-1] <
                    best_model_candidate.scores[score_name][-1]):

                best_model = best_model_candidate

        models_to_compare_with = [
            m for m in experiment.models.values()
            if len(m.scores[score_name]) > 0
        ]

        assert all([
            m.scores[score_name][-1] <= best_model.scores[score_name][-1]
            for m in models_to_compare_with
        ])
        assert any([
            m.scores[score_name][-1] < best_model.scores[score_name][-1]
            for m in models_to_compare_with
        ])
