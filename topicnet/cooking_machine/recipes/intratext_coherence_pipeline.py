import os
import warnings

from typing import List

from .recipe_wrapper import BaseRecipe
from .. import Dataset


class IntratextCoherenceRecipe(BaseRecipe):
    """
    The recipe mainly consists of basic cube stages,
    such as Decorrelation, Sparsing and Smoothing.
    In this way it is similar to ARTM baseline recipe.
    The core difference is that models selected based on their IntratextCoherenceScore
    (which is one of the scores included in TopicNet).
    PerplexityScore is also calculated to assure that models don't have high perplexity,
    but the main criteria is IntratextCoherenceScore.

    For more details about IntratextCoherence
    one may see the paper http://www.dialog-21.ru/media/4281/alekseevva.pdf

    """
    def __init__(self):
        recipe_template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'intratext_coherence_maximization.yml'
        )
        recipe_template = open(recipe_template_path, 'r').read()

        super().__init__(recipe_template=recipe_template)

    def format_recipe(
            self,
            dataset_path: str,
            num_specific_topics: int,
            main_modality: str = None,
            num_background_topics: int = 1,
            modalities: List[str] = None,
            keep_dataset_in_memory: bool = True,
            keep_dataset: bool = False,
            documents_fraction: float = 0.5,
            one_stage_num_iter: int = 20,
            verbose: bool = True) -> str:
        """

        Parameters
        ----------
        dataset_path
            Path to the dataset .csv file
        num_specific_topics
            Number of specific topics in models to be trained
        main_modality
            Main modality in the dataset
            (usually it is plain text, and not, for example, @author or @title)
            If not specified, it will be the first modality in `modalities`
        num_background_topics
            Number of background topics in models
        modalities
            What modalities to use from those that are in the dataset.
            If not specified, all dataset's modalities will be used.
            If specified, should be non empty
        keep_dataset_in_memory
            Whether or not to keep dataset in memory when running experiment.
            True is faster, so, if dataset is not very huge, it is better to use True
        keep_dataset
            If True, the dataset will be loaded in memory only when computing coherence.
            So, memory will be free of the dataset during model training.
            This may help if the dataset is fairly big,
            but `keep_dataset_in_memory=True` still works without crash.
        documents_fraction
            Determines the number of documents that will be used for computing coherence.
            Better keep this one less than 1.0.
            For example, suppose we want to use not all dataset,
            but just a fragment of 25,000 words.
            Then we can do like so

            >>> document_lengths = dataset._data['vw_text'].apply(lambda text: len(text.split()))
            >>> median_document_length = np.median(document_lengths)
            >>> num_documents = dataset._data.shape[0]
            >>> dataset_fragment_length = 25000
            >>> num_documents_for_computing = dataset_fragment_length / median_document_length
            >>> documents_fraction = num_documents_for_computing / num_documents

        one_stage_num_iter
            There will be five stages, each with nearly 5-values-grid search.
            One such search lasts `one_stage_num_iter` iterations
            with coherence computation in the end.
            So, there is going to be `one_stage_num_iter` * 5 * 5 training iterations (not slow)
            and 5 * 5 coherence computations (here may be slow if `documents_fraction` is high)
        verbose
            Whether to show experiment progress or not

        """
        all_modalities = list(Dataset(dataset_path).get_possible_modalities())

        if len(all_modalities) == 0:
            warnings.warn(f'No modalities in the dataset "{dataset_path}"!')

        if modalities is None:
            modalities = all_modalities
        if any([m not in all_modalities for m in modalities]):
            warnings.warn(f'Not all `modalities` are found in the dataset "{dataset_path}"!')

        if main_modality is None:
            main_modality = modalities[0]

            warnings.warn(
                f'Main modality not specified!'
                f' So modality "{main_modality}" will be used as the main one'
            )

        specific_topics = [
            f'topic_{i}' for i in range(num_specific_topics)
        ]
        background_topics = [
            f'bcg_topic_{i}'
            for i in range(num_specific_topics, num_specific_topics + num_background_topics)
        ]

        self._recipe = self.recipe_template.format(
            modality_names=modalities,
            main_modality=main_modality,
            dataset_path=dataset_path,
            keep_dataset_in_memory=keep_dataset_in_memory,
            keep_dataset=keep_dataset,
            documents_fraction=documents_fraction,
            specific_topics=specific_topics,
            background_topics=background_topics,
            one_stage_num_iter=one_stage_num_iter,
            verbose=verbose,
        )

        return self._recipe
