from typing import List

from .recipe_wrapper import BaseRecipe
from .. import Dataset


ARTM_baseline_template = '''
# This config follows a strategy described by Murat Apishev
# one of the core programmers of BigARTM library in personal correspondence.
# According to his letter 'decent' topic model can be obtained by
# Decorrelating model topics simultaneously looking at retrieved TopTokens


# Use .format(modality_list=modality_list, main_modality=main_modality, dataset_path=dataset_path,
# specific_topics=specific_topics, background_topics=background_topics)
# when loading the recipe to adjust for your dataset

topics:
# Describes number of model topics, better left to the user to define optimal topic number
    specific_topics: {specific_topics}
    background_topics: {background_topics}

# Here is example of model with one modality
regularizers:
    - DecorrelatorPhiRegularizer:
        name: decorrelation_phi
        topic_names: specific_topics
        class_ids: {modality_list}
    - SmoothSparsePhiRegularizer:
        name: smooth_phi_bcg
        topic_names: background_topics
        class_ids: {modality_list}
        tau: 0.1
        relative: true
    - SmoothSparseThetaRegularizer:
        name: smooth_theta_bcg
        topic_names: background_topics
        tau: 0.1
        relative: true
scores:
    - BleiLaffertyScore:
        num_top_tokens: 30
model:
    dataset_path: {dataset_path}
    {dictionary_filter_parameters}
    modalities_to_use: {modality_list}
    main_modality: '{main_modality}'

stages:
- RegularizersModifierCube:
    num_iter: 20
    reg_search: add
    regularizer_parameters:
        name: decorrelation_phi
    selection:
        - PerplexityScore@all < 1.05 * MINIMUM(PerplexityScore@all) and BleiLaffertyScore -> max
    strategy: PerplexityStrategy
    # parameters of this strategy are intended for revision
    strategy_params:
        start_point: 0
        step: 0.01
        max_len: 50
    tracked_score_function: PerplexityScore@all
    verbose: false
    use_relative_coefficients: true
'''

ONE_CONFIG_INDENT = 4 * ' '


class BaselineRecipe(BaseRecipe):
    """
    Class for baseline recipe creation and
    unification of recipe interface
    """
    def __init__(self):
        super().__init__(recipe_template=ARTM_baseline_template)

    def format_recipe(
        self,
        dataset_path: str,
        dictionary_filter_parameters: dict = None,
        modality_list: List[str] = None,
        topic_number: int = 20,
        background_topic_number: int = 1,
        num_iter: int = 20,
    ):
        if modality_list is None:
            modality_list = list(Dataset(dataset_path).get_possible_modalities())

        specific_topics = [f'topic_{i}' for i in range(topic_number)]
        background_topics = [f'bcg_{i}' for i in range(
            len(specific_topics), len(specific_topics) + background_topic_number)]

        if dictionary_filter_parameters is None:
            dictionary_filter_parameters = dict()

        dictionary_filter_parameters_as_yml = self._format_dictionary_filter_parameters(
            dictionary_filter_parameters,
            indent=2 * ONE_CONFIG_INDENT,
        )

        self._recipe = self.recipe_template.format(
            dataset_path=dataset_path,
            dictionary_filter_parameters=dictionary_filter_parameters_as_yml,
            modality_list=modality_list,
            main_modality=modality_list[0],
            specific_topics=specific_topics,
            background_topics=background_topics,
        )

        return self._recipe
