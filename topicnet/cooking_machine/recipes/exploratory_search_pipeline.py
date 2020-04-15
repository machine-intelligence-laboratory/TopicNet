from .recipe_wrapper import BaseRecipe
from .. import Dataset

modality_selection_template = (
    'PerplexityScore{modality}'
    ' < 1.01 * MINIMUM(PerplexityScore{modality}) and SparsityPhiScore{modality} -> max'
)
general_selection_template = (
    'PerplexityScore@all'
    ' < 1.01 * MINIMUM(PerplexityScore@all) and SparsityPhiScore{modality} -> max'
)

exploratory_search_template = '''
# This config follows a strategy described in the article
# Multi-objective Topic Modeling for Exploratory Search in Tech News
# by Anastasya Yanina, Lev Golitsyn and Konstantin Vorontsov, Jan 2018


# Use .format(modality=modality, dataset_path=dataset_path,
# specific_topics=specific_topics, background_topics=background_topics)
# when loading the recipe to adjust for your dataset

topics:
# Describes number of model topics, in the actuall article 200 topics were found to be optimal
    specific_topics: {{specific_topics}}
    background_topics: {{background_topics}}

regularizers:
- DecorrelatorPhiRegularizer:
    name: decorrelation_phi_{{modality}}
    topic_names: specific_topics
    tau: 1
    class_ids: ['{{modality}}']
- SmoothSparsePhiRegularizer:
    name: smooth_phi_{{modality}}
    topic_names: specific_topics
    tau: 1
    class_ids: ['{{modality}}']
- SmoothSparseThetaRegularizer:
    name: sparse_theta
    topic_names: specific_topics
    tau: 1

model:
    dataset_path: {{dataset_path}}
    modalities_to_use: ['{{modality}}']
    main_modality: '{{modality}}'

stages:
# repeat the following two cubes for every modality in the dataset
- RegularizersModifierCube:
    num_iter: 8
    reg_search: mul
    regularizer_parameters:
        name: decorrelation_phi_{{modality}}
    selection:
        - {0}
    strategy: PerplexityStrategy
    strategy_params:
        start_point: 100000
        step: 10
        max_len: 6
    tracked_score_function: PerplexityScore@all
    verbose: false
    use_relative_coefficients: false
- RegularizersModifierCube:
    num_iter: 8
    reg_search: add
    regularizer_parameters:
        name: smooth_phi_{{modality}}
    selection:
        - {0}
    strategy: PerplexityStrategy
    strategy_params:
        start_point: 0.25
        step: 0.25
        max_len: 6
    tracked_score_function: PerplexityScore{{modality}}
    verbose: false
    use_relative_coefficients: false
#last cube is independent of modalities and can be used only once
- RegularizersModifierCube:
    num_iter: 8
    reg_search: add
    regularizer_parameters:
        name: sparse_theta
    selection:
        - {1}
    strategy: PerplexityStrategy
    strategy_params:
        start_point: -0.5
        step: -0.5
        max_len: 6
    tracked_score_function: PerplexityScore@all
    verbose: false
    use_relative_coefficients: false

'''.format(modality_selection_template, general_selection_template)


class SearchRecipe(BaseRecipe):
    """
    Class for baseline recipe creation and
    unification of recipe interface
    """
    def __init__(self):
        super().__init__(recipe_template=exploratory_search_template)

    def format_recipe(
        self,
        dataset_path: str,
        modality: str = None,
        topic_number: int = 20,
        background_topic_number: int = 1,
    ):
        if modality is None:
            modality = list(Dataset(dataset_path).get_possible_modalities())[0]

        specific_topics = [f'topic_{i}' for i in range(topic_number)]
        background_topics = [f'bcg_{i}' for i in range(
            len(specific_topics), len(specific_topics) + background_topic_number)]

        self._recipe = self.recipe_template.format(
            dataset_path=dataset_path,
            modality=modality,
            specific_topics=specific_topics,
            background_topics=background_topics,
        )
        return self._recipe
