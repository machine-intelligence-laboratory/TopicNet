from typing import List, Union, Dict
from .recipe_wrapper import BaseRecipe
from .. import Dataset

multimodal_search_template = '''
# This config modifies a strategy described in the article
# Multi-objective Topic Modeling for Exploratory Search in Tech News
# by Anastasya Yanina, Lev Golitsyn and Konstantin Vorontsov, Jan 2018


# Use .format_recipe(modality_list=modality_list, modality=modality,
# dataset_path=dataset_path, specific_topics=specific_topics,
# background_topics=background_topics, num_iter=num_iter)
# when loading the recipe to adjust for your dataset

topics:
# Describes number of model topics, in the actuall article 200 topics were found to be optimal
    specific_topics: {specific_topics}
    background_topics: {background_topics}

regularizers:
{syntesized_regularizers}
- SmoothSparseThetaRegularizer:
    name: sparse_theta
    topic_names: specific_topics
    tau: 1

model:
    dataset_path: {dataset_path}
    {modalities_description}
    main_modality: '{modality}'

stages:
{syntesized_stages}
'''

decorrelator_reg_template = '''
- DecorrelatorPhiRegularizer:
    name: decorrelation_phi_{modality}
    topic_names: specific_topics
    tau: 1
    class_ids: ['{modality}']
'''

sparse_phi_reg_template = '''
- SmoothSparsePhiRegularizer:
    name: smooth_phi_{modality}
    topic_names: specific_topics
    tau: 1
    class_ids: ['{modality}']
'''

sparse_theta_cube_template = '''
- RegularizersModifierCube:
    num_iter: {{num_iter}}
    reg_search: add
    regularizer_parameters:
        name: sparse_theta
    selection:
        - {0}
    strategy: PerplexityStrategy
    strategy_params:
        start_point: -0.3
        step: 0.01
        max_len: 20
    tracked_score_function: PerplexityScore@all
    verbose: false
    use_relative_coefficients: True
'''.format('PerplexityScore@all < 1.01 * MINIMUM(PerplexityScore@all)' +
           ' and SparsityThetaScore -> max')

# Had to change tracked score function. Is it fine?
decor_phi_cube_template = '''
- RegularizersModifierCube:
    num_iter: {{num_iter}}
    reg_search: add
    regularizer_parameters:
        name: decorrelation_phi_{{modality}}
    selection:
        - {0}
    strategy: PerplexityStrategy
    strategy_params:
        start_point: 0.005
        step: 0.005
        max_len: 10
    tracked_score_function: PerplexityScore{{modality}}
    verbose: false
    use_relative_coefficients: True
'''.format('PerplexityScore{modality} < ' +
           '1.01 * MINIMUM(PerplexityScore{modality})' +
           ' and SparsityPhiScore{modality} -> max')

smooth_phi_cube_template = '''
- RegularizersModifierCube:
    num_iter: {{num_iter}}
    reg_search: add
    regularizer_parameters:
        name: smooth_phi_{{modality}}
    selection:
        - {0}
    strategy: PerplexityStrategy
    strategy_params:
        start_point: 0.0
        step: 0.02
        max_len: 20
    tracked_score_function: PerplexityScore{{modality}}
    verbose: false
    use_relative_coefficients: True
'''.format('PerplexityScore{modality} < ' +
           '1.01 * MINIMUM(PerplexityScore{modality})' +
           ' and SparsityPhiScore{modality} -> max')


class MultimodalSearchRecipe(BaseRecipe):
    """
    Class for multimodal search recipe creation and
    unification of recipe usage interface
    """
    def __init__(self, order='extended_modalities'):
        """
        Parameters
        ----------

        order : str
            can be 'extended_modalities' or 'repeated_default'
            where 'repeated_default' repeats the original recipe
            for each dataset modality
            while 'extended_modalities' extends only modality-reliant
            blocks of training keeping last part equivalent to the original pipeline
        """
        super().__init__(recipe_template=multimodal_search_template)
        self._order = order

    def format_recipe(
        self,
        dataset_path: str,
        modality_list: List[str] or Dict = None,
        main_modality: str = None,
        topic_number: int = 20,
        background_topic_number: int = 1,
        num_iter: Union[int, List[int]] = 20,
    ):
        '''
        Creates a recipe for multimodal search
        using basic template at the top of this file

        Parameters
        ----------
        dataset_path : path to the data
        main_modality : str
            chosen to be main modality from modality list, if possible
            if it is not specified, the function attempts to user
            the first entry of `modality_list` instead

        modality_list : list of modality names to use
                        or a dict specifying the (relative) weight of each
        topic_number:
            number of the model topics
        background_topic_number :
            number of background topics
        num_iter :
            specifying number of iterations for each cube

        Returns
        -------
        string specifying recipe for multimodal search
        '''

        if modality_list is None:
            modality_list = list(Dataset(dataset_path).get_possible_modalities())

        specific_topics = [f'topic_{i}' for i in range(topic_number)]
        background_topics = [f'bcg_{i}' for i in range(
            len(specific_topics), len(specific_topics) + background_topic_number)]

        if main_modality is None:
            if isinstance(modality_list, list):
                main_modality = modality_list[0]
            else:
                raise TypeError("main_modality should be specified")
        self._make_multimodal_recipe(
            modality=main_modality,
            dataset_path=dataset_path,
            specific_topics=specific_topics,
            background_topics=background_topics,
            modality_list=modality_list,
            num_iter=num_iter,
        )
        return self._recipe

    def _form_regularizers(self, modality_list: List[str]):
        '''
        Creates regularizer configs for each
        modality following templates deufined above

        Parameters
        ----------
        modality_list : list of str
            list with modality names

        Returns
        -------

        string with configs for all needed regularizers
        '''
        regularizer_templates = []
        for modality in modality_list:
            regularizer_templates.append(decorrelator_reg_template.format(modality=modality))
            regularizer_templates.append(sparse_phi_reg_template.format(modality=modality))
        return ''.join(regularizer_templates)

    def _form_and_order_cubes(
            self,
            modality_list: List[str],
            num_iter: int = 20,
    ):
        '''
        Creates cube configs for each modality
        following cube templates defined above

        Parameters
        ----------
        modality_list : list of str
            list with modality names
        num_iter : number or list of numbers
            specifying number of iterations for each cube

        Returns
        -------
        string ordering cube templates for recipe
        '''
        if isinstance(num_iter, int):
            num_iter = [num_iter] * (len(modality_list) + 1)
        cube_templates = []
        for modality, iterations in zip(modality_list, num_iter):
            if self._order == 'extended_modalities':
                cube_templates.append(decor_phi_cube_template.format(modality=modality,
                                                                     num_iter=iterations))
                cube_templates.append(smooth_phi_cube_template.format(modality=modality,
                                                                      num_iter=iterations))
            elif self._order == 'repeated_default':
                cube_templates.append(decor_phi_cube_template.format(modality=modality,
                                                                     num_iter=iterations))
                cube_templates.append(smooth_phi_cube_template.format(modality=modality,
                                                                      num_iter=iterations))
                cube_templates.append(sparse_theta_cube_template.format(num_iter=iterations))
            else:
                raise ValueError('That option is not availiable')
        if self._order == 'extended_modalities':
            iterations = num_iter[-1]
            cube_templates.append(sparse_theta_cube_template.format(num_iter=iterations))
        return ''.join(cube_templates)

    def _make_multimodal_recipe(
            self,
            dataset_path: str,
            modality: str,
            specific_topics: List[str],
            background_topics: List[str],
            modality_list: List[str] or Dict = None,
            background_topic_number: int = 1,
            num_iter: Union[int, List[int]] = 20,
    ):
        reg_forms = self._form_regularizers(modality_list)
        cube_forms = self._form_and_order_cubes(
            modality_list,
            num_iter=num_iter,)
        if isinstance(modality_list, list):
            modalities_description = f"modalities_to_use: {modality_list}"
        elif isinstance(modality_list, dict):
            # this line has correct whitespace count
            header_string = "modalities_weights:"
            # these ones should be indented one level more, so 8 spaces
            data_strings = [f"'{k}': {v}" for k, v in modality_list.items()]
            strings = [header_string] + data_strings
            modalities_description = "\n        ".join(strings)
        else:
            raise TypeError("modality_list should be either list or dict, not {type(modality_list}")
        self._recipe = self.recipe_template.format(
            modality=modality,
            dataset_path=dataset_path,
            specific_topics=specific_topics,
            background_topics=background_topics,
            modalities_description=modalities_description,
            syntesized_regularizers=reg_forms,
            syntesized_stages=cube_forms)
