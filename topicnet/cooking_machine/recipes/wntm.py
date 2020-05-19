from typing import List, Tuple

from .recipe_wrapper import BaseRecipe
from ..config_parser import parse

from .. import Dataset
from .. import DatasetCooc
from .. import Experiment

WNTM_template = '''
# This config follows a pipline for training a Word Net Topic Model
# https://link.springer.com/article/10.1007/s10115-015-0882-z


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
        tau: 0.2

scores:
    - BleiLaffertyScore:
        num_top_tokens: 30
model:
    dataset_path: {dataset_path}
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
        step: 0.001
        max_len: 50
    tracked_score_function: PerplexityScore@all
    verbose: false
    use_relative_coefficients: true
'''


class WNTMRecipe(BaseRecipe):
    """
    Class for baseline recipe creation and
    unification of recipe interface
    """
    def __init__(self):
        super().__init__(recipe_template=WNTM_template)

    def format_recipe(
        self,
        dataset_path: str,
        modality_list: List[str] = None,
        main_modality: str = None,
        topic_number: int = 20,
        background_topic_number: int = 0,
        num_iter: int = 20,
    ):
        self.dataset_path = dataset_path

        if modality_list is None:
            modality_list = list(Dataset(dataset_path).get_possible_modalities())

        if main_modality is None:
            main_modality = modality_list[0]

        specific_topics = [f'topic_{i}' for i in range(topic_number)]
        background_topics = [f'bcg_{i}' for i in range(
            len(specific_topics), len(specific_topics) + background_topic_number)]

        self._recipe = self.recipe_template.format(
            dataset_path=dataset_path,
            modality_list=modality_list,
            main_modality=main_modality,
            specific_topics=specific_topics,
            background_topics=background_topics,
        )
        return self._recipe

    def build_experiment_environment(
            self,
            save_path: str,
            experiment_id: str = 'default_experiment_name',
            force_separate_thread: bool = False
    ) -> Tuple[Experiment, Dataset]:
        """
        Returns experiment and dataset instances
        needed to perform the hyperparameter tuning on the data
        according to recipe

        Parameters
        ----------
        save_path: path to the folder to save experiment logs and models
        experiment_id: name of the experiment folder
        force_separate_thread: train each model in dedicated process
            this feature helps to handle resources in Jupyter notebooks

        """
        if self._recipe is None:
            raise ValueError(
                'Recipe missing data specific parameters. '
                'Provide them with "format_recipe" method!')

        settings, regs, model, dataset = parse(
            self._recipe,
            force_separate_thread=force_separate_thread,
            dataset_class=DatasetCooc
        )
        # TODO: handle dynamic addition of regularizers
        experiment = Experiment(experiment_id=experiment_id, save_path=save_path, topic_model=model)
        experiment.build(settings)
        return experiment, dataset
