from ..config_parser import build_experiment_environment_from_yaml_config

recipe_template_example = """
This string should be formatted as a confing in YAML format.
If you struggle making yours, look in other recipes for guidance.
Also cooking_machine/config_parser.py docstring
provides some insight on the matter.
{field_to_fill}
"""


class BaseRecipe:
    """
    Base class to work with recipes
    """
    def __init__(self, recipe_template):
        self.recipe_template = recipe_template
        self._recipe = None

    def __str__(self):
        if self._recipe:
            return self._recipe
        else:
            return self.recipe_template

    def format_recipe(self, *args, **kwargs) -> str:
        """
        Updates self._recipe variable
        with variables specific for the dataset
        """
        raise NotImplementedError(
            'Method needs to be specified for the recipe template'
        )
        return self._recipe

    def build_experiment_environment(
            self,
            save_path: str,
            experiment_id: str = 'default_experiment_name',
            force_separate_thread: bool = False
    ):
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

        Returns
        -------
        tuple experiment, dataset  instances of classes from topicnet
        """
        if self._recipe is None:
            raise ValueError(
                'Recipe missing data specific parameters. '
                'Provide them with "format_recipe" method!')

        return build_experiment_environment_from_yaml_config(
            self._recipe,
            save_path=save_path,
            experiment_id=experiment_id,
            force_separate_thread=force_separate_thread
        )
