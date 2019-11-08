from .base_cube import BaseCube
from inspect import signature
from copy import deepcopy
import warnings


class CubeCreator(BaseCube):
    """
    Class for creating models with different initial parameters.

    """
    DEFAULT_SEED_VALUE = 4

    def __init__(self, num_iter: int, parameters, reg_search="grid", strategy=None,
                 model_class='TopicModel', second_level=False,
                 tracked_score_function=None, verbose=False, separate_thread=True):
        """

        Parameters
        ----------
        model : TopicModel
            TopicModel instance
        num_iter : int
            number of iterations or method
        parameters : list[dict] or dict
            parameters for model initialization
        reg_search: str
            "grid" or "pair"
        strategy : BaseStrategy
            optimization approach (Default value = None)
        second_level : bool
            if this cube is a second model level (Default value = False)
        tracked_score_function : retrieve_score_for_strategy
            optimizable function for strategy (Default value = None)
        verbose : bool
            visualization flag (Default value = False)
        separate_thread : bool
            will train models inside a separate thread if True

        """
        import topicnet.cooking_machine.models as tnmodels

        if second_level:
            action = 'HIER: LEVEL 2'
        else:
            action = 'INIT + TRAIN'
        super().__init__(num_iter=num_iter, action=action, strategy=strategy,
                         tracked_score_function=tracked_score_function,
                         reg_search=reg_search, verbose=verbose, separate_thread=separate_thread)

        if isinstance(parameters, dict):
            parameters = [parameters]
        parameters = self._preprocess_parameters(parameters)
        self._raw_parameters = parameters

        try:
            if model_class == 'TopicModel':
                model = getattr(tnmodels, model_class)(num_topics=-1)
            else:
                model = getattr(tnmodels, model_class)()
        except AttributeError:
            raise AttributeError('This model is not implemented')

        self._model_class = model.__class__
        self._library_version = model._model.library_version

        param_set = [dictionary['name'] for dictionary in parameters]
        topic_related = set(['topic_names', 'num_topics']) & set(param_set)
        not_include = ['topic_names', ] if len(topic_related) > 0 else list()
        self._not_include = not_include

        self._second_level = second_level
        self._check_all_parameters(parameters)
        self._prepare_models_parameters(parameters)

    def _preprocess_parameters(self, parameters):
        clean_parameters = []
        for params in parameters:
            if "name" in params:
                clean_parameters.append(params)
            else:
                for (name, values) in params.items():
                    new_params = {"name": name, "values": values}
                    clean_parameters.append(new_params)
        return clean_parameters

    def _check_all_parameters(self, parameters):
        """
        Checks input parameters.

        Parameters
        ----------
        parameters : dict

        Returns
        -------

        """
        if len(parameters) <= 0:
            raise ValueError("There are no parameters.")

        possible_init_params = list(signature(self._model_class.__init__).
                                    parameters.keys())[1:]
        is_args_or_kwargs = ('kwargs' in possible_init_params) or ('args' in possible_init_params)
        for parameter in parameters:
            if not isinstance(parameter, dict):
                wrong_type = type(parameter)
                raise ValueError(f"Parameter should be dict, not {wrong_type}")
            if not is_args_or_kwargs and parameter['name'] not in possible_init_params:
                raise ValueError(
                    f"There is no parameter {parameter['name']} in {self._model_class}"
                )

        if self.reg_search == "pair":
            grid_size = len(parameters[0]["values"])
            for parameter in parameters:
                if len(parameter["values"]) != grid_size:
                    raise ValueError("Grid size is not the same.")

    def _prepare_models_parameters(self, parameters):
        """

        Parameters
        ----------
        parameters : dict

        Returns
        -------

        """
        self.parameters = []
        for params in parameters:
            name = params['name']
            if not name.startswith('class_ids'):
                self.parameters.append({
                    "object": "",
                    "field": params["name"],
                    "values": params["values"]
                })
            else:
                if name == "class_ids":
                    new_params = params
                else:
                    _, class_id = name.split("class_ids")

                    weights = [float(w) for w in params["values"]]
                    new_params = {
                        "name": "class_ids",
                        "values": {class_id: weights}
                    }

                for modality_name, modality_values in new_params['values'].items():
                    if modality_name[0] == '@':
                        self.parameters.append({
                            "object": "",
                            "field": modality_name,
                            "values": modality_values
                        })
                    else:
                        warnings.warn(f'Unexpected parameter {modality_name} was encountered.')

    def get_jsonable_from_parameters(self):
        """ """
        jsonable_parameters = dict()

        for one_parameter in self._raw_parameters:
            jsonable_values = []
            for parameter in one_parameter['values']:
                jsonable_values.append(str(parameter))
            jsonable_parameters[one_parameter['name']] = jsonable_values

        if self._second_level:
            jsonable_parameters['additional_info'] = 'hierarchical: Second level.'

        try:
            jsonable_parameters['version'] = self._library_version
        except AttributeError:
            jsonable_parameters['version'] = "undefined"
        return [jsonable_parameters]

    def apply(self, topic_model, one_cube_parameter, dictionary=None, model_id=None):
        """

        Parameters
        ----------
        topic_model : TopicModel
        one_cube_parameter : list or tuple
        dictionary : Dictionary
            (Default value = None)
        model_id : str
            (Default value = None)

        Returns
        -------

        """
        new_model_parameters = deepcopy(
            topic_model.get_init_parameters(not_include=self._not_include)
        )
        for parameter_entry in one_cube_parameter:
            _, parameter_name, parameter_value = parameter_entry
            if parameter_name[0] == '@':
                new_model_parameters['class_ids'][parameter_name] = parameter_value
            else:
                new_model_parameters[parameter_name] = parameter_value
        experiment = topic_model.experiment
        model_class = topic_model.__class__
        if self._second_level:
            new_model_parameters['parent_model'] = topic_model._model
            if new_model_parameters.get('seed', -1) == -1:
                # for some reason, for the second level you need to specify seed
                new_model_parameters['seed'] = self.DEFAULT_SEED_VALUE
            # for the tree
            parent_model_id = topic_model.model_id
            description = list(topic_model.description)
        else:
            parent_model_id = experiment.tree.tree['model_id']
            description = None

        new_model_parameters['dictionary'] = dictionary
        new_model = model_class(
            experiment=experiment,
            model_id=model_id,
            parent_model_id=parent_model_id,
            description=description,
            custom_scores=deepcopy(topic_model.custom_scores),
            **new_model_parameters
        )
        for reg_name, reg in topic_model._model.regularizers.data.items():
            new_model._model.regularizers.add(deepcopy(reg))
        for score_name, score in topic_model._model._scores.data.items():
            new_model._model.scores.add(deepcopy(score))
        return new_model
