from .base_cube import BaseCube
from inspect import signature
from copy import deepcopy
import warnings


class CubeCreator(BaseCube):
    """
    Class for creating models with different initial parameters.

    """
    DEFAULT_SEED_VALUE = 4

    def __init__(self, model, num_iter, parameters, reg_search,
                 second_level=False, tracked_score_function=None, verbose=False):
        """

        Parameters
        ----------
        model : TopicModel
            TopicModel instance
        num_iter : str or int
            number of iterations or method
        parameters : list[dict] or dict
            parameters for model initialization
        reg_search: str
            "grid" or "pair"
        second_level : bool
            if this cube is a second model level (Defatult value = False)
        tracked_score_function : retrieve_score_for_strategy
            optimizable function for strategy (Defatult value = None)
        verbose : bool
            visualization flag (Defatult value = False)

        """
        if second_level:
            action = 'HIER: LEVEL 2'
        else:
            action = 'INIT + TRAIN'
        super().__init__(num_iter=num_iter, action=action,
                         tracked_score_function=tracked_score_function,
                         reg_search=reg_search, verbose=verbose)

        if isinstance(parameters, dict):
            parameters = [parameters]
        self.raw_parameters = parameters
        self._model = model
        self._model_class = model.__class__

        param_set = [dictionary['name'] for dictionary in parameters]
        topic_related = set(['topic_names', 'num_topics']) & set(param_set)
        not_include = ['topic_names', ] if len(topic_related) > 0 else list()

        self._model_init_parameters = model.get_init_parameters(not_include=not_include)
        self._second_level = second_level
        self._check_all_parameters(parameters)
        self._prepare_models_parameters(parameters)

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
            if params['name'] != 'class_ids':
                self.parameters.append({
                    "object": "",
                    "field": params["name"],
                    "values": params["values"]
                })
            else:
                for modality_name, modality_values in params['values'].items():
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
        jsonable_parameters = dict(self._model_init_parameters)

        for one_parameter in self.raw_parameters:
            jsonable_values = []
            for parameter in one_parameter['values']:
                jsonable_values.append(str(parameter))
            jsonable_parameters[one_parameter['name']] = jsonable_values

        if self._second_level:
            jsonable_parameters['additional_info'] = 'hierarchical: Second level.'

        jsonable_parameters['version'] = self._model._model.library_version
        return [jsonable_parameters]

    def apply(self, topic_model, one_cube_parameter, dictionary=None):
        """

        Parameters
        ----------
        topic_model : TopicModel
        one_cube_parameter : list or tuple
        dictionary : Dictionary
             (Default value = None)

        Returns
        -------

        """
        new_model_parameters = deepcopy(self._model_init_parameters)
        if self._model:
            topic_model = self._model
        for parameter_entry in one_cube_parameter:
            _, parameter_name, parameter_value = parameter_entry
            if parameter_name[0] == '@':
                new_model_parameters['class_ids'][parameter_name] = parameter_value
            else:
                new_model_parameters[parameter_name] = parameter_value
        experiment = topic_model.experiment

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

        new_model_parameters['scores'] = list(self._model._model._scores._data.values())
        new_model_parameters['dictionary'] = dictionary

        new_model = self._model_class(experiment=experiment,
                                      parent_model_id=parent_model_id,
                                      description=description,
                                      custom_scores=deepcopy(self._model.custom_scores),
                                      **new_model_parameters)
        for reg_name, reg in self._model.regularizers.data.items():
            new_model.regularizers.add(deepcopy(reg))
        return new_model
