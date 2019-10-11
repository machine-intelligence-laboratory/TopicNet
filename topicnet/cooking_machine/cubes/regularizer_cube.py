from .base_cube import BaseCube
from ..routine import transform_complex_entity_to_dict
from ..rel_toolbox_lite import count_vocab_size, transform_regularizer
from copy import deepcopy


class RegularizersModifierCube(BaseCube):
    """
    Allows to create cubes of training and apply them to a topic model.

    """
    def __init__(self, num_iter: int, regularizer_parameters,
                 reg_search='grid', relative_coefficients: bool = True, strategy=None,
                 tracked_score_function=None, verbose: bool = False):
        """
        Initialize stage. Checks params and update internal attributes.

        Parameters
        ----------
        num_iter : int
            number of iterations or method
        regularizer_parameters : list[dict] or dict
            regularizers params
        reg_search : str
            "grid", "pair", "add" or "mul". 
            "pair" for elementwise grid search in the case of several regularizers 
            "grid" for the fullgrid search in the case of several regularizers 
            "add" and "mul" for the ariphmetic and geometric progression
            respectively for PerplexityStrategy 
            (Default value = "grid")
        relative_coefficients : bool
            forces the regularizer coefficient to be in relative form
            i.e. normalized over collection properties
        strategy : BaseStrategy
            optimization approach (Default value = None)
        tracked_score_function : retrieve_score_for_strategy
            optimizable function for strategy (Default value = None)
        verbose : bool
            visualization flag (Default value = False)

        """  # noqa: W291
        super().__init__(num_iter=num_iter, action='reg_modifier',
                         reg_search=reg_search, strategy=strategy,
                         tracked_score_function=tracked_score_function, verbose=verbose)
        self._relative = relative_coefficients
        if isinstance(regularizer_parameters, dict):
            regularizer_parameters = [regularizer_parameters]
        self._add_regularizers(regularizer_parameters)

    def _check_all_regularizer_parameters(self, regularizer_parameters):
        """
        Checks and updates params of all regularizers. Inplace.

        Parameters
        ----------
        regularizer_parameters : list of dict

        """
        if len(regularizer_parameters) <= 0:
            raise ValueError("There is no parameters.")

        for i, one_regularizer_parameters in enumerate(regularizer_parameters):
            if not isinstance(one_regularizer_parameters, dict):
                wrong_type = type(one_regularizer_parameters)
                raise ValueError(f"One regularizer should be dict, not {wrong_type}")

        if self.reg_search == "pair":
            # TODO: infinite length support
            grid_size = len(regularizer_parameters[0]["tau_grid"])
            for one_regularizer_parameters in regularizer_parameters:
                if len(one_regularizer_parameters["tau_grid"]) != grid_size:
                    raise ValueError("Grid size is not the same.")

    def _add_regularizers(self, all_regularizer_parameters):
        """

        Parameters
        ----------
        all_regularizer_parameters : list of dict

        """
        self._check_all_regularizer_parameters(all_regularizer_parameters)
        self.raw_parameters = all_regularizer_parameters

        def _retrieve_object(params):
            """

            Parameters
            ----------
            params : dict

            Returns
            -------

            """
            if "regularizer" in params:
                return params["regularizer"]
            else:
                return {"name": params["name"]}

        self.parameters = [
            {
                "object": _retrieve_object(params),
                "field": "tau",
                "values": params.get('tau_grid', [])
            }
            for params in all_regularizer_parameters
        ]

    def _handle_regularizer(
        self,
        model,
        modalities,
        regularizer,
        regularizer_type,
    ):
        """
        Handles the case of various regularizers that
        contain 'Regularizer' in their name, namely all artm regularizers

        Parameters
        ----------
        model : TopicModel
            to be changed in place
        modalities : dict
            modalities used in the model
        regularizer : an instance of Regularizer from artm library
        regularizer_type : str
            type of regularizer to be added

        Returns
        -------
        None

        """

        fallback_options = (AttributeError, TypeError, AssertionError)
        try:
            n_topics = len(regularizer.topic_names)
            assert n_topics > 0
        except fallback_options:
            n_topics = len(model.topic_names)

        if self._relative and 'SmoothSparse' in regularizer_type:
            regularizer = transform_regularizer(
                self.data_stats,
                regularizer,
                modalities,
                n_topics,
            )

        model.regularizers.add(regularizer, overwrite=True)
        if 'Decorrelator' in regularizer_type:
            if self._relative:
                model.regularizers[regularizer.name].gamma = 0
            else:
                model.regularizers[regularizer.name].gamma = None

    def apply(self, topic_model, one_model_parameter, dictionary=None, model_id=None):
        """
        Applies regularizers and parameters to model

        Parameters
        ----------
        topic_model : TopicModel
        one_model_parameter : list or tuple
        dictionary : Dictionary
            (Default value = None)
        model_id : str
            (Default value = None)

        Returns
        -------
        TopicModel

        """
        new_model = topic_model.clone(model_id)
        new_model.parent_model_id = topic_model.model_id

        modalities = dict()
        self.data_stats = None
        if self._relative:
            modalities = new_model.class_ids
            if not getattr(self, 'data_stats', None):
                self.data_stats = count_vocab_size(dictionary, modalities)

        for regularizer_data in one_model_parameter:
            regularizer, field_name, params = regularizer_data
            regularizer_type = str(type(regularizer))
            if isinstance(regularizer, dict):
                if regularizer['name'] in new_model.regularizers.data:
                    setattr(new_model.regularizers[regularizer['name']],
                            field_name,
                            params)
                else:
                    error_msg = (f"Regularizer {regularizer['name']} does not exist. "
                                 f"Cannot be modified.")
                    raise ValueError(error_msg)
            elif 'Regularizer' in regularizer_type:
                new_regularizer = deepcopy(regularizer)
                new_regularizer._tau = params
                self._handle_regularizer(
                    new_model,
                    modalities,
                    new_regularizer,
                    regularizer_type,
                )
            else:
                error_msg = f"Regularizer instance or name must be specified for {regularizer}."
                raise ValueError(error_msg)
        return new_model

    def get_jsonable_from_parameters(self):
        """ """
        jsonable_parameters = []
        for one_model_parameters in self.raw_parameters:
            one_jsonable = {"tau_grid": one_model_parameters.get("tau_grid", [])}
            if "regularizer" in one_model_parameters:
                one_regularizer = one_model_parameters['regularizer']
                if not isinstance(one_regularizer, dict):
                    one_regularizer = transform_complex_entity_to_dict(one_regularizer)
                one_jsonable["regularizer"] = one_regularizer
            else:
                one_jsonable["name"] = one_model_parameters["name"]
            jsonable_parameters.append(one_jsonable)

        return jsonable_parameters
