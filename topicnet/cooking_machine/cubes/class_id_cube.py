from .base_cube import BaseCube


class ClassIdModifierCube(BaseCube):
    """
    Allows to create cubes of training and apply them to a topic model.

    """
    def __init__(self, num_iter, class_id_parameters, reg_search,
                 strategy=None, tracked_score_function=None, verbose=False):
        """
        Initialize stage. Checks params and update internal attributes.

        Parameters
        ----------
        num_iter : str or int
            number of iterations or method
        class_id_parameters : dict of list
            regularizers params
        reg_search : str
            "grid" or "pair". "pair" for elementwise grid search in the case
            of several regularizers, "grid" for the fullgrid search in the
            case of several regularizers
        strategy : BaseStrategy
            optimization approach (Defatult value = None)
        tracked_score_function : retrieve_score_for_strategy
            optimizable function for strategy (Defatult value = None)
        verbose : bool
            visualization flag (Defatult value = False)

        """
        super().__init__(num_iter=num_iter, action='class_id_modifier',
                         strategy=strategy,
                         tracked_score_function=tracked_score_function,
                         verbose=verbose)

        self.reg_search = reg_search
        self._add_class_ids(class_id_parameters)

    def _check_all_class_id_parameters(self, class_id_parameters):
        """
        Checks and updates params of all class_id grids. Inplace.

        Parameters
        ----------
        class_id_parameters : dict

        Returns
        -------

        """
        if not isinstance(class_id_parameters, dict):
            raise TypeError('class_id_parameters muse be dict')

        for class_id_name, class_id_coefficients in class_id_parameters.items():
            if not isinstance(class_id_coefficients, list):
                raise TypeError('class_id_coefficients must be list')

        if self.reg_search == "pair":
            grid_size = len(next(iter(class_id_parameters.values())))
            for class_id_coefficients in class_id_parameters.values():
                if len(class_id_coefficients) != grid_size:
                    raise ValueError("Grid size is not the same.")

    def _add_class_ids(self, all_class_id_parameters):
        """

        Parameters
        ----------
        all_class_id_parameters : dict

        Returns
        -------

        """
        self.raw_parameters = all_class_id_parameters
        self._check_all_class_id_parameters(all_class_id_parameters)
        self.parameters = [
            {"object": "", "field": class_id_name, "values": class_id_coefficients}
            for class_id_name, class_id_coefficients
            in all_class_id_parameters.items()
        ]

    def apply(self, topic_model, one_model_parameter, dictionary=None):
        """

        Parameters
        ----------
        topic_model : TopicModel
        one_model_parameter : list or tuple
        dictionary : Dictionary
             (Default value = None)

        Returns
        -------

        """
        new_model = topic_model.clone()
        new_model.parent_model_id = topic_model.model_id

        for _, class_id_name, class_id_coefficient in one_model_parameter:
            new_model.class_ids[class_id_name] = class_id_coefficient

        return new_model

    def get_jsonable_from_parameters(self):
        """ """
        return [self.raw_parameters]
