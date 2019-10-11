import numpy as np
from .strategy import BaseStrategy


class GreedyStrategy(BaseStrategy):
    """
    Allows to visit nodes of parameters' grid in a particular order.

    The rough idea:  
        We are given grid of (values1 x values2 x values3).  
        This strategy will find best value among points of form [v1, 0, 0]
        and will mark first coordinate as finished.  
        Then we search for best v2 among [v1, v2, 0].  
        Then [v1, v2, v3] etc.

    """  # noqa: W291
    def __init__(self, renormalize: bool = False):
        """
        Initialize stage. Updates internal attributes.

        """
        self.score = []
        self.best_point = None
        self.grid_len = None
        self.renormalize = renormalize

    def _check_parameters(self, parameters):
        """

        Parameters
        ----------
        parameters : optional

        """
        # TODO: check that [0, 1]
        # increasing
        # at least 2

        # or maybe its not range but an interval..?
        pass

    def _set_parameters(self, parameters):
        """
        Sets the parameters describing search space
        with some rudimentary sanity checking.

        Parameters
        ----------
        parameters : dict or list of dict

        Returns
        -------

        """
        if isinstance(parameters, dict):
            parameters = [parameters]
        for entry in parameters:
            if any(key not in entry.keys() for key in ["field", "object", "values"]):
                raise ValueError(entry)
        self.parameters = parameters

    def _get_strategy_parameters(self, saveable_only=False):
        """
        """
        strategy_parameters = {
            "score": self.score,
            "best_score": self.best_score,
            "best_point": self.best_point,
            "grid_len": self.grid_len
        }

        if not saveable_only and hasattr(self, "parameters"):
            strategy_parameters["parameters"] = self.parameters

        return strategy_parameters

    def _convert_return_value(self, processed_coordinates, found_values):
        """
        Converts the search point given to the internal format
        Notably, pads with zero and (optionally) normalizes

        Parameters
        ----------
        processed_coordinates : list of str
            names of the coordinates we already visited
        found_values : list of float
            coordinates of locally best points we already found

        Returns
        -------
        list of lists
            internal lists contain coefficients of classes

        """
        processed_coordinates = list(processed_coordinates)
        found_values = list(found_values)
        for params in self.parameters:
            class_name = params["field"]
            if class_name not in processed_coordinates:
                processed_coordinates.append(class_name)
                found_values.append(0)

        if self.renormalize:
            found_values = np.asarray(found_values) / sum(found_values)
        return [
            [params["object"], class_name, class_id_coefficient]
            for class_name, class_id_coefficient in zip(processed_coordinates, found_values)
        ]

    def prepare_grid(self, other_parameters, reg_search):
        """
        Sets parameters of grid and prepares grid length for verbosity.

        Parameters
        ----------
        other_parameters : dict or list of dict
        reg_search : str

        """
        self._set_parameters(other_parameters)
        self.grid_len = sum(map(lambda x: len(x['values']), self.parameters[1:]), 1)

    def _iterate_over_line(self, params, processed_coordinates, found_values):
        processed_coordinates.append(params["field"])
        found_values.append(0)
        cur_scores = []
        for value in params["values"]:
            found_values[-1] = value
            yield self._convert_return_value(processed_coordinates, found_values)
            cur_scores.append(self.score[-1])
        cur_scores = np.asarray(cur_scores)
        best_index = cur_scores.argmax()
        best_value = params["values"][best_index]
        found_values[-1] = best_value

    def grid_visit_generator(self, other_parameters, reg_search):
        """
        Converts the search point given to the internal format
        Notably, pads with zero and normalizees
        with some rudimentary sanity checking.

        Parameters
        ----------
        other_parameters : dict or list of dict
        reg_search : str

        Yields
        -------
        list of lists

        """
        if reg_search != "grid":
            raise TypeError("currently only 'grid' search type is supported")

        processed_coordinates = []
        found_values = []
        for params in self.parameters:
            if not processed_coordinates:
                if self.renormalize:
                    processed_coordinates.append(params["field"])
                    found_values.append(1)
                    yield self._convert_return_value(processed_coordinates, found_values)
                else:
                    yield from self._iterate_over_line(params, processed_coordinates, found_values)
            else:
                yield from self._iterate_over_line(params, processed_coordinates, found_values)
        self.best_point = self._convert_return_value(processed_coordinates, found_values)
        self.best_score = found_values[-1]
