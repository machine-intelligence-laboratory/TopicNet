from itertools import product
from functools import reduce
from operator import mul


class BaseStrategy():
    """
    Allows to visit nodes of parameters' grid in a particular order.

    """
    def __init__(self):
        """
        Initialize stage. Checks params and update internal attributes.

        """
        self.score = []
        self.grid = []
        self.grid_len = None

    def _set_parameters(self, parameters):
        """

        Parameters
        ----------
        parameters : dict or list of dict

        """
        if isinstance(parameters, dict):
            parameters = [parameters]
        for entry in parameters:
            if any(key not in entry.keys() for key in ["field", "object", "values"]):
                raise ValueError(entry)
        self.parameters = parameters

    def prepare_grid(self, other_parameters, reg_search):
        """
        Creates grid for the search. Inplace.

        Parameters
        ----------
        other_parameters : dict or list of dict
        reg_search : str
            "grid" or "pair" (and "add" or "mul" for perplexity)

        """
        self._set_parameters(other_parameters)
        all_coeffs_grid = [
            [[params["object"], params["field"], one_value] for one_value in params["values"]]
            for params in self.parameters
        ]

        if reg_search == "grid":
            self.grid = product(*all_coeffs_grid)
            self.grid_len = reduce(mul, map(len, all_coeffs_grid), 1)
        elif reg_search == "pair":
            self.grid = zip(*all_coeffs_grid)
            self.grid_len = len(all_coeffs_grid[0])

    def grid_visit_generator(self, other_parameters, reg_search):
        """

        Parameters
        ----------
        other_parameters : dict or list of dict
        reg_search : str

        Yields
        ------
        list or tuple
            one parameters set for model

        """
        for one_model_values in self.grid:
            yield one_model_values

    def update_scores(self, new_value):
        """

        Parameters
        ----------
        new_value : float

        """
        self.score.append(new_value)
