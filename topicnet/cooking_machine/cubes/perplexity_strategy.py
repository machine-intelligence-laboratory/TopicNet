import numpy as np
from itertools import product
import warnings

from .strategy import BaseStrategy


class PerplexityStrategy(BaseStrategy):
    """
    Search for the best perplexity score.

    """
    def __init__(self, start_point: float = None, step: float = None,
                 max_len: float = 25, threshold: float = 1.05):
        """
        Initialize stage.

        Parameters
        ----------
        start_point : float
           first point for tau progression
        step : float
            step in tau progression
        max_len : int
            length of progression
        threshold : float
            threshold for "perplexity out of control"

        """
        self.score = []
        self.threshold = threshold
        self.best_point = None
        self.last_point = None
        self.grid = None
        self.grid_len = None

        self.start_point = start_point
        self.step = step
        self.max_len = max_len

    def _set_parameters(self, parameters):
        """

        Parameters
        ----------
        parameters : dict

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
            "threshold": self.threshold,
            "grid_len": self.grid_len,
            "start_point": self.start_point,
            "step": self.step,
            "max_len": self.max_len
        }

        if not saveable_only:
            strategy_parameters["best_point"] = self.best_point
            strategy_parameters["last_point"] = self.last_point
            strategy_parameters["grid"] = self.grid
            if hasattr(self, "parameters"):
                strategy_parameters["parameters"] = self.parameters
        else:
            strategy_parameters["best_point"] = self.best_point[0][-1]
            strategy_parameters["last_point"] = self.last_point[0][-1]

        return strategy_parameters

    def _set_strategy_parameters(self, strategy_parameters):
        """
        """
        if not isinstance(strategy_parameters, dict):
            raise ValueError("Input parameters must be dict.")

        for parameter_name in strategy_parameters.keys():
            if parameter_name in ["best_point", "last_point"]:
                if isinstance(strategy_parameters[parameter_name], (int, float)):
                    setattr(self, parameter_name,
                            self._implement_tau(strategy_parameters[parameter_name]))
            else:
                setattr(self, parameter_name, strategy_parameters[parameter_name])

    def _implement_tau(self, tau):
        """
        Converts the tau value given to the internal format.

        Parameters
        ----------
        tau : float

        Returns
        -------
        tuple

        """
        return ([self.parameters[0]["object"], self.parameters[0]["field"], tau],)

    def _endless_generator(self, mode):
        """

        Parameters
        ----------
        mode : str
            "add" or "mul"

        Yields
        ------
        float in internal format

        """
        yield self._implement_tau(0)

        start_point = self.start_point
        step = self.step

        if mode == "add":
            if step == 0:
                warnings.warn('The hyperparameter search space is limited to one point',
                              UserWarning)
            while True:
                yield self._implement_tau(start_point)
                start_point += step
        else:
            if start_point == 0:
                raise ValueError("Invalid start point {} for mul strategy".format(start_point))
            if step <= 1:
                raise ValueError("Invalid step {} for mul strategy".format(step))
            while True:
                yield self._implement_tau(start_point)
                start_point *= step

    def prepare_grid(self, other_parameters, reg_search="add"):
        """
        Creates search space and length for tqdm.
        Note, that first point in sequence is always 0.

        Parameters
        ----------
        other_parameters : dict or list of dict
            the parameters describing search space. This is a list of entries like
            {"object": "smoothSparsePhi", "field": "tau", "values": []}
        reg_search : str
            "grid", "add" or "mul"
            defines grid search or arithmetic or geometric progression

        """
        self.score = []
        if self.start_point and self.step:
            if reg_search == "grid":
                warnings.warn(f"Grid would be used "
                              f"instead of start point {self.start_point} and step {self.step}")
            elif reg_search not in ["add", "mul"]:
                raise TypeError("Invalid search type")

        self._set_parameters(other_parameters)
        if reg_search == "grid":
            self.parameters[0]["values"] = [0] + self.parameters[0]["values"]
        all_coeffs_grid = [
            [[params["object"], params["field"], one_value] for one_value in params["values"]]
            for params in self.parameters
        ]

        if reg_search != "grid" and self.start_point is not None and self.step is not None:
            self.grid = self._endless_generator(reg_search)
        elif reg_search == "grid":
            self.grid = product(*all_coeffs_grid)
            self.grid_len = len(all_coeffs_grid[0])
        if self.grid is None:
            raise ValueError(f'Failed to initialize self.grid, check initial parameters.')

    def grid_visit_generator(self, other_parameters, reg_search):
        """
        Yields points from search space with sanity checking of current result.

        Parameters
        ----------
        other_parameters : dict

        reg_search : str
            "add", "mul" or "grid"

        Yields
        ------
        sequence of points in search space

        """
        for one_model_values in self.grid:
            yield one_model_values

            if reg_search != "grid":
                self.parameters[0]["values"].append(one_model_values[0][2])
                if self.score[-1] / max(self.score[0], 1e-5) > self.threshold:
                    warnings.warn(f"Perplexity is too high for threshold {self.threshold}")
                    break

            if len(self.score) > 4 and len(set(self.score[:-6:-1])) == 1:
                warnings.warn("Last five scores are equal, interrupting search")
                break
            if len(self.score) > self.max_len:
                warnings.warn("Max progression length exceeded")
                break

        best_tau = self.parameters[0]["values"][1:][np.argmin(self.score[1:])]
        self.best_point = self._implement_tau(best_tau)
        self.last_point = self._implement_tau(self.parameters[0]["values"][-1])

    def update_scores(self, new_value):
        """

        Parameters
        ----------
        new_value : float

        """
        if isinstance(new_value, list):
            new_value = new_value[0]
        self.score.append(new_value)
