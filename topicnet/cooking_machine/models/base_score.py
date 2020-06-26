import dill

from typing import (
    Any,
    Callable,
    Dict,
)

from . import scores as tn_scores


class BaseScore:
    """
    Base Class to construct custom score functions.

    """
    _PRECOMPUTED_DATA_PARAMETER_NAME = 'precomputed_data'

    # TODO: name should not be optional
    def __init__(
            self,
            name: str = None,
            should_compute: Callable[[int], bool] or bool = None):
        """

        Parameters
        ----------
        name
            Name of the score
        should_compute
            Function which decides whether the score should be computed
            on the current fit iteration or not.
            If `should_compute` is `None`, then score is going to be computed on every iteration.
            At the same time, whatever function one defines,
            score is always computed on the last fit iteration.
            This is done for two reasons.
            Firstly, so that the score is always computed at least once during `model._fit()`.
            Secondly, so that `experiment.select()` works correctly.

            The parameter `should_compute` might be helpful
            if the score is slow but one still needs
            to get the dependence of the score on iteration
            (for the described case, one may compute the score
            on every even iteration or somehow else).
            However, be aware that if `should_compute` is used for some model's scores,
            then the scores may have different number of values in `model.scores`!
            Number of score values is the number of times the scores was calculated;
            first value corresponds to the first fit iteration
            which passed `should_compute` etc.

            There are a couple of things also worth noting.
            Fit iteration numbering starts from zero.
            And every new `model._fit()` call is a new range of fit iterations.

        Examples
        --------
        Scores created below are unworkable (as BaseScore has no `call` method inplemented).
        These are just the examples of how one can create a score and set some of its parameters.

        Scores to be computed on every iteration:

        >>> score = BaseScore()
        >>> score = BaseScore(should_compute=BaseScore.compute_always)
        >>> score = BaseScore(should_compute=lambda i: True)
        >>> score = BaseScore(should_compute=True)

        Scores to be computed only on the last iteration:

        >>> score = BaseScore(should_compute=BaseScore.compute_on_last)
        >>> score = BaseScore(should_compute=lambda i: False)
        >>> score = BaseScore(should_compute=False)

        Score to be computed only on even iterations:

        >>> score = BaseScore(should_compute=lambda i: i % 2 == 0)
        """
        self._name = name

        if should_compute is None:
            should_compute = self.compute_always
        elif should_compute is True:
            should_compute = self.compute_always
        elif should_compute is False:
            should_compute = self.compute_on_last
        elif not isinstance(should_compute, type(lambda: None)):
            raise TypeError(f'Unknown type of `should_compute`: {type(should_compute)}!')
        else:
            pass

        self._should_compute = should_compute
        self.value = []

        if not hasattr(tn_scores, self.__class__.__name__):
            setattr(tn_scores, self.__class__.__name__, self.__class__)

    @staticmethod
    def compute_always(fit_iteration: int) -> bool:
        return True

    @staticmethod
    def compute_on_last(fit_iteration: int) -> bool:
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def save(self, path):
        with open(path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            score = dill.load(f)

        return score

    def update(self, score):
        """

        Parameters
        ----------
        score : float
            score value

        Returns
        -------

        """
        known_errors = (ValueError, TypeError)

        try:
            score = float(score)
        except known_errors:
            raise ValueError(f'Score call should return float but not {score}')

        self.value.append(score)

    def call(self, model, precomputed_data: Dict[str, Any] = None):
        """
        Call to custom score function.

        Parameters
        ----------
        model : TopicModel
            a TopicNet model inherited from BaseModel
        precomputed_data
            Data which scores may share between each other during *one fit iteration*.
            For example, if the model has several scores of the same score class,
            and there is a heavy time consuming computation inside this score class,
            it may be useful to perform the calculations *only once*, for one score instance,
            and then make the result visible for all other scores that might need it.

        Returns
        -------
        float
            score

        Notes
        -----
        Higher score not necessarily should correspond to better model.
        It is up to user to decide what the meaning is behind the score,
        and then use this logic in query in Experiment's `select()` method.

        If one need ARTM model for score (not TopicNet one), it is available as model._model

        When creating a custom score class,
        it is recommended to use `**kwargs` in the score's `call` method,
        so that all `BaseScore` optional parameters are also available
        in its successor score classes.

        Examples
        --------

        Score which uses `precomputed_data`:

        >>> import time
        ...
        >>> class NewScore(BaseScore):
        ...     def __init__(self, name: str, multiplier: float):
        ...         super().__init__(name=name)
        ...
        ...         self._multiplier = multiplier
        ...         self._heavy_value_name = 'time_consuming_value_name'
        ...
        ...     def call(self, model, precomputed_data = None):
        ...         if precomputed_data is None:
        ...             # Parameter `precomputed_data` is optional in BaseScore
        ...             # So this case also should be supported
        ...             heavy_value = self._compute_heavy(model)
        ...         elif self._heavy_value_name in precomputed_data:
        ...             # This is going to be fast
        ...             heavy_value = precomputed_data[self._heavy_value_name]
        ...         else:
        ...             # This is slow (but only one such call!)
        ...             heavy_value = self._compute_heavy(model)
        ...             precomputed_data[self._heavy_value_name] = heavy_value
        ...
        ...         return heavy_value * self._multiplier
        ...
        ...     def _compute_heavy(self, model):
        ...         time.sleep(100)  # just for demonstration
        ...
        ...         return 0
        """
        raise NotImplementedError('Define your score here')
