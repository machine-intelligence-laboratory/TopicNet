import dill
from . import scores as tn_scores


class BaseScore:
    """
    Base Class to construct custom score functions.

    """
    def __init__(self, name: str = None):  # TODO: name should not be optional
        """

        Parameters
        ----------
        name:
            Name of the score

        """
        self._name = name
        self.value = []

        if not hasattr(tn_scores, self.__class__.__name__):
            setattr(tn_scores, self.__class__.__name__, self.__class__)

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

    def call(self, model):
        """
        Call to custom score function.

        Parameters
        ----------
        model : TopicModel
            a TopicNet model inherited from BaseModel

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
        """
        raise NotImplementedError('Define your score here')
