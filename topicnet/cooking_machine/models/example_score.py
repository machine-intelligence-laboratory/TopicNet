import numpy as np
from .base_score import BaseScore


class ScoreExample(BaseScore):
    """
    Example score that calculates
    average size of topic kernel across all topics.
    We inherit from BaseScore in order to have self.value property and self.update() method
    (the internal logic of TopicNet relies on them)

    """
    def __init__(self, token_threshold=1e-3):
        """

        Parameters
        ----------
        token_threshold : float
            what probabilities to take as token belonging to the topic

        """
        super().__init__()
        self.threshold = token_threshold

    def call(self, model):
        """
        Method that calculates the score

        Parameters
        ----------
        model : TopicModel

        Returns
        -------
        score : float
            mean kernel size for all topics in the model

        """
        phi = model.get_phi().values
        score = np.sum((phi > self.threshold).astype('int'), axis=0).mean()
        return score
