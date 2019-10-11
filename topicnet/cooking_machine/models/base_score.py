class BaseScore:
    """
    Base Class to construct custom score functions.

    """
    def __init__(self, *args, **kwargs):
        self.value = []

    def update(self, score):
        """

        Parameters
        ----------
        score : float
            score value

        Returns
        -------

        """
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
