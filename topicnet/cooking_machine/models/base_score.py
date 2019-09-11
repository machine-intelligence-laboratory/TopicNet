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
        Call to custom score function

        Parameters
        ----------
        model : TopicModel
            a TopicNet model inherited from BaseModel

        Returns
        -------
        float
            score

        """
        raise NotImplementedError('define your score here')
