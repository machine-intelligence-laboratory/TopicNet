class BaseRegularizer:
    """
    Base regularizer class to construct custom regularizers.

    """
    def __init__(self, name, tau, gamma=None):
        self.name = name
        self.tau = tau
        self.gamma = gamma

    def attach(self, model):
        """

        Parameters
        ----------
        model : ARTM model
            necessary to apply master component
        """
        self._model = model

    def grad(self, pwt, nwt):
        raise NotImplementedError('grad method should be overrided in an inherited class')
