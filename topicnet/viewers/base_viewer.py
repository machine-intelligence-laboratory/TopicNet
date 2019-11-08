from ..cooking_machine.models import BaseModel


class BaseViewer:
    """ """
    def __init__(self, model):
        if not isinstance(model, BaseModel):
            raise TypeError('Parameter "model" should derive from BaseModel')

        self._model = model

    @property
    def model(self):
        """ """
        return self._model

    def view(self, *args, **kwargs):
        """
        Main method of viewer.

        Returns
        -------
        optional

        """
        raise NotImplementedError('Should be implemented in subclass')
