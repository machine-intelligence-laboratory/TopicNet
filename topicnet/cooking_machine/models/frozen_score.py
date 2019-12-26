import warnings

from .base_score import BaseScore


class FrozenScore(BaseScore):
    """
    Custom scores can have anything inside.
    So there is a probability that pickle will not be able to dump them.
    Frozen score helps to store the value of the original score without its internal logic,
    so as it can be saved.
    """
    def __init__(self, value):
        super().__init__()

        self.value = value

    def update(self, score):
        """
        Update is not supposed to be applied to Frozen score.
        It is not supposed to be changed.
        Still, the situation with an endeavour to update can generally happen if one tries
        to train the model further after loading.
        """
        warnings.warn(
            f'Trying to update Frozen score! Update value "{score}". '
            f'Frozen score is not supposed to be updated, '
            f'as there is no computation logic inside'
        )

        if score is not None:
            # TODO: it shouldn't be possible to pass such score value to update()
            #  other than the one returned by self.call()
            warnings.warn(
                f'Can\'t update Frozen score with value other than None: "{score}"! '
                f'Saving None score'
            )

        self.value.append(None)

    def call(self, model, *args, **kwargs):
        return None
