import warnings

from enum import Enum
from numbers import Number
from typing import (
    List,
    Optional
)

from .base_score import BaseScore


class FrozenScore(BaseScore):
    """
    Custom scores can have anything inside.
    So there is a probability that pickle will not be able to dump them.
    Frozen score helps to store the value of the original score without its internal logic,
    so as it can be saved.
    """
    def __init__(self, value: List[Optional[float]], original_score: BaseScore = None):
        super().__init__()

        self.value = value
        self._original_score: BaseScore = None

        if original_score is not None:
            self._save_original(original_score)

    def __repr__(self):
        return f'{self.__class__.__name__}(original_score={self._original_score!r})'

    def __getattr__(self, attribute_name):
        if attribute_name.startswith('__'):
            raise AttributeError()

        if attribute_name == '_original_score':  # some dill-loading stuff?
            raise AttributeError()

        if self._original_score is not None and hasattr(self._original_score, attribute_name):
            return getattr(self._original_score, attribute_name)

        raise AttributeError(
            f'Frozen score doesn\'t have such attribute: "{attribute_name}"'
        )

    def update(self, score_value: float) -> None:
        """
        Update is not supposed to be applied to Frozen score.
        It is not supposed to be changed.
        Still, the situation with an endeavour to update can generally happen if one tries
        to train the model further after loading.
        """
        warnings.warn(
            f'Trying to update Frozen score! Update value "{score_value}". '
            f'Frozen score is not supposed to be updated, '
            f'as there is no computation logic inside'
        )

        if score_value is not None:
            # TODO: it shouldn't be possible to pass such score_value value to update()
            #  other than the one returned by self.call()
            warnings.warn(
                f'Can\'t update Frozen score with value other than None: "{score_value}"!'
                f' Saving None score'
            )

        self.value.append(None)

    def call(self, model, *args, **kwargs) -> Optional[float]:
        return None

    def _save_original(self, original_score: BaseScore) -> None:
        field_types_for_saving = (Number, str, bool, Enum)
        self._original_score = BaseScore()

        for field_name in dir(original_score):
            field_value = getattr(original_score, field_name)

            if field_value is not None and not isinstance(field_value, field_types_for_saving):
                continue

            try:
                setattr(self._original_score, field_name, field_value)
            except AttributeError:
                # TODO: log?
                pass
