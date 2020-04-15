import artm
import copy
from collections.abc import Mapping

from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from .base_score import BaseScore
from .frozen_score import FrozenScore


class ScoresWrapper(Mapping):
    def __init__(self,
                 topicnet_scores: Dict[str, BaseScore],
                 artm_scores: artm.scores.Scores):

        self._topicnet_scores = topicnet_scores
        self._artm_scores = artm_scores

        # returned by model.score, reset by model._fit
        self._score_caches: Optional[Dict[str, List[float]]] = None

    @property
    def _scores(self) -> Dict[str, List[float]]:
        assert self._score_caches is not None  # maybe empty dict, but not None

        return self._score_caches

    def _reset_score_caches(self):
        self._score_caches = None

    def __getitem__(self, key):
        return self._scores[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        raise RuntimeError('Use `model.scores.add()` method!')

    def __delitem__(self, key):
        raise RuntimeError('Not possible to delete model score!')

    def __iter__(self):
        return iter(self._scores)

    def __len__(self):
        return len(self._scores)

    def __keytransform__(self, key):
        return key

    def add(self, score: Union[BaseScore, artm.scores.BaseScore]):
        if isinstance(score, FrozenScore):
            raise TypeError('FrozenScore is not supposed to be added to model')

        elif isinstance(score, BaseScore):
            if score._name is None:
                raise ValueError(
                    f'When using `model.scores.add(score)` method,'
                    f' one should specify score name parameter during score initialization.'
                    f' For example `model.scores.add(IntratextCoherenceScore(name="name", ...))'
                )

            self._topicnet_scores[score._name] = score

        elif isinstance(score, artm.scores.BaseScore):
            self._artm_scores.add(score)

        else:
            raise TypeError(
                f'Unexpected score type "{type(score)}"!'
                f' Score should be either'
                f' topicnet.cooking_machine.models.BaseScore'
                f' or artm.scores.BaseScore!'
            )

    def __copy__(self):
        return copy.copy(self._scores)

    def __deepcopy__(self, memo: Dict):
        return copy.deepcopy(self._scores, memo)
