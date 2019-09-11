import numpy as np
import pandas as pd
import numpy.testing as npt
from ..viewers import topic_mapping
from ..cooking_machine.models.base_model import BaseModel


class dummy_model(BaseModel):
    def __init__(self, matrix):
        self.values = matrix

    def get_phi(self, class_ids):
        """ """
        index = ['topic_'+str(num) for num in range(len(self.values))]
        phi = pd.DataFrame(self.values, index=index)
        return phi


def test_diagonal_answer_same():
    """ """
    test_m = np.array([[1, 0],
                       [2, 0],
                       [3, 0]])

    answer = ([0, 1, 2], [0, 1, 2])

    result = topic_mapping.compute_topic_mapping(test_m, test_m)
    npt.assert_array_equal(result, answer)


def test_diagonal_answer_different():
    """ """
    test_left = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    test_right = np.array([
                    [0.5, 0.5, 0],
                    [0, 0.5, 0.5],
                    [0, 0, 1]])
    answer = ([0, 1, 2], [0, 1, 2])

    result = topic_mapping.compute_topic_mapping(test_left, test_right)
    npt.assert_array_equal(result, answer)


def test_map_viewer_min():
    """ """
    model_one = dummy_model([
        [1, 2, 3, 4, 5, 6],
        [0, 0, 0, 0, 0, 0],
       ])
    model_two = dummy_model([
        [0, 0],
        [1, 2],
       ])
    maping = topic_mapping.TopicMapViewer(model_one, model_two).view()
    right_maping = ([0, 1], [0, 1])
    npt.assert_array_equal(maping, right_maping)


def test_map_viewer_max():
    """ """
    model_one = dummy_model([
        [1, 2, 3, 4, 5, 6],
        [0, 0, 0, 0, 0, 0],
    ])
    model_two = dummy_model([
        [0, 0],
        [1, 2],
       ])
    maping = topic_mapping.TopicMapViewer(model_one, model_two, mode='max').view()
    right_maping = ([0, 1, 2, 3, 4, 5], [0, 1, 0, 1, 0, 1])
    npt.assert_array_equal(maping, right_maping)
