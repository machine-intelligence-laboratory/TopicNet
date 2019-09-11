import numpy as np
from scipy import optimize
from scipy.spatial import distance

from .base_viewer import BaseViewer


def compute_topic_mapping(matrix_left, matrix_right, metric='euclidean'):
    """
    This function provides mapping of topics
    from one model to the topics of the other model
    based on their simmularity defined by the metrics.

    Parameters
    ----------
    matrix_left : np.array
        a matrix of N1 topics x M tokens from the first model
        each row is a cluster in M-dimensional feature space
    matrix_right : np.array
        a matrix of N2 topics x M tokens from the second model
        each row is a cluster in M-dimensional feature space
    metric : str or class
        a string defining metric to use, or function that computes
        pairwise distance between 2 matrices (Default value = 'euclidean')

    Returns
    -------
    tuple of ndarrays
        returns two ndarrays of indices, where each index
        corresponds to a topic from respective models

    """
    if isinstance(metric, str):
        costs = distance.cdist(matrix_left, matrix_right, metric=metric)
    else:
        costs = metric(matrix_left, matrix_right)

    results = optimize.linear_sum_assignment(costs)
    return results


class TopicMapViewer(BaseViewer):
    def __init__(
        self,
        model,
        second_model,
        mode='min',
        metric='euclidean',
        class_ids=None,
    ):
        """
        Performs a mapping between topics of two model
        matching two closest topics together based on
        the Hungarian algorithm.

        Parameters
        ----------
        model : TopicModel
            first model to compare
        second_model : TopicModel
            second model to compare
        mode : string
            "min" or "max"  
            "min" performs one to one mapping of 
            min(n_topics_first_model, n_topics_second_model) length  
            "max" performs mapping for
            max(n_topics_first_model, n_topics_second_model), in that case
            topics from model with minimal number will have a few topics mapped on it
        metric : str or function
            name of scipy metrics used in distance computation
            or function that computes pairwise distance between 2 matrices
            (Default value = "euclidean")

        """  # noqa: W291
        super().__init__(model=second_model)
        self.second_model = self.model
        super().__init__(model=model)
        # TODO the default library method for get_phi
        # returns  N x T matrix while we implemented T x N
        self.metric = metric
        self.mode = mode
        self.class_ids = class_ids

    def view(self, class_ids=None):
        """
        Returns pairs of close topics.
        :param class_ids: list of str, default - None
            parameter for model.get_phi method
        :return: tuple of nd.arrays of strings:
            two ordered arrays of topic name pairs

        """
        if class_ids is None:
            class_ids = self.class_ids
        model_phi = self.model.get_phi(class_ids=class_ids).T
        second_model_phi = self.second_model.get_phi(class_ids=class_ids).T
        num_topics_first = model_phi.values.shape[0]
        num_topics_second = second_model_phi.values.shape[0]
        if self.mode == 'min':
            first_map_order, second_map_order = compute_topic_mapping(model_phi.values,
                                                                      second_model_phi.values,
                                                                      metric=self.metric)
            first_model_order = (
                model_phi
                .iloc[first_map_order]
                .index.values
            )
            second_model_order = (
                second_model_phi
                .iloc[second_map_order]
                .index.values
            )
            return first_model_order, second_model_order

        elif self.mode == 'max':
            more_topics_second = num_topics_first <= num_topics_second

            if more_topics_second:
                iterate_phi_first = model_phi.values
                iterate_phi_second = second_model_phi.values
                phi_first_indexes = model_phi.index
                phi_second_indexes = second_model_phi.index
            else:
                iterate_phi_first = second_model_phi.values
                iterate_phi_second = model_phi.values
                phi_first_indexes = second_model_phi.index
                phi_second_indexes = model_phi.index

            first_map_order = []
            second_map_order = []
            while iterate_phi_second.shape[0] > 0:
                answer_batch = compute_topic_mapping(iterate_phi_first,
                                                     iterate_phi_second,
                                                     metric=self.metric)
                first_map_order += list(phi_first_indexes[answer_batch[0]])
                second_map_order += list(phi_second_indexes[answer_batch[1]])
                iterate_phi_second = np.delete(iterate_phi_second, answer_batch[1], axis=0)
                phi_second_indexes = np.delete(phi_second_indexes, answer_batch[1], axis=0)

            if more_topics_second:
                first_model_order = (
                    model_phi
                    .loc[first_map_order]
                    .index.values
                )
                second_model_order = (
                    second_model_phi
                    .loc[second_map_order]
                    .index.values
                )
                return first_model_order, second_model_order

            second_model_order = (
                second_model_phi
                .loc[first_map_order]
                .index.values
            )
            first_model_order = (
                model_phi
                .loc[second_map_order]
                .index.values
            )
            return first_model_order, second_model_order
        else:
            raise TypeError('unknown self.mode value')
