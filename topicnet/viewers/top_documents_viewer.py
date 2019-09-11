import numpy as np

from collections import defaultdict
from .base_viewer import BaseViewer


def transform_cluster_objects_list_to_dict(object_clusters):
    """
    Transforms list of object clusters to dict.

    Parameters
    ----------
    object_clusters : list
        ith element of list is cluster of ith object

    Returns
    -------
    clusters : dict
        dict, where key is clusterlabel (int), value is cluster objects (list)

    """
    clusters = defaultdict(list)

    for object_label, cluster_label in enumerate(object_clusters):
        clusters[cluster_label].append(object_label)

    clusters = dict(clusters)

    return clusters


def predict_cluster_by_precomputed_distances(precomputed_distances):
    """
    Predict a cluster for each object with precomputed distances.

    Parameters
    ----------
    precomputed_distances : np.array
        array of shape (n_topics, n_objects) - distances from clusters to objects

    Returns
    -------
    np.array
        array of length X.shape[0], each element is cluster of ith object

    """
    return precomputed_distances.T.argmin(axis=1).ravel()


def compute_cluster_top_objects_by_distance(precomputed_distances,
                                            max_top_number=10,
                                            object_clusters=None):
    """
    Compute the most representative objects for each cluster
    using the precomputed_distances.

    Parameters
    ----------
    precomputed_distances : np.array
        array of shape (n_topics, n_objects) -
        a matrix of pairwise distances: distance from ith cluster centroid to the jth object
    max_top_number : int
        maximum number of top objects of cluster (resulting number can be less than it) 
        (Default value = 10)
    object_clusters : np,array
        array of shape n_objects - precomputed clusters for objects

    Returns
    -------
    clusters_top_objects : list of list of indexes 
        (Default value = None)
    """  # noqa: W291
    # prediction for objects
    if object_clusters is None:
        object_clusters = predict_cluster_by_precomputed_distances(precomputed_distances)
    # transformation from list to dict
    clusters = transform_cluster_objects_list_to_dict(object_clusters)
    n_topics = precomputed_distances.shape[0]

    clusters_top_objects = []
    for cluster_label in range(n_topics):
        # cluster is empty
        if cluster_label not in clusters.keys():
            clusters_top_objects.append([])
            continue
        cluster_objects = np.array(clusters[cluster_label])
        cluster_objects_to_center_distances = (
            precomputed_distances[cluster_label][cluster_objects]
        )
        if max_top_number >= cluster_objects.shape[0]:
            # cluster is too small; grab all objects
            indexes_of_top_objects = np.arange(0, cluster_objects.shape[0])
        else:
            # filter by distance with partition
            indexes_of_top_objects = np.argpartition(
                cluster_objects_to_center_distances,
                kth=max_top_number
            )[:max_top_number]

        distances_of_top_objects = cluster_objects_to_center_distances[indexes_of_top_objects]
        top_objects = cluster_objects[indexes_of_top_objects]

        # sorted partitioned array
        indexes_of_top_objects_sorted_by_distance = np.argsort(distances_of_top_objects)
        sorted_top_objects = top_objects[indexes_of_top_objects_sorted_by_distance]

        clusters_top_objects.append(sorted_top_objects.tolist())

    return clusters_top_objects


class TopDocumentsViewer(BaseViewer):
    """ """
    def __init__(self,
                 model,
                 dataset=None,
                 precomputed_distances=None,
                 object_clusters=None,
                 max_top_number=10):
        """
        The class provide information about
        top documents for the model topics
        from some collection.

        Parameters
        ----------
        model : TopicModel
            a class of topic model
        dataset : Dataset
            a class that stores information about the collection
        precomputed_distances :  np.array
            array of shape (n_topics, n_objects) -
            an optional matrix of pairwise distances:
            distance from ith cluster centroid to the jth object
        object_clusters : list of int
            an optional array of topic number labels
            for each document from the collection
            ith element of list is cluster of ith object
        max_top_number : int
            number of top documents to provide for each cluster

        """
        super().__init__(model=model)
        self.precomputed_distances = precomputed_distances
        self.object_clusters = object_clusters
        self._dataset = dataset
        self.max_top_number = max_top_number

    def view(self, current_num_top_doc=None):
        """
        Returns list of tuples (token,score) for
        each topic in the model.

        Parameters
        ----------
        current_num_top_doc : int
            number of top documents to provide for
            each cluster (Default value = None)

        Returns
        -------
        all_cluster_top_titles: list of list
            returns list for each topic of the model list
            contains document_ids of top documents for that topic

        """
        if current_num_top_doc is None:
            current_num_top_doc = self.max_top_number

        theta = self.model.get_theta(dataset=self._dataset)
        document_ids = theta.columns.values
        if self.precomputed_distances is None:
            precomputed_distances = 1.0 - theta.values
        else:
            precomputed_distances = self.precomputed_distances
        if self.object_clusters is not None:
            num_clusters, num_documents = precomputed_distances.shape
            if len(self.object_clusters) != num_documents:
                raise ValueError('number of topics differ from number of labels')
            if not set(range(num_clusters)) >= set(self.object_clusters):
                raise ValueError('provided clusters are not in 0 to num_clusters - 1 range')
        all_cluster_top_indexes = compute_cluster_top_objects_by_distance(
            precomputed_distances,
            max_top_number=current_num_top_doc,
            object_clusters=self.object_clusters
        )

        all_cluster_top_titles = list()
        for cluster_top in all_cluster_top_indexes:
            all_cluster_top_titles += [list(document_ids[cluster_top])]
        return all_cluster_top_titles
