import numpy as np
from scipy import optimize
from scipy.spatial import distance

from .top_tokens_viewer import TopTokensViewer
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

        Parameters
        ----------
        class_ids : list of str, default - None
            parameter for model.get_phi method

        Returns
        -------
        tuple of nd.arrays of strings:
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
            first_model_order = list(
                model_phi
                .iloc[first_map_order]
                .index.values
            )
            second_model_order = list(
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
                first_model_order = list(
                    model_phi
                    .loc[first_map_order]
                    .index.values
                )
                second_model_order = list(
                    second_model_phi
                    .loc[second_map_order]
                    .index.values
                )
                return first_model_order, second_model_order

            second_model_order = list(
                second_model_phi
                .loc[first_map_order]
                .index.values
            )
            first_model_order = list(
                model_phi
                .loc[second_map_order]
                .index.values
            )
            return first_model_order, second_model_order
        else:
            raise TypeError('unknown self.mode value')

    def view_from_jupyter(
            self,
            display_output: bool = True,
            give_html: bool = False,
            **kwargs
    ):
        """
        TopicMapViewer method recommended for use
        from jupyter notebooks
        returns closest pairs of models topics
        and visualizes their top tokens

        The class provide information about top tokens
        of the model topics providing with different methods to score that.

        Parameters
        ----------
        display_output
            if provide output at the end of method run
        give_html
            return html string generated by the method

        Returns
        -------
        out_html
            html string of the output

        Another Parameters
        ------------------
        **kwargs
            *kwargs* are optional `~.TopTokenViewer` properties
        """
        from IPython.display import display_html
        from topicnet.cooking_machine.pretty_output import make_notebook_pretty
        if 'digits' in kwargs:
            digits = kwargs.pop('digits')
        else:
            digits = 5

        make_notebook_pretty()
        first_model_order, second_model_order = self.view()
        token_view = (TopTokensViewer(model=self.model, **kwargs)
                      .view_from_jupyter(
                          topic_names=first_model_order,
                          digits=digits,
                          display_output=False,
                          give_html=True))
        second_token_view = (TopTokensViewer(model=self.second_model, **kwargs)
                             .view_from_jupyter(
                                 topic_names=second_model_order,
                                 digits=digits,
                                 display_output=False,
                                 give_html=True))
        model_name = self.model.model_id
        second_model_name = self.second_model.model_id
        out_html = '<table style=display:inline; cellpadding="5";><tbody>{0}</tbody></table>'
        first_element = (f'<tr><td> First model name: '
                         f'{model_name}</td><td> Second model '
                         f'name: {second_model_name}</td></tr>{{0}}'
                         )
        out_html = out_html.format(first_element)
        table_contents = []
        for t1, t2 in zip(token_view, second_token_view):
            table_contents += [f'<tr><td>{t1}</td><td>{t2}</td></tr>']
        out_html = out_html.format(''.join(table_contents))
        if display_output:
            display_html(out_html, raw=True)
        if give_html:
            return out_html
