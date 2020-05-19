import bisect
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Iterator, List, Tuple, Union
import warnings

from .base_viewer import BaseViewer


def get_top_values(values, top_number):
    """
    Returns top_number top values from the matrix for each column.

    Parameters
    ----------
    values : np.array
        a two dimensional array of values
    top_number : int
        number of top values to return

    Returns
    -------
    top_values : nd.array
        array of top_number top values for each column of the initial array
    top_indexes : nd.array
        array of original indexes for top_values array (Default value = True)

    """
    if top_number > len(values):
        top_number = len(values)
        warnings.warn('num_top_tokens greater than modality size', UserWarning)

    top_indexes = np.argpartition(
        values, len(values) - top_number
    )[-top_number:]

    top_values = values[top_indexes]
    sorted_top_values_indexes = top_values.argsort()[::-1]

    top_values = top_values[sorted_top_values_indexes]

    # get initial indexes
    top_indexes = top_indexes[sorted_top_values_indexes]

    return top_values, top_indexes


def get_top_values_by_sum(values, min_sum_value,):
    """
    Returns top values until sum of their scores breaches `min_sum_value`.

    Parameters
    ----------
    values : np.array
        a one dimensional array of values
    min_sum_value : float
        min sum value of top values to return

    Returns
    -------
    top_values : nd.array
        array of top values with sum at least min_sum_value
    top_indexes : nd.array
        array of original indexes for top_values array (Default value = True)

    Examples
    --------
    >>> values = np.array([1, 3, 2, 0.1, 5, 0])
    >>> min_sum = 8.1
    >>> top_values, top_indexes = get_top_values_by_sum(values, min_sum)
    Result: top_values, top_indexes = (array([5., 3., 2.]), array([4, 1, 2]))
    """
    all_sum = np.sum(values)
    if all_sum < min_sum_value:
        warnings.warn(f'min_sum_value = {min_sum_value}'
                      f' is greater than sum of all elements = {all_sum}',
                      UserWarning)
        min_sum_value = all_sum

    top_indexes = np.argsort(values)[::-1]
    top_values = values[top_indexes]
    cum_sum = np.cumsum(top_values)
    ind_min_sum = bisect.bisect_left(cum_sum, min_sum_value)
    top_indexes = top_indexes[:ind_min_sum + 1]
    top_values = values[top_indexes]

    return top_values, top_indexes


def compute_pt_distribution(model, class_ids=None):
    """
    Calculates the Prob(t) vector (vector contains an entry for each topic).

    Parameters
    ----------
    model : TopicModel
        model under the scope
    class_ids : list of str or None
        list of modalities to consider, which takes all modalities in the model
        (Default value = None)

    Returns
    -------
    float probability that a random token from the collection belongs to that topic
    """

    n_wt = model.get_phi(class_ids=class_ids, model_name=model.model_nwt)
    n_t = n_wt.sum(axis=0)  # sum over all words
    # TODO: maybe this is not P(t)
    #  P(t) means prior P()? here using info from model, so not P(t), more like P(t | model)
    return n_t / n_t.sum()


def compute_joint_pwt_distribution(phi, p_t):
    """
    p(t) is prob(topic = t), defined as p(t) = sum_t n_t / n  

    if we fix some word w, we can calculate weighted_pk:  
    wp_t = p(t) p(w|t)

    Parameters
    ----------
    phi : pd.Dataframe
        phi matrix of the model
    p_t : np.array of float
        probability that a random token from the collection belongs to that topic

    Returns
    -------
    joint_pwt : np.array of float
        array of probabilities that a fixed token from the collection
        belongs to that topic

    """  # noqa: W291

    joint_pwt = p_t[:, np.newaxis] * phi.transpose()
    return joint_pwt


def compute_ptw(joint_pwt):
    return joint_pwt / np.sum(joint_pwt, axis=0)  # sum by all T


def compute_likelihood_vectorised(phi, p_t, joint_pwt):
    """
    Likelihood ratio is defined as  
        L = phi_wt / sum_k p(k)/p(!t) phi_wk  
    equivalently:  
        L = phi_wt * p(!t) / sum_k!=t p(k) phi_wk  
    after some numpy magic, you can get:  
        L = phi[topic, id] * (1 - p_t[topic]) / {(sum(joined_pwt) - joined_pwt[topic])}  
    numerator and denominator are calculated separately.  

    Parameters
    ----------
    phi : pd.Dataframe
        phi matrix of the model
    p_t : np.array of float
        probability that a random token from the collection belongs to that topic
    joint_pwt : np.array of float
        array of probabilities that a fixed token from the collection
        belongs to that topic

    Returns
    -------
    target_values : np.array of float
        vector of likelihood ratios that tokens belong to the given topic

    """  # noqa: W291
    # if phi and joint_pwt are DataFrame, then
    # denominator will have the same Index/Columns as them
    # TODO: check equality
    denominator = (np.sum(joint_pwt, axis=0) - joint_pwt)
    multiplier = (1 - p_t)[:, np.newaxis]
    if hasattr(phi, "values"):
        numerator = phi.values.transpose() * multiplier
    else:
        numerator = phi.transpose() * multiplier

    bad_indices = (denominator == 0)
    denominator[bad_indices] = 1
    target_values = numerator / denominator

    # infinite likelihood ratios aren't interesting
    target_values[bad_indices] = float("-inf")
    return target_values


def compute_blei_scores(phi):
    """
    Computes Blei score  
    phi[wt] * [log(phi[wt]) - 1/T sum_k log(phi[wk])]

    Parameters
    ----------
    phi : pd.DataFrame
        phi matrix of the model

    Returns
    -------
    score : pd.DataFrame
        weighted phi matrix

    """  # noqa: W291

    topic_number = phi.shape[0]
    blei_eps = 1e-42
    log_phi = np.log(phi + blei_eps)
    denominator = np.sum(log_phi, axis=0)
    denominator = denominator[np.newaxis, :]

    if hasattr(log_phi, "values"):
        multiplier = log_phi.values - denominator / topic_number
    else:
        multiplier = log_phi - denominator / topic_number

    score = (phi * multiplier).transpose()
    return score


def compute_clusters_top_tokens_by_clusters_tfidf(
        objects_cluster, objects_content,
        max_top_number=10, n_topics=None):
    """
    Function for document-like clusters.  
    For each cluster compute top tokens of cluster. Top tokens are defined by tf-idf scheme.
    Tf-idf is computed as if clusters is concatenation of all it documents.

    Parameters
    ----------
    objects_cluster : list of int
        ith element of list is cluster of ith object
    objects_content : list of list of str
        each element is sequence of tokens
    max_top_number : int
        maximum number of top tokens of cluster (resulting number can be less than it) 
        (Default value = 10)
    n_topics : int
        number of topics in model (Default value = None) 
        if None than it will be calculated automatically from object_clusters

    Returns
    -------
    clusters_top_tokens : list of list of str:
        ith element of list is list of top tokens of ith cluster

    """  # noqa: W291
    # TODO: check type of cluster_content, raise Error if it has spaces in it

    n_topics = (
        n_topics if n_topics is not None
        else max(objects_cluster) + 1
    )

    cluster_tokens = {
        num_cluster: []
        for num_cluster in range(n_topics)
    }

    for object_cluster, object_content in zip(objects_cluster, objects_content):
        cluster_tokens[object_cluster] += object_content

    cluster_tokens = [
        cluster_content
        for cluster_label, cluster_content in sorted(cluster_tokens.items(), key=lambda x: x[0])
    ]

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf_array = vectorizer.fit_transform(cluster_tokens).toarray()
    index_to_word = [
        word
        for word, index in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1])
    ]

    cluster_top_tokens_indexes = (
        tfidf_array
        .argsort(axis=1)[:, tfidf_array.shape[1] - max_top_number:]
    )

    cluster_top_tokens = []
    for cluster_label, cluster_top_tokens_indexes in enumerate(cluster_top_tokens_indexes):
        cluster_top_tokens += [
            (index_to_word[index], tfidf_array[cluster_label, index])
            for index in cluster_top_tokens_indexes[::-1]
            if tfidf_array[cluster_label, index] != 0
        ]

    return cluster_top_tokens


# TODO: check why this better than plain df.to_html()
def convert_df_to_html(df):
    return df.style\
               .set_table_attributes("style='display:inline'")\
               ._repr_html_()


class TopTokensViewer(BaseViewer):
    """Gets top tokens from topic (sorted by scores)"""
    def __init__(self,
                 model,
                 class_ids: List[str] = None,
                 method: str = 'blei',
                 num_top_tokens: int = 10,
                 alpha: float = 1,
                 by_sum: bool = False,
                 sum_value: float = None,
                 dataset=None):
        """
        The class provide information about top tokens 
        of the model topics providing with different methods to score that.

        Parameters
        ----------
        model : TopicModel
            a class of topic model
        class_ids : list of int
            class ids for documents in topic needed only for tfidf method
        method : str
            method to score the topics could be any of
            top, phi - top tokens by probability in topic  
            blei - some magical Blei article score  
            tfidf - Term Frequency inversed Document Frequency  
            likelihood - Likelihood ratio score  
            ptw - something like likelihood  
        num_top_tokens : int
            number of top tokens to provide for each topic
        alpha : float between 0 and 1
            additional constant needed for
            ptw method of scoring
        by_sum
            a flag providing adjustable ammount of top tokens
            based on sum of their scores
        sum_value
            a constant deciding "how many" tokens to return in each topic
            a good default value might be different depending on self.method value
        dataset: Dataset
            a class that stores infromation about the collection

        """  # noqa: W291
        known = ['top', 'phi', 'blei', 'tfidf', 'likelihood', 'ptw']

        super().__init__(model=model)

        self.num_top_tokens = num_top_tokens
        self.class_ids = class_ids
        self.sum_value = sum_value
        self.by_sum = by_sum

        if self.sum_value is not None:
            self.by_sum = True

        if method in known:
            self.method = method
        else:
            raise ValueError(f'method {method} is not known')

        self.alpha = alpha
        self._dataset = dataset
        self._cached_top_tokens = None

    @property
    def cached_top_tokens(self):
        if self._cached_top_tokens is None:
            self._cached_top_tokens = self.view(three_levels=False)
        return self._cached_top_tokens

    def _get_target_values(self, phi):
        """
        Precomputes various model scores
        """
        if self.method == 'blei':
            return compute_blei_scores(phi)

        elif self.method in ['top', 'phi']:
            return phi.transpose()

        elif self.method in ['ptw', 'likelihood']:
            p_t = compute_pt_distribution(self._model)
            joint_pwt = compute_joint_pwt_distribution(phi, p_t)

            if self.method == 'likelihood':
                return compute_likelihood_vectorised(phi, p_t, joint_pwt)

            elif self.method == 'ptw':
                ptw_vector = compute_ptw(joint_pwt)
                ptw_component = self.alpha * ptw_vector
                phi_component = (1 - self.alpha) * phi.transpose()

                return ptw_component + phi_component

    def _determine_sum(self, num_words_in_vocab):
        """ """
        if self.method == 'blei':
            self.sum_value = 2.0

        elif self.method in ['top', 'phi']:
            self.sum_value = 1 / num_words_in_vocab * self.num_top_tokens

        elif self.method == 'ptw':
            self.sum_value = self.num_top_tokens

        elif self.method == 'likelihood':
            raise ValueError('There is no good way to determine'
                             ' automatical sum_value for method "likelihood".'
                             ' Please, define it manually')

    def view(
            self,
            class_ids: List[str] = None,
            raw_data: List[List[str]] = None,
            three_levels: bool = True
    ) -> Union[Dict[str, Dict[str, Dict[str, float]]],
               Dict[str, Dict[Tuple[str, str], float]]]:
        """
        Returns list of tuples (token, score) for each topic in the model.

        Parameters
        ----------
        class_ids
            Modalities from which to retrieve top tokens
        raw_data : list of list of str
            Necessary for 'tfidf' option
        three_levels
            If true, three level dict will be returned, otherwise — two level one
        returns
        -------
        topic_top_tokens : nested 3 or 2-level dict
            Topic -> Modality -> Token -> Probability or
            Topic -> (Modality, Token) -> Probability

        """
        if class_ids is None:
            class_ids = self.class_ids

        phi = self.model.get_phi(class_ids=class_ids)
        if self.by_sum and self.sum_value is None:
            self._determine_sum(num_words_in_vocab=phi.shape[0])

        if self.method == 'tfidf':
            objects_cluster = (
                self._model
                .get_theta(dataset=self._dataset)
                .values
                .argmax(axis=0)
            )
            top_tokens_sorted = compute_clusters_top_tokens_by_clusters_tfidf(
                objects_cluster, raw_data
            )

            return top_tokens_sorted

        target_values = self._get_target_values(phi)

        phi = target_values.T
        phi.index = pd.MultiIndex.from_tuples(phi.index)
        topic_names = phi.columns.values

        if self.class_ids is None:
            modalities = phi.index.levels[0].values
        else:
            modalities = self.class_ids

        topic_top_tokens = {}

        for topic_name in topic_names:
            topic_column = phi[topic_name]
            modality_top_tokens = {}

            for modality in modalities:
                if self.by_sum:
                    top_tokens_values, top_tokens_indexes = get_top_values_by_sum(
                        topic_column.loc[modality].values,
                        min_sum_value=self.sum_value,
                    )
                else:
                    top_tokens_values, top_tokens_indexes = get_top_values(
                        topic_column.loc[modality].values,
                        top_number=self.num_top_tokens,
                    )
                top_tokens = topic_column.loc[modality].index[top_tokens_indexes]

                if three_levels:
                    modality_top_tokens[modality] = dict(zip(top_tokens, top_tokens_values))
                else:
                    modality_top_tokens.update(
                        dict(zip([(modality, token) for token in top_tokens], top_tokens_values))
                    )

            topic_top_tokens[topic_name] = modality_top_tokens

        return topic_top_tokens

    def to_html(
            self,
            topic_names: Union[str, List[str]] = None,
            digits: int = 5,
            thresh: float = None,  # Deprecated
            horizontally_stack: bool = True) -> str:
        """
        Generates html version of dataframes to be displayed by Jupyter notebooks

        Parameters
        ----------
        topic_names : list of strings
            Initial dictionary keys
        digits : int
            Number of digits to round each probability to
        thresh : float [Deprecated]
            Threshold used for calculating `digits` and throwing out too low probabilities
        horizontally_stack : bool
            if True, then tokens for each modality will be stacked horizontally
            (instead of being a single long multi-line DataFrame)

        Examples
        --------
        >>> from IPython.display import HTML, display_html
        >>>
        >>> # model training here
        >>> # ...
        >>> viewer = TopTokensViewer(model)
        >>> display_html(viewer.to_html(), raw=True)
        >>> # or
        >>> HTML(viewer.to_html())
        """
        if topic_names is not None:
            if isinstance(topic_names, str):
                topic_names = [topic_names]
            num_topics_requested = len(topic_names)
            topic_names = [t for t in topic_names if t in self._model.topic_names]
            if len(topic_names) < num_topics_requested:
                warnings.warn(
                    'Some of the requested topics are absent from the model',
                )

        if thresh is not None:  # TODO: remove thresh some day
            warnings.warn(
                'Don\'t specify `thresh` in `to_html()` anymore, use `digits`',
                DeprecationWarning
            )

            digits = int(-np.log10(thresh))

        df = self.to_df(topic_names, digits)

        if len(df) > 0:
            for level, old_names in enumerate(df.index.levels):
                new_names = old_names.str.replace('<', '&lt;').str.replace('>', '&gt;')
                renamer = dict(zip(old_names, new_names))
                df.rename(index=renamer, inplace=True, level=level)

        if horizontally_stack:
            modalities = df.index.levels[0].unique()
            result = ''.join(
                convert_df_to_html(df.query("modality == @m"))
                for m in modalities
            )
            return result

        return convert_df_to_html(df)

    def to_df(self, topic_names: Iterator[str] = None, digits: int = 5) -> pd.DataFrame:
        topic_top_tokens = self.cached_top_tokens

        if topic_names is not None:
            topic_top_tokens = {
                topic: tokens for topic, tokens in topic_top_tokens.items()
                if topic in topic_names
            }
        if not isinstance(digits, int):
            warnings.warn(
                f'Need "int" digits. '
                f'Casting given value "{digits}" of type "{type(digits)}" to int'
            )

            digits = int(digits)

        return self._to_df(topic_top_tokens, digits)

    @staticmethod
    def _to_df(
            topic_top_tokens: Dict[str, Dict[Tuple[str, str], float]],
            digits: int) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(topic_top_tokens).round(digits)
        df.index = pd.MultiIndex.from_tuples(
            df.index,
            names=['modality', 'token']  # TODO: names should be the same as in TopicModel's Phi?
        )

        df.fillna(0.0, inplace=True)

        # Due to some problems with pandas following crunch is applied:
        if len(df.columns) == 1:
            col_to_sort_by = df.columns.values[0]
            return (df.set_index(col_to_sort_by, append=True)
                    .sort_index(level=[0, 2], ascending=[True, False])
                    .reset_index(col_to_sort_by))

        return df

    def view_from_jupyter(
            self,
            topic_names: Union[str, List[str]] = None,
            digits: int = 5,
            horizontally_stack: bool = True,
            one_topic_per_row: bool = True,
            display_output: bool = True,
            give_html: bool = False,
    ):
        """
        TopTokensViewer method recommended for use
        from jupyter notebooks

        Parameters
        ----------
        topic_names
            topics requested for viewing
        digits
            number of digits to round each probability to
        horizontally_stack
            if True, then tokens for each modality will be stacked horizontally
            (instead of being a single long multi-line DataFrame)
        one_topic_per_row
            if True, each topic will be on its own row;
            if False, topics will be arranged in one row
        display_output
            request for function to output the information
            together with iterable output intended to be used
            as user defined output
        give_html
            return html string generated by the method

        Returns
        -------
        topic_html_strings: list of strings in HTML format

        Examples
        --------
        >>> # model training here
        >>> # ...
        >>> viewer = TopTokensViewer(model)
        >>> information = viewer.view_from_jupyter()
        >>> # or
        >>> information = viewer.view_from_jupyter(output=False)
        """
        from IPython.core.display import display_html
        from topicnet.cooking_machine.pretty_output import make_notebook_pretty

        make_notebook_pretty()
        if isinstance(topic_names, list):
            pass
        elif isinstance(topic_names, str):
            topic_names = [topic_names]
        elif topic_names is None:
            topic_names = self._model.topic_names
        else:
            raise TypeError(f'Invalid type `topic_names` type: "{type(topic_names)}"')

        topic_html_strings = []

        for topic in topic_names:
            topic_html = self.to_html(
                topic_names=topic,
                digits=digits,
                horizontally_stack=horizontally_stack,
            )

            topic_html_strings.append(topic_html)

        if not display_output:
            pass
        elif one_topic_per_row:
            display_html('</br>'.join(topic_html_strings), raw=True)
        else:
            display_html('&nbsp;'.join(topic_html_strings), raw=True)

        if give_html:
            return topic_html_strings
