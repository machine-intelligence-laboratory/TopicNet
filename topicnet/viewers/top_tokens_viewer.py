import pandas as pd
import numpy as np
import warnings

from .base_viewer import BaseViewer
from sklearn.feature_extraction.text import TfidfVectorizer


def get_top_values(values, top_number, return_indexes=True):
    """
    Returns top_number top values from the matrix for each column.

    Parameters
    ----------
    values : np.array
        a two dimensional array of values
    top_number : int
        number of top values to return
    return_indexes : bool
        a flag to return indexes together with the top values

    Returns
    -------
    top_values : nd.array
        array of top_number top values for each column of the initial array
    (optional) top_indexes : nd.array
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

    if return_indexes:
        return top_values, top_indexes

    return top_values


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
    float
        probability that a random token from the collection belongs to that topic

    """

    n_wt = model.get_phi(class_ids=class_ids, model_name=model.model_nwt)
    n_t = n_wt.sum(axis=0)  # sum over all words
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
    phi : pd.Dataframe
        phi matrix of the model

    Returns
    -------
    score : pd.Dataframe
        wheighted phi matrix

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


def compute_clusters_top_tokens_by_clusters_tfidf(objects_cluster, objects_content,
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


# get top tokens from topic (sorted by scores)
class TopTokensViewer(BaseViewer):
    """ """
    def __init__(self,
                 model,
                 class_ids=None,
                 method='blei',
                 num_top_tokens=5,
                 alpha=1,
                 dataset=None,):
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
        dataset: Dataset
            a class that stores infromation about the collection

        """  # noqa: W291
        known = ['top', 'phi', 'blei', 'tfidf', 'likelihood', 'ptw']

        super().__init__(model=model)
        self.num_top_tokens = num_top_tokens
        self.class_ids = class_ids
        if method in known:
            self.method = method
        else:
            raise ValueError(f'method {method} is not known')
        self.alpha = alpha
        self._dataset = dataset

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

    def view(self, class_ids=None, raw_data=None):
        """
        Returns list of tuples (token,score) for each topic in the model.

        Parameters
        ----------
        raw_data : list of list of str
            necessary for 'tfidf' option

        Returns
        -------
        all_topics_top_tokens : nested 3 level dict
            dict of topics each key contains dict of modalities with each modality key
            having dict of top tokens format: (token, value)

        """
        if class_ids is None:
            class_ids = self.class_ids
        phi = self.model.get_phi(class_ids=class_ids)

        if self.method == 'tfidf':
            objects_cluster = (
                self._model
                .get_theta(dataset=self._dataset)
                .values.argmax(axis=0)
            )
            top_tokens_sorted = compute_clusters_top_tokens_by_clusters_tfidf(objects_cluster,
                                                                              raw_data,)
            return top_tokens_sorted

        target_values = self._get_target_values(phi)

        phi = target_values.T
        phi.index = pd.MultiIndex.from_tuples(phi.index)
        topic_names = phi.columns.values
        if self.class_ids is None:
            modalities = phi.index.levels[0].values
        else:
            modalities = self.class_ids

        all_topics_top_tokens = {}
        for topic_name in topic_names:
            phi_column = phi[topic_name]
            topic_top_tokens = {}

            for modality in modalities:
                top_tokens_values, top_tokens_indexes = get_top_values(
                    phi_column.loc[modality].values,
                    top_number=self.num_top_tokens,
                )
                top_tokens = phi_column.loc[modality].index[top_tokens_indexes]

                topic_top_tokens[modality] = dict(zip(top_tokens, top_tokens_values))

            all_topics_top_tokens[topic_name] = topic_top_tokens

        return all_topics_top_tokens

    def to_html(self, all_topics_top_tokens, topic_names=None, thresh=1e-5):
        """
        method that generates html version of
        dataframes to be displayed by jupyter notebooks
        display by:
        from IPython.display import display_html
        display_html(html_line, raw=True)
        :param all_topics_top_tokens: dict of dicts
            dict where first level keys are topic names
            second level keys are modalities
            third level keys are tokens with their scores
            as float values
        :param topic_names: list of strings -- optional
            ordered list of initial dictionary keys
        :return topic_dataframes: list of strings
            list of dataframes in html encoding
        """
        topic_dataframes = []
        if topic_names is None:
            topic_names = all_topics_top_tokens.keys()
        digits = int(- np.log10(thresh))
        for topic in topic_names:
            data = all_topics_top_tokens[topic]
            dfs = [
                pd.DataFrame.from_dict(
                    mod,
                    orient='index',
                    columns=[topic + ' ' + key]
                ).round(digits)
                for key, mod in data.items()
            ]
            df_style = []
            for df in dfs:
                df = df.loc[(df.values > thresh).flatten()]
                if len(df) > 0:
                    df.index = df.index.str.replace('<', '&lt;').str.replace('>', '&gt;')
                df_style += [df.style.set_table_attributes("style='display:inline'")._repr_html_()]
            topic_dataframes += [''.join(df_style)]
        return topic_dataframes
