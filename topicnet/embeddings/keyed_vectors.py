from ..cooking_machine.models import TopicModel
from topicnet.cooking_machine.models.thetaless_regularizer import artm_dict2df
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
import pandas as pd
import numpy as np


def topic_model_to_keyed_vectors(model: TopicModel, modality: str = None):
    """
    Converts TopicModel to Gensim.KeyedVectors object

    Parameters
    ----------
    model: TopicModel
    modality: str
        the name of modality (others would be dropped)

    Returns
    -------
    Gensim::KeyedVectors object
    """
    if modality is not None:
        return phi_to_keyed_vectors(model.get_phi(class_ids=[modality]))
    return phi_to_keyed_vectors(model.get_phi())


def phi_to_keyed_vectors(phi):
    """
    Converts Phi matrix to Gensim.KeyedVectors object

    Parameters
    ----------
    phi: pandas.DataFrame

    Returns
    -------
    Gensim::KeyedVectors object
    """
    topics = phi.columns
    is_phi_multiindex = isinstance(phi.index, pd.core.indexes.multi.MultiIndex)
    words = phi.index.levels[1] if is_phi_multiindex else phi.index

    model = KeyedVectors(len(topics))
    model.add(words, phi)
    return model


def calc_dataset_statistics(dataset):
    """
    Converts Dataset to a pd.DataFrame object that additionaly
    contains normalized "idf" and normalized "tf" columns

    Parameters
    ----------
    dataset: Dataset

    Returns
    -------
    pandas.DataFrame
    """
    dict_parap = artm_dict2df(dataset.get_dictionary()).set_index(['class_id', 'token'])
    D = dataset._data.shape[0]

    dict_parap['idf'] = -np.log(dict_parap.token_df / D)
    dict_parap['tf'] = dict_parap.token_tf / sum(dict_parap.token_tf)
    return dict_parap


def get_weight_for(token, dict_df, avg_scheme, modality="@lemmatized"):
    """
    Calculates weight used for averaging
    See: Schmidt C. W. Improving a tf-idf weighted document vector embedding
    https://arxiv.org/abs/1902.09875

    Not all schemes are supported currently

    Parameters
    ----------
    token: str
    dict_df: pandas.DataFrame
    avg_scheme: str

    Returns
    -------
    float
    """
    key = (modality, token)
    data = dict_df.loc[key, :]
    if avg_scheme == "tf-idf":
        w = data.idf
    elif avg_scheme == "unit":
        w = 1
    else:
        raise NotImplementedError
    return w


def get_doc_vec_keyedvectors(kv_obj, doc_counter, dict_df, avg_scheme):
    """
    Calculates a document embedding via weighted averaging

    Parameters
    ----------
    kv_obj: KeyedVectors-like object
    doc_counter: Counter() that contains the same information as an entry in VowpalWabbit file
    dict_df: pandas.DataFrame
    avg_scheme: str

    Returns
    -------
    numpy ndarray
    """
    if isinstance(kv_obj, Word2VecKeyedVectors):
        dim = kv_obj.vector_size
    else:
        # for Navec
        dim = kv_obj.pq.dim
    # we can initialize this as numeric 0, but then
    # the function might return number instead of vector
    vec = pd.Series(data=np.zeros(dim))
    n = 0
    for token, tf in doc_counter.items():
        w = get_weight_for(token, dict_df, avg_scheme)
        if token in kv_obj:
            n += 1
            vec += tf * w * kv_obj[token]
    return vec / (n or 1)


def get_doc_vec_phi(phi, doc_counter, dict_df, avg_scheme, modality="@lemmatized"):
    """
    Calculates a document embedding via weighted averaging of entries from Phi matrix

    Parameters
    ----------
    phi: pandas.DataFrame
    doc_counter: Counter() that contains the same information as an entry in VowpalWabbit file
    dict_df: pandas.DataFrame
    avg_scheme: str

    Returns
    -------
    numpy ndarray
    """
    dim = phi.shape[1]
    optional_index = phi.columns
    # we can initialize this as numeric 0, but then
    # the function might return number instead of vector
    vec = pd.Series(data=np.zeros(dim), index=optional_index)
    n = 0
    for token, tf in doc_counter.items():
        w = get_weight_for(token, dict_df, avg_scheme, modality)
        key = (modality, token)
        if key in phi.index:
            # print(token, key, w, tf, phi.loc[key].sum())
            n += 1
            vec += tf * w * phi.loc[key]
    return vec / (n or 1)
