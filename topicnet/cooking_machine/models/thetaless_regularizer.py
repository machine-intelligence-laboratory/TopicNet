import numpy as np
import os
import pandas as pd
import scipy.sparse
import warnings

from numba import jit

import artm

from .base_regularizer import BaseRegularizer
from ..dataset import Dataset


# TODO: move this to BigARTM
# ==================================

FIELDS = 'token class_id token_value token_tf token_df'.split()


def artm_dict2df(artm_dict):
    """
    :Description: converts the BigARTM dictionary of the collection
        to the pandas.DataFrame.
        This is approximately equivalent to the dictionary.save_text()
        but has no I/O overhead

    """
    dictionary_data = artm_dict._master.get_dictionary(artm_dict._name)
    dict_pandas = {field: getattr(dictionary_data, field)
                   for field in FIELDS}
    return pd.DataFrame(dict_pandas)

# ==================================


EPS = 1e-20


# TODO: is there a better way to do this?
def obtain_token2id(dataset: Dataset):
    """
    Allows one to obtain the mapping from token to the artm.dictionary id of that token
    (useful for low-level operations such as reading batches manually)

    Returns
    -------
    dict:
        maps (token, class_id) to integer (corresponding to the row of Phi / dictionary id)

    """
    df = artm_dict2df(dataset.get_dictionary())
    df_inverted_index = df[['token', 'class_id']].reset_index().set_index(['token', 'class_id'])

    return df_inverted_index.to_dict()['index']


def dataset2sparse_matrix(dataset, modality, modalities_to_use=None):
    """
    Builds a sparse matrix from batch_vectorizer linked to the Dataset

    If you need an inverse mapping:

    >>> d = sparse_n_dw_matrix.todok()  # convert to dictionary of keys format
    >>> dict_of_csr = dict(d.items())

    Parameters
    ----------
    dataset: Dataset
    modality: str
        the remaining modalities will be ignored
        (their occurrences will be replaced with zeros, but they will continue to exist)
    modalities_to_use: iterable
        a set of modalities the underlying topic model is using (this is about topic model,
        not regularizer; this parameter ensures that the shapes of n_dw matrix and actual
        Phi matrix match).

        The tokens outside of this list will be discarded utterly
        (the resulting matrix will have no entries corresponding to them)

        For artm.ARTM() models, you need to pass whatever is inside class_ids;
        while TopicModel usually requires this to be set inside modalities_to_use.

        If you hadn't explicitly listed any modalities yet, you probably could
        leave this argument as None.

        If you use a single modality, wrap it into a list (e.g.['@word'])

    Returns
    -------
    n_dw_matrix: scipy.sparse.csr_matrix  
        The matrix of document-word occurrences.  
        `n_dw` is a number of the occurrences of the word `w` in the document `d`  
        this matrix determines the dependence between the Theta and Phi matrices  
        (Phi is the result of one iteration of the ARTM's EM algorihtm  
        with uniform theta initialization and `n_dw` matrix of the document-word occurrences)  
    """  # noqa: W291
    token2id = obtain_token2id(dataset)

    batch_vectorizer = dataset.get_batch_vectorizer()

    return _batch_vectorizer2sparse_matrix(
        batch_vectorizer, token2id, modality, modalities_to_use
    )


def _batch_vectorizer2sparse_matrix(batch_vectorizer, token2id, modality, modalities_to_use=None):
    """
    """
    theta_column_naming = 'id'  # scipy sparse matrix doesn't support non-integer indices
    matrix_row, matrix_col, matrix_data = [], [], []

    for batch_id in range(len(batch_vectorizer._batches_list)):
        batch_name = batch_vectorizer._batches_list[batch_id]._filename
        batch = artm.messages.Batch()
        with open(batch_name, "rb") as f:
            batch.ParseFromString(f.read())

        for item_id in range(len(batch.item)):
            item = batch.item[item_id]
            theta_item_id = getattr(item, theta_column_naming)

            for local_token_id, token_weight in zip(item.token_id, item.token_weight):
                token_class_id = batch.class_id[local_token_id]
                token = batch.token[local_token_id]
                if (token, token_class_id) not in token2id:
                    # probably dictionary was filtered
                    continue
                if modalities_to_use and token_class_id not in modalities_to_use:
                    continue
                if token_class_id != modality:
                    # we still need these tokens,
                    # shapes of n_dw matrix and actual Phi matrix should be in sync.
                    # this will be changed to zero at the end
                    token_weight = np.nan
                token_id = token2id[(token, token_class_id)]
                matrix_row.append(theta_item_id)
                matrix_col.append(token_id)
                matrix_data.append(token_weight)

    sparse_n_dw_matrix = scipy.sparse.csr_matrix(
        (matrix_data, (matrix_row, matrix_col)),
    )
    # remove the columns whose all elements are zero
    # (i.e. tokens which are of different modalities)
    # and renumber index (fill any "holes")
    # this is needed to be in sync with artm dictionary after filtering elements out
    # (they need to have the same shape)
    ind = sparse_n_dw_matrix.sum(axis=0)
    nonzeros = np.ravel(ind > 0)
    sparse_n_dw_matrix = sparse_n_dw_matrix[:, nonzeros]

    # re-encode values to transform NaNs to explicitly stored zeros
    sparse_n_dw_matrix.data = np.nan_to_num(sparse_n_dw_matrix.data)

    return sparse_n_dw_matrix


@jit(nopython=True)
def memory_efficient_inner1d(fst_arr, fst_indices, snd_arr, snd_indices):
    """
    Parameters
    ----------

    fst_arr: array-like
        2d array, shape is N x T
    fst_indices: array-like
        indices of the rows in fst_arr
    snd_arr: array-like
        2d array, shape is M x T
    snd_indices: array-like
        indices of the rows in fst_arr

    Returns
    -------
    np.array
        This is an array of the following form:
            np.array([
                sum(fst_arr[i, k] * snd_arr[j, k] for k in 0..T)
                for i, j in fst_indices, snd_indices
            ])
    """
    assert fst_arr.shape[1] == snd_arr.shape[1]
    assert len(fst_indices) == len(snd_indices)

    _, T = fst_arr.shape
    size = len(fst_indices)
    result = np.zeros(size)
    for i in range(size):
        fst_index = fst_indices[i]
        snd_index = snd_indices[i]
        for j in range(T):
            result[i] += fst_arr[fst_index, j] * snd_arr[snd_index, j]
    return result


@jit(nopython=True)
def _get_docptr(D, indptr):
    docptr = []
    for doc_num in range(D):
        docptr.extend(
            [doc_num] * (indptr[doc_num + 1] - indptr[doc_num])
        )
    return np.array(docptr, dtype=np.int32)


def get_docptr(n_dw_matrix):
    """
    Parameters
    ----------
    n_dw_matrix: array-like

    Returns
    -------
    np.array
        row indices for the provided matrix
    """
    return _get_docptr(n_dw_matrix.shape[0], n_dw_matrix.indptr)


def calc_docsizes(n_dw_matrix):
    D, _ = n_dw_matrix.shape
    docsizes = []
    indptr = n_dw_matrix.indptr
    for doc_num in range(D):
        size = indptr[doc_num + 1] - indptr[doc_num]
        value = np.sum(
            n_dw_matrix.data[indptr[doc_num]:indptr[doc_num + 1]]
        )
        docsizes.extend([value] * size)
    return np.array(docsizes)


def get_prob_matrix_by_counters(counters, inplace=False):
    if inplace:
        res = counters
    else:
        res = np.copy(counters)
    res[res < 0] = 0.
    # set rows where sum of row is small to uniform
    res[np.sum(res, axis=1) < EPS, :] = 1.
    res /= np.sum(res, axis=1)[:, np.newaxis]

    return res


def calc_A_matrix(
    n_dw_matrix, theta_matrix, docptr, phi_matrix_tr, wordptr
):
    s_data = memory_efficient_inner1d(
        theta_matrix, docptr,
        phi_matrix_tr, wordptr
    )
    return scipy.sparse.csr_matrix(
        (
            n_dw_matrix.data / (s_data + EPS),
            n_dw_matrix.indices,
            n_dw_matrix.indptr
        ),
        shape=n_dw_matrix.shape
    )


class ThetalessRegularizer(BaseRegularizer):
    def __init__(self, name, tau, modality, dataset: Dataset):
        """
        A regularizer based on a "thetaless" topic model inference

        Note: this implementation stores sparse `n_dw` matrix in memory,
        so this is not particularly memory- and space-efficient for huge datasets

        Parameters
        ----------
        name: str
            name of the regularizer
        tau: Number
            according to the math, `tau` should be set to 1 (to correctly emulate a different  
            inference process). But you do you, it's not like there's a regularizer  
            police or something.  
        modality: str
            name of modality on which the inference should be based
        dataset
            will be transformed to n_dw_matrix
        """  # noqa: W291
        super().__init__(name, tau)

        self.modality = modality
        self.modalities_to_use = None
        self.n_dw_matrix = None

        self.token2id = obtain_token2id(dataset)
        self._batches_path = os.path.join(dataset._internals_folder_path, "batches")

    def _initialize_matrices(self, batch_vectorizer, token2id):
        self.n_dw_matrix = _batch_vectorizer2sparse_matrix(
            batch_vectorizer, token2id, self.modality, self.modalities_to_use
        )
        self.B = scipy.sparse.csr_matrix(
            (
                1. * self.n_dw_matrix.data / calc_docsizes(self.n_dw_matrix),
                self.n_dw_matrix.indices,
                self.n_dw_matrix.indptr
            ),
            shape=self.n_dw_matrix.shape
        ).tocsc()
        self.docptr = get_docptr(self.n_dw_matrix)
        self.wordptr = self.n_dw_matrix.indices

    def grad(self, pwt, nwt):
        phi_matrix_tr = np.array(pwt)
        phi_matrix = phi_matrix_tr.T
        phi_rev_matrix = get_prob_matrix_by_counters(phi_matrix_tr)

        if self.n_dw_matrix.shape[1] != phi_rev_matrix.shape[0]:
            raise ValueError(
                f"Thetaless regularizer has prepared {self.n_dw_matrix.shape} n_dw matrix,"
                f" but was passed {phi_rev_matrix.T.shape} Phi matrix containing different"
                f" number of tokens ({self.n_dw_matrix.shape[1]} != {phi_rev_matrix.shape[0]})"
                f"\n(Are modalities the same?)"
            )

        theta_matrix = get_prob_matrix_by_counters(
            self.n_dw_matrix.dot(phi_rev_matrix)
        )

        A = calc_A_matrix(
            self.n_dw_matrix,
            theta_matrix,
            self.docptr,
            phi_matrix_tr,
            self.wordptr
        ).tocsc()

        n_tw = A.T.dot(theta_matrix).T * phi_matrix
        g_dt = A.dot(phi_matrix_tr)
        tmp = g_dt.T * self.B / (phi_matrix_tr.sum(axis=1) + EPS)
        n_tw += (tmp - np.einsum('ij,ji->i', phi_rev_matrix, tmp)) * phi_matrix

        return self.tau * (n_tw.T - nwt)

    def attach(self, model):
        """

        Parameters
        ----------
        model : ARTM model
            necessary to apply master component
        """
        if model.num_document_passes != 1:
            warnings.warn(
                f"num_document_passes is equal to {model.num_document_passes}, but it"
                f" should be set to {1} to correctly emulate a thetaless inference process"
            )

        self.modalities_to_use = model.class_ids.keys()
        bv = artm.BatchVectorizer(data_path=self._batches_path, data_format='batches')
        self._initialize_matrices(bv, self.token2id)

        self._model = model
