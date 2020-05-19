import numpy as np
from numba import jit
import scipy.sparse

from .base_regularizer import BaseRegularizer


EPS = 1e-20


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
    def __init__(self, name, tau, n_dw_matrix):
        """
        Creates a node in the graph with the given args and kwargs.

        Parameters
        ----------
        name: str
            name of the regularizer
        tau: Number
            fictive parameter it's not used, just passed to the parent conctructor
        n_dw_matrix: scipy.sparse.csr_matrix
            The matrix of document-word occurrences
            n_dw is a number of the occurrences of the word w in the document d
            this matrix determines the dependence between the Theta and Phi matrices
            (Phi is the result of one iteration of the ARTM's EM algorihtm
            with uniform theta initialization and n_dw matrix of the document-word occurrences)
        """
        super().__init__(name, tau)
        self.n_dw_matrix = n_dw_matrix
        self.B = scipy.sparse.csr_matrix(
            (
                1. * n_dw_matrix.data / calc_docsizes(n_dw_matrix),
                n_dw_matrix.indices,
                n_dw_matrix.indptr
            ),
            shape=n_dw_matrix.shape
        ).tocsc()
        self.docptr = get_docptr(n_dw_matrix)
        self.wordptr = n_dw_matrix.indices

    def grad(self, pwt, nwt):
        phi_matrix_tr = np.array(pwt)
        phi_matrix = phi_matrix_tr.T
        phi_rev_matrix = get_prob_matrix_by_counters(phi_matrix_tr)
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

        return n_tw.T - nwt
