import artm
import pandas as pd
import numpy as np
from collections import Counter
from scipy.optimize import curve_fit
from .base_score import BaseScore

# change log style
lc = artm.messages.ConfigureLoggingArgs()
lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)


def calculate_n(model, batch_vectorizer):
    """
    Calculate all necessary statistics from batch. This may take some time.
    """
    tokens = []
    for batch_id in range(len(batch_vectorizer._batches_list)):
        batch_name = batch_vectorizer._batches_list[batch_id]._filename
        batch = artm.messages.Batch()
        with open(batch_name, "rb") as f:
            batch.ParseFromString(f.read())

        for item_id in range(len(batch.item)):
            item = batch.item[item_id]
            for token_id in item.token_id:
                tokens.append(batch.token[token_id])

    ntdw = model.transform(batch_vectorizer=batch_vectorizer, theta_matrix_type='dense_ptdw')
    docs = ntdw.columns
    ntdw.columns = pd.MultiIndex.from_arrays([docs, tokens], names=('doc', 'token'))

    ntd = ntdw.groupby(level=0, axis=1).sum()

    nwt = ntdw.groupby(level=1, axis=1).sum().T

    nt = nwt.sum(axis=0)

    return ntdw, ntd, nwt, nt


def synthetic_doc_ntdw_and_ntd(doc_len, nwt):
    """
    Create synthetic document from nwt with specific doc_len.
    """
    pwt = np.float64(nwt) / np.sum(np.float64(nwt)).astype(float)
    doc_idx = np.random.choice(len(pwt), doc_len, p=pwt)
    doc_count = dict(Counter(doc_idx))

    ntdw = np.empty((len(pwt)))
    for word_idx in range(len(ntdw)):
        ntdw[word_idx] = doc_count.get(word_idx, 0)
    ntd = np.sum(ntdw)

    return ntdw, ntd


def cressie_reed_sampled(topic, ntdw_calc, ntd_calc, nwt, nt, gimel=-1/2):
    """
    Calculate Cressie-Reed divergence for sampled pseudo-document.
    """
    mul_part = ntd_calc * nwt.iloc[:, topic]

    if np.all(ntdw_calc == 0) or nt[topic] == 0 or np.all(mul_part == 0):
        gimel_part = np.array([0])
    else:
        gimel_part = 0
        for token_id, token in enumerate(nwt.index):
            token_ntdw = ntdw_calc[token_id]
            token_denom = mul_part.iloc[token_id]
            if token_ntdw and token_denom:
                gimel_part += token_ntdw * (
                    np.power(token_ntdw * nt[topic] / token_denom, gimel) - 1
                )

    cressie_reed_for_l = 2 / (gimel * (gimel + 1)) * np.sum(gimel_part)

    return cressie_reed_for_l


def third_degree(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3


def radius_vs_ndt(topic, max_len, sample_step, sample_size, nwt, nt, alpha):
    """
    Calculate third degree approximation for radius vs ndt dependency.
    """
    crs_for_alpha = []
    ntds_sampled = []
    for doc_len in range(1, max_len, sample_step):
        local_crs_for_alpha = []
        for _ in range(sample_size):
            ntdw_sampled, ntd_sampled = synthetic_doc_ntdw_and_ntd(doc_len, nwt.iloc[:, topic])
            local_crs_for_alpha.append(cressie_reed_sampled(
                topic, ntdw_sampled, ntd_sampled, nwt, nt
            ))

        crs_for_alpha.append(np.quantile(local_crs_for_alpha, 1 - alpha))
        ntds_sampled.append(ntd_sampled)

    regression_coeff, cov = curve_fit(third_degree, ntds_sampled, crs_for_alpha)
    return regression_coeff


def radii_vs_ntd(max_len, sample_step, sample_size, nwt, nt, alpha):
    regression_coeffs = []
    for topic in range(len(nt)):
        regression_coeffs.append(radius_vs_ndt(
            topic, max_len, sample_step, sample_size, nwt, nt, alpha
        ))

    return regression_coeffs


def radius_for_ntd(ntd, regression_coeff):
    return third_degree(ntd, *regression_coeff)


def radii_for_ntd(ntd, regression_coeff):
    return ntd.apply(lambda x: third_degree(x, *regression_coeff))


class SemanticRaduisScore(BaseScore):
    """
    This score implements cluster semantic radius, described in paper
    'Проверка гипотезы условной независимости 
    для оценивания качества тематической кластеризации' by Rogozina A.
    At the core this score helps to discover topics uniformity.
    The lower this score - better
    """  # noqa: W291
    def __init__(self, batch_vectorizer):
        """

        Parameters
        ----------
        batch_vectorizer

        """
        super().__init__()
        self.batch_vectorizer = batch_vectorizer

    def update(self, score):
        known_errors = (ValueError, TypeError)
        try:
            score = np.array(score, float)
        except known_errors:
            raise ValueError(f'Score call should return list of float but not {score}')
        self.value.append(score)

    def call(self, model, max_sampled_document_len=None, sample_step=5, sample_size=3, alpha=0.1):
        """

        Parameters
        ----------
        model : TopicModel
        max_sampled_document_len : int
            Maximum length of pseudo-document for quantile regression
            (Default value = None)
        sample_step : int
            Grain for quantile regression
            (Default value = 5)
        sample_size : int
            Size of every sample for quantile regression  
            (Default value = 3)
        alpha : float
            (1 - alpha) quantile level, must be <= 1  
            (Default value = 0.1)

        """  # noqa: W291
        ntdw, ntd, nwt, nt = calculate_n(model._model, self.batch_vectorizer)

        if max_sampled_document_len is None:
            max_sampled_document_len = int(np.max(ntd.values))

        regression_coeffs = radii_vs_ntd(
            max_sampled_document_len, sample_step, sample_size, nwt, nt, alpha
        )
        radii = [
            radius_for_ntd(topic_ntd, coeff)
            for topic_ntd, coeff
            in zip(ntd.values.mean(axis=1), regression_coeffs)
        ]

        return radii
