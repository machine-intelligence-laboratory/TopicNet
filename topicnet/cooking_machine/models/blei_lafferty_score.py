import numpy as np
from .base_score import BaseScore


class BleiLaffertyScore(BaseScore):
    """
    This score implements method described in 2009 paper
    Blei, David M., and John D. Laﬀerty. "Topic models." Text Mining.
    Chapman and Hall/CRC, 2009. 101-124.
    At the core this score helps to discover tokens that are most likely
    to describe given topic. Summing up that score helps to estimate how
    well the model distinguishes between topics. The higher this score - better
    """
    def __init__(self, num_top_tokens: int = 30):
        """

        Parameters
        ----------
        num_top_tokens : int
            now many tokens we consider to be

        """
        super().__init__()
        self.num_top_tokens = num_top_tokens

    def _compute_blei_scores(self, phi):
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

        topic_number = phi.shape[1]
        blei_eps = 1e-42
        log_phi = np.log(phi + blei_eps)
        numerator = np.sum(log_phi, axis=1)
        numerator = numerator[:, np.newaxis]

        if hasattr(log_phi, "values"):
            multiplier = log_phi.values - numerator / topic_number
        else:
            multiplier = log_phi - numerator / topic_number

        scores = phi * multiplier
        return scores

    def call(self, model):
        modalities = list(model.class_ids.keys())

        score = 0
        for modality in modalities:
            phi = model.get_phi(class_ids=modality)
            modality_scores = np.sort(self._compute_blei_scores(phi).values)
            score += np.sum(modality_scores[-self.num_top_tokens:, :])
        if modalities is None:
            phi = model.get_phi()
            modality_scores = np.sort(self._compute_blei_scores(phi).values)
            score = np.sum(modality_scores[-self.num_top_tokens:, :])
        return score
