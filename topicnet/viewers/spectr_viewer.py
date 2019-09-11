import numpy as np
from sklearn.manifold import TSNE


from .base_viewer import BaseViewer


class SpectrViewer(BaseViewer):
    """ """
    def __init__(self, model, method='tsne'):
        super().__init__(model=model)

        self.method = method

    def _get_tsne_spectr(self, phi):
        """

        Parameters
        ----------
        phi : np.array
            phi matrix

        Returns
        -------
        topic_spectr : np.array

        """
        projection = TSNE(n_components=1).fit_transform(phi.T).reshape(-1)
        permutation = np.argsort(projection)
        topics_spectr = np.array(self.model.model.topic_names)[permutation].tolist()
        return topics_spectr

    def view(self):
        """ """
        phi = self.model.get_phi()
        if self.method == 'tsne':
            return self._get_tsne_spectr(phi)
