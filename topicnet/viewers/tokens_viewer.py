from .base_viewer import BaseViewer


# example with result caching
class TokensViewer(BaseViewer):
    """ """
    def view(self):
        """
        Class method, returning tokens for every modality

        """
        phi = self._model.get_phi()

        result = {}
        for modality, token in phi.index:
            result.setdefault(modality, [])
            result[modality].append(token)

        return result
