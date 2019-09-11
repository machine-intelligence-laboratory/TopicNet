from .base_viewer import BaseViewer


class TopTopicsFeatures(BaseViewer):
    """ """
    def __init__(self, dataset_id, model):
        super(TopTopicsFeatures, self).__init__(model=model)
        self._dataset = model.experiment.datasets[dataset_id]

    def view(self, document_id, topic_name=None, batch_vectorizer=None):
        """

        Parameters
        ----------
        document_id : str
            id of document
        topic_name : str
            (Default value = None)
        batch_vectorizer : optional
            (Default value = None)

        Returns
        -------
        result : dict

        """
        if topic_name is None:
            topic_name = (
                self._model
                .get_theta(dataset=self._dataset)[document_id]
                .idxmax()
            )
        phi_column = self._model.get_phi()[topic_name]
        src_text = self._dataset.get_source_document(document_id)
        result = {}
        for modality in phi_column.index.levels[0]:
            result[modality] = []
            tokens_weights = phi_column.loc[modality]
            for token in src_text[modality].split():
                if token in tokens_weights.index:
                    dropped = False
                    weight = tokens_weights.loc[token]
                else:
                    dropped = True
                    weight = 0
                result[modality].append((token, dropped, weight))
        return result
