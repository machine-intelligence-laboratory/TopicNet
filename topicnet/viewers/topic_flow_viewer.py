import numpy as np
import plotly.graph_objects as go
import artm
from .base_viewer import BaseViewer
from .top_tokens_viewer import TopTokensViewer


class TopicFlowViewer(BaseViewer):
    """
    Viewer to show trending topics over time.
    """
    def __init__(self, model, time_labels,
                 dataset,
                 modality='@lemmatized',
                 sort_key_function=None):
        """
        Parameters
        ----------
        model : TopicModel
            an instance of topic model class
        time_labels : list of numbers
            time label that supports comparrison for each document
        dataset : Dataset
            dataset used for model training (is used to compute nwd here)
        modality : str
            model's modality for topics description
        sort_key_function : Function
            function that can be used with python sorted
        """
        super().__init__(model)
        self.dataset = dataset

        theta = model.get_theta()
        self.unique_time_labels = sorted(np.unique(time_labels))

        attached_model_nwt = model._model.master.attach_model('nwt')
        nt = np.sum(attached_model_nwt[1], axis=0)
        nd = self.compute_nd(theta.shape[1])

        scaled_theta = theta.values * nd.reshape(1, -1)
        self.topic_values = np.zeros((theta.shape[0], len(self.unique_time_labels)))
        for time_ind, t in enumerate(self.unique_time_labels):
            indices = np.argwhere(time_labels == t)
            self.topic_values[:, time_ind] = (
                np.sum(scaled_theta[:, indices] / np.array(nt).reshape(-1, 1), axis=1)
            )
        self.topic_tokens_str = self.compute_top_tokens(model, modality)

    def compute_nd(self, number_of_docs):
        """
        Compute number of tokens in each document from dataset.

        Parameters
        ----------
        number_of_docs : int
            number of documents in theta
        """
        batches_list = self.dataset.get_batch_vectorizer().batches_ids
        nd = np.zeros(number_of_docs)

        current_doc = 0
        for batch_path in batches_list:
            batch = artm.messages.Batch()

            with open(batch_path, "rb") as f:
                batch.ParseFromString(f.read())

            for item in batch.item:
                doc_number_of_words = 0
                for (token_id, token_weight) in zip(item.token_id, item.token_weight):
                    doc_number_of_words += token_weight
                nd[current_doc] = doc_number_of_words
                current_doc += 1
        return nd

    def compute_top_tokens(self, model, modality):
        """
        Function for top tokens extraction.

        Parameters:
        ----------
        model : TopicModel
        modality : str
            modality for topic representation
        """
        top_tokens_viewer = TopTokensViewer(model)
        top_tokens_dict = top_tokens_viewer.view()
        topic_tokens_str = {}
        for topic, value in top_tokens_dict.items():
            topic_tokens_str[topic] = '<br>'.join(value[modality].keys())
        return topic_tokens_str

    def plot(self, topics, significance_threshold=1e-2):
        """
        Function for plotly graph building.

        Parameters
        ----------
        topics : list of int
            topics that need to be visualized
        significance_threshold : float
            plot ignores values lower than threshold
        """
        fig = go.Figure()

        for t in topics:
            fig.add_trace(go.Scatter(x=np.arange(len(self.unique_time_labels)),
                                     y=[
                                         value if value > significance_threshold
                                         else None
                                         for value in self.topic_values[t, :]
                                     ],
                                     text=self.topic_tokens_str[f'topic_{t}'],
                                     hoverinfo='text',
                                     mode=None,
                                     hoveron='points+fills',
                                     fill='tozeroy',
                                     name=f'topic_{t}'))

        fig.update_layout(
            title='Trending Topics Over Time',
            title_font_size=30,
            autosize=True,
            paper_bgcolor='LightSteelBlue'
        )

        fig.update_xaxes(title_text='Time',
                         tickvals=np.arange(len(self.unique_time_labels))[::4],
                         ticktext=self.unique_time_labels[::4])
        fig.update_yaxes(title_text='Value')
        fig.show()

    def view(self, topic_names=None):
        """
        Parameters
        ----------
        topic_names : list of str
            topics that user wants to see on plot
        """
        topics = list(map(lambda x: int(x.split('_')[1]), topic_names))
        self.plot(topics)
