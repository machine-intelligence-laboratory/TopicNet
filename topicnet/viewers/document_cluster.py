import numpy as np
import colorlover as cl
import plotly.graph_objs as go
import sklearn.manifold as clusterization

from plotly.offline import plot, iplot
from .base_viewer import BaseViewer
from functools import partial


class DocumentClusterViewer(BaseViewer):
    """
    This viewer performs dimesionality reduction over document embeddings
    """
    def __init__(self, model):
        """
        Parameters
        ----------

        model: TopicModel

        """
        super().__init__(model=model)

    def view(
            self,
            dataset,
            save_path,
            method='TSNE',
            to_html=True,
    ):
        """
        Parameters
        ----------
        dataset: Dataset
        save_path: str
            save path for the plot
        method: string
            any of the methods in sklearn.manifold
        to_html: Bool
            if user wants the plot to be saved in html format

        Returns
        -------
        reduced_data: an np.array of (num_docs, dim) dimensions
            reduced dumensions of the original document embeddings
        html_div: string
            an html string containing the plotly graph
            returned only if to_html is True

        """
        from ..cooking_machine.dataset import BaseDataset
        if not isinstance(dataset, BaseDataset):
            raise TypeError('Parameter "dataset" should derive from BaseDataset')

        handler = getattr(clusterization, method,)
        bv = dataset.get_batch_vectorizer()
        model_data = self._model.transform(batch_vectorizer=bv).T

        reduced_data = handler(n_components=2).fit_transform(model_data)
        data_dict = {}
        data_dict['x'] = reduced_data[:, 0]
        data_dict['y'] = reduced_data[:, 1]
        data_dict['label'] = np.argmax(model_data.values, axis=1)
        data_dict['text'] = model_data.index
        base_scheme = cl.scales['12']['qual']['Paired']
        if not to_html:
            drawing_handle = partial(iplot, show_link=False,)
            save_path = None
        else:
            drawing_handle = partial(plot, show_link=False, output_type='div')

        html_div = drawing_handle(
            [go.Scatter(
                x=data_dict['x'],
                y=data_dict['y'],
                mode='markers',
                marker=dict(colorscale=base_scheme,
                            size=4,
                            opacity=0.6,
                            colorbar=dict(title='Topics')),
                marker_color=data_dict['label'],
                text=data_dict['text'],)],
        )
        if save_path is not None:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_div)

        if to_html:
            return html_div

        return reduced_data

    def viev_from_jupyter(
        self,
        dataset,
        method: str = 'TSNE',
        save_path: str = 'DocumentCluster_view.html',
        width: int = 800,
        height: int = 600,
        display_output: bool = True,
        give_html: bool = False,
    ):
        """
        Parameters
        ----------
        dataset: Dataset
        method: string
            any of the methods in sklearn.manifold
        to_html: Bool
            if user wants the plot to be saved in html format
        save_path: str
            save path for the plot requires to be able to create
            the visualisation
        width
            width of the plot in pixels
        height
            height of the plot in pixels
        display_output
            show the plot in the notebook
        give_html
            if return the html string (with javascript) that
            performs the visualisation

        Returns
        -------
        out_html: string
            an html string containing the plotly graph
            returned only if give_html is True

        """
        from IPython.display import IFrame, display_html
        out_html = self.view(
            dataset=dataset,
            method=method,
            to_html=True,
        )
        if display_output:
            display_html(IFrame(save_path, width=width, height=height))

        if give_html:
            return out_html
