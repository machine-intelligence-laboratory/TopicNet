from ..cooking_machine.models import TopicModel
from gensim.models import KeyedVectors


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
    words = phi.index.levels[1]
    model = KeyedVectors(len(topics))
    model.add(words, phi)
    return model
