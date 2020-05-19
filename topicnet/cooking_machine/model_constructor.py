import warnings

from typing import (
    Dict,
    List,
)

import artm

from .dataset import Dataset
from .rel_toolbox_lite import (
    count_vocab_size,
    modality_weight_rel2abs,
)


def add_standard_scores(
        model: artm.ARTM,
        dictionary: artm.Dictionary = None,
        main_modality: str = "@lemmatized",
        all_modalities: List[str] = ("@lemmatized", "@ngramms")
) -> None:
    """
    Adds standard scores for the model.

    Parameters
    ----------
    model
    dictionary
        Obsolete parameter, not used
    main_modality
    all_modalities
    """
    assert main_modality in all_modalities, "main_modality must be part of all_modalities"

    if dictionary is not None:
        warnings.warn(
            'Parameter `dictionary` is obsolete:'
            ' it is not used in the function "add_standard_scores"!'
        )

    model.scores.add(
        artm.scores.PerplexityScore(
            name='PerplexityScore@all',
            class_ids=all_modalities,
        )
    )

    model.scores.add(
        artm.scores.SparsityThetaScore(name='SparsityThetaScore')
    )

    for modality in all_modalities:
        model.scores.add(
            artm.scores.SparsityPhiScore(
                name=f'SparsityPhiScore{modality}',
                class_id=modality,
            )
        )
        model.scores.add(
            artm.scores.PerplexityScore(
                name=f'PerplexityScore{modality}',
                class_ids=[modality],
            )
        )
        model.scores.add(
            artm.TopicKernelScore(
                name=f'TopicKernel{modality}',
                probability_mass_threshold=0.3,
                class_id=modality,
            )
        )


def init_model(topic_names, seed=None, class_ids=None):
    """
    Creates basic artm model

    """
    model = artm.ARTM(
        topic_names=topic_names,
        # Commented for performance uncomment if has zombie issues
        # num_processors=3,
        theta_columns_naming='title',
        show_progress_bars=False,
        class_ids=class_ids,
        seed=seed
    )

    return model


def create_default_topics(specific_topics, background_topics):
    """
    Creates list of background topics and specific topics

    Parameters
    ----------
    specific_topics : list or int
    background_topics : list or int

    Returns
    -------
    (list, list)
    """
    # TODO: what if specific_topics = 4
    # and background_topics = ["topic_0"] ?
    if isinstance(specific_topics, list):
        specific_topic_names = list(specific_topics)
    else:
        specific_topics = int(specific_topics)
        specific_topic_names = [
            f'topic_{i}'
            for i in range(specific_topics)
        ]
    n_specific_topics = len(specific_topic_names)
    if isinstance(background_topics, list):
        background_topic_names = list(background_topics)
    else:
        background_topics = int(background_topics)
        background_topic_names = [
            f'background_{n_specific_topics + i}'
            for i in range(background_topics)
        ]
    if set(specific_topic_names) & set(background_topic_names):
        raise ValueError(
            "Specific topic names and background topic names should be distinct from each other!"
        )

    return specific_topic_names, background_topic_names


def init_simple_default_model(
        dataset: Dataset,
        modalities_to_use: List[str] or Dict[str, float],
        main_modality: str,
        specific_topics: List[str] or int,
        background_topics: List[str] or int,
) -> artm.ARTM:
    """
    Creates simple `artm.ARTM` model with standard scores.

    Parameters
    ----------
    dataset
        Dataset for model initialization
    modalities_to_use
        What modalities a model should know.
        If `modalities_to_use` is a dictionary,
        all given weights are assumed to be relative to `main_modality`:
        weights will then be recalculated to absolute ones
        using `dataset` and `main_modality`.
        If `modalities_to_use` is a list,
        then all relative weights are set equal to one.

        The result model's `class_ids` field will contain absolute modality weights.
    main_modality
        Modality relative to which all modality weights are considered
    specific_topics
        Specific topic names or their number
    background_topics
        Background topic names or their number

    Returns
    -------
    model : artm.ARTM

    """
    if isinstance(modalities_to_use, dict):
        modalities_weights = modalities_to_use
    else:
        modalities_weights = {class_id: 1 for class_id in modalities_to_use}

    specific_topic_names, background_topic_names = create_default_topics(
        specific_topics, background_topics
    )
    dictionary = dataset.get_dictionary()

    tokens_data = count_vocab_size(dictionary, modalities_to_use)
    abs_weights = modality_weight_rel2abs(
        tokens_data,
        modalities_weights,
        main_modality
    )

    model = init_model(
        topic_names=specific_topic_names + background_topic_names,
        class_ids=abs_weights,
    )

    if len(background_topic_names) > 0:
        model.regularizers.add(
            artm.SmoothSparsePhiRegularizer(
                 name='smooth_phi_bcg',
                 topic_names=background_topic_names,
                 tau=0.0,
                 class_ids=[main_modality],
            ),
        )
        model.regularizers.add(
            artm.SmoothSparseThetaRegularizer(
                 name='smooth_theta_bcg',
                 topic_names=background_topic_names,
                 tau=0.0,
            ),
        )

    model.initialize(dictionary)
    add_standard_scores(model, main_modality=main_modality,
                        all_modalities=modalities_to_use)

    return model
