from .rel_toolbox_lite import count_vocab_size, modality_weight_rel2abs
import artm

# change log style
lc = artm.messages.ConfigureLoggingArgs()
lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)


def add_standard_scores(
        model,
        dictionary,
        main_modality="@lemmatized",
        all_modalities=("@lemmatized", "@ngramms")
):
    """
    Adds standard scores for the model.

    """
    assert main_modality in all_modalities, "main_modality must be part of all_modalities"

    model.scores.add(artm.scores.PerplexityScore(
        name='PerplexityScore@all',
        class_ids=all_modalities
    ))

    model.scores.add(
        artm.scores.SparsityThetaScore(name='SparsityThetaScore')
    )

    for modality in all_modalities:
        model.scores.add(artm.scores.SparsityPhiScore(
            name=f'SparsityPhiScore{modality}', class_id=modality)
        )
        model.scores.add(artm.scores.PerplexityScore(
            name=f'PerplexityScore{modality}',
            class_ids=[modality]
        ))
        model.scores.add(
            artm.TopicKernelScore(name=f'TopicKernel{modality}',
                                  probability_mass_threshold=0.3, class_id=modality)
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
        dataset, modalities_to_use, main_modality,
        specific_topics, background_topics,
        modalities_weights=None
):
    """
    Creates simple artm model with standard scores.

    Parameters
    ----------
    dataset : Dataset
    modalities_to_use : list of str
    main_modality : str
    specific_topics : list or int
    background_topics : list or int
    modalities_weights : dict or None

    Returns
    -------
    model: artm.ARTM() instance
    """
    if modalities_weights is not None:
        assert sorted(list(modalities_to_use)) == sorted(list(modalities_weights.keys()))
    baseline_class_ids = {class_id: 1 for class_id in modalities_to_use}

    specific_topic_names, background_topic_names = create_default_topics(
        specific_topics, background_topics
    )
    dictionary = dataset.get_dictionary()

    tokens_data = count_vocab_size(dictionary, modalities_to_use)
    abs_weights = modality_weight_rel2abs(
        tokens_data,
        modalities_weights if modalities_weights is not None else baseline_class_ids,
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
    add_standard_scores(model, dictionary, main_modality=main_modality,
                        all_modalities=modalities_to_use)

    return model
