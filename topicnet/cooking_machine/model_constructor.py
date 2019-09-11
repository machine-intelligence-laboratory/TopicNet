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
        name='PerplexityScore@all', dictionary=dictionary,
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
            name=f'PerplexityScore{modality}', dictionary=dictionary,
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
        num_processors=3,
        theta_columns_naming='title',
        show_progress_bars=False,
        class_ids=class_ids,
        seed=seed
    )

    return model


def init_simple_default_model(
        dictionary, modalities_to_use, main_modality,
        n_specific_topics, n_background_topics
):
    """
    Creates simple artm model with standart scores.

    """
    specific_topic_names = [
        f'topic_{i}'
        for i in range(n_specific_topics)
    ]
    background_topic_names = [
        f'background_{n_specific_topics + i}'
        for i in range(n_background_topics)
    ]

    baseline_class_ids = {class_id: 1 for class_id in modalities_to_use}
    tokens_data = count_vocab_size(dictionary, modalities_to_use)
    abs_weights = modality_weight_rel2abs(tokens_data, baseline_class_ids, main_modality)

    model = init_model(
        topic_names=specific_topic_names + background_topic_names,
        class_ids=abs_weights,
    )

    if n_background_topics > 0:
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
