
ARTM_baseline = '''
# This config follows a strategy described by Murat Apishev
# one of the core programmers of BigARTM library in personal correspondence.
# According to his letter 'decent' topic model can be obtained by
# Decorrelating model topics simultaneously looking at retrieved TopTokens


# Use .format(modality_list=modality_list, main_modality=main_modality, dataset_path=dataset_path,
# specific_topics=specific_topics, background_topics=background_topics)
# when loading the recipe to adjust for your dataset

topics:
# Describes number of model topics, better left to the user to define optimal topic number
    specific_topics: {specific_topics} 
    background_topics: {background_topics}

# Here is example of model with one modality
regularizers:
    - DecorrelatorPhiRegularizer:
        name: decorrelation_phi
        topic_names: specific_topics
        class_ids: {modality_list}
    - SmoothSparsePhiRegularizer:
        name: smooth_phi_bcg
        topic_names: background_topics
        class_ids: {modality_list}
        tau: 0.1
        relative: true
    - SmoothSparseThetaRegularizer:
        name: smooth_theta_bcg
        topic_names: background_topics
        tau: 0.1
        relative: true
scores:
    - BleiLaffertyScore:
        num_top_tokens: 30
model: 
    dataset_path: {dataset_path}
    modalities_to_use: {modality_list}
    main_modality: '{main_modality}'

stages:
- RegularizersModifierCube:
    num_iter: 20
    reg_search: add
    regularizer_parameters:
        name: decorrelation_phi
    selection:
        - PerplexityScore@all < 1.05 * MINIMUM(PerplexityScore@all) and BleiLaffertyScore -> max
    strategy: PerplexityStrategy
    # parameters of this strategy are intended for revision
    strategy_params:
        start_point: 0
        step: 0.01
        max_len: 50
    tracked_score_function: PerplexityScore@all
    verbose: false
    use_relative_coefficients: true
'''

exploratory_search = '''
# This config follows a strategy described in the article
# Multi-objective Topic Modeling for Exploratory Search in Tech News
# by Anastasya Yanina, Lev Golitsyn and Konstantin Vorontsov, Jan 2018


# Use .format(modality=modality, dataset_path=dataset_path,
# specific_topics=specific_topics, background_topics=background_topics)
# when loading the recipe to adjust for your dataset

topics:
# Describes number of model topics, in the actuall article 200 topics were found to be optimal
    specific_topics: {specific_topics}
    background_topics: {background_topics}

regularizers:
- DecorrelatorPhiRegularizer:
    name: decorrelation_phi_{modality}
    topic_names: specific_topics
    tau: 1
    class_ids: ['{modality}']
- SmoothSparsePhiRegularizer:
    name: smooth_phi_{modality}
    topic_names: specific_topics
    tau: 1
    class_ids: ['{modality}']
- SmoothSparseThetaRegularizer:
    name: sparse_theta
    topic_names: specific_topics
    tau: 1

model: 
    dataset_path: {dataset_path}
    modalities_to_use: ['{modality}']
    main_modality: '{modality}'

stages:
# repeat the following two cubes for every modality in the dataset
- RegularizersModifierCube:
    num_iter: 8
    reg_search: mul
    regularizer_parameters:
        name: decorrelation_phi_{modality}
    selection:
        - PerplexityScore{modality} < 1.01 * MINIMUM(PerplexityScore{modality}) and SparsityPhiScore{modality} -> max
    strategy: PerplexityStrategy
    strategy_params:
        start_point: 100000
        step: 10
        max_len: 6
    tracked_score_function: PerplexityScore@all
    verbose: false
    use_relative_coefficients: false
- RegularizersModifierCube:
    num_iter: 8
    reg_search: add
    regularizer_parameters:
        name: smooth_phi_{modality}
    selection:
        - PerplexityScore{modality} < 1.01 * MINIMUM(PerplexityScore{modality}) and SparsityPhiScore{modality} -> max
    strategy: PerplexityStrategy
    strategy_params:
        start_point: 0.25
        step: 0.25
        max_len: 6
    tracked_score_function: PerplexityScore{modality}
    verbose: false
    use_relative_coefficients: false
#last cube is independent of modalities and can be used only once
- RegularizersModifierCube:
    num_iter: 8
    reg_search: add
    regularizer_parameters:
        name: sparse_theta
    selection:
        - PerplexityScore@all < 1.01 * MINIMUM(PerplexityScore@all) and SparsityPhiScore{modality} -> max
    strategy: PerplexityStrategy
    strategy_params:
        start_point: -0.5
        step: -0.5
        max_len: 6
    tracked_score_function: PerplexityScore@all
    verbose: false
    use_relative_coefficients: false

'''
