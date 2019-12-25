"""
Parsing text file into Experiment instance using strictyaml
(github.com/crdoconnor/strictyaml/)

The aim here is to make config:
* possible to use even for non-programmers
* hard to misuse
* easy debuggable

Hence, the process of parsing config is a bit more complicated than
it could be, but it produces more useful error messages. For example:

    File $YOUR_CONFIG.yaml, line 42
        topic_names: 10
        ^ this value should be a 'list' instead of 'int'
    YAMLValidationError: 'int' passed instead of 'list'

instead of:

    File $SOME_FILE.py, line 666, in $SOME_FUNCTION
        for topic_name in topic_names:
    TypeError: 'int' object is not iterable

To achieve this, strictyaml makes use of various validators which
keep track of individual line numbers and which fragments are already
checked and which aren't quite here yet.

Our process consists of three stages:
1) we check the high-level structure using `base_schema`.
The presence of each required key is ensured.
After this stage we could be sure than we can create a valid model
using specified parameters.

2) we make a second pass and revalidate 'regularizers' and 'stages'
This step is performed semi-automatically: using `inspect`,
we extract everything from `__init__` method signature.
    For example:
        def __init__(self, num_iters: int = 5)
    allows us to infer that num_iters parameter should be int,
    but it isn't strictly required.

3) we construct instances of classes required, convert types manually
and implement some shortcuts.
Ideally, this stage should be performed using revalidate() as well,
but it's a work-in-progress currently.

"""  # noqa: W291
from .cubes import RegularizersModifierCube, CubeCreator
from .experiment import Experiment
from .dataset import Dataset
from .models import scores as tnscores
from .models import TopicModel

from .cubes import PerplexityStrategy, GreedyStrategy
from .model_constructor import init_simple_default_model, create_default_topics
from .rel_toolbox_lite import count_vocab_size, handle_regularizer

import artm

from inspect import signature, Parameter
from strictyaml import Map, Str, Int, Seq, Float, Bool
from strictyaml import Any, Optional, EmptyDict, EmptyNone, EmptyList
from strictyaml import dirty_load

# TODO: use stackoverflow.com/questions/37929851/parse-numpydoc-docstring-and-access-components
# for now just hardcode most common / important types
ARTM_TYPES = {
    "tau": Float(),
    "topic_names": Str() | Seq(Str()) | EmptyNone(),
    # TODO: handle class_ids in model and in regularizers separately
    "class_ids": Str() | Seq(Str()) | EmptyNone(),
    "gamma": Float() | EmptyNone(),
    "seed": Int(),
    "num_document_passes": Int(),
    "num_processors": Int(),
    "cache_theta": Bool(),
    "reuse_theta": Bool(),
    "theta_name": Str()
}

# change log style
lc = artm.messages.ConfigureLoggingArgs()
lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)

element = Any()
base_schema = Map({
    'regularizers': Seq(element),
    Optional('scores'): Seq(element),
    'stages': Seq(element),
    'model': Map({
        "dataset_path": Str(),
        Optional("modalities_to_use"): Seq(Str()),
        Optional("modalities_weights"): Any(),
        "main_modality": Str(),
    }),
    'topics': Map({
        "background_topics": Seq(Str()) | Int() | EmptyList(),
        "specific_topics": Seq(Str()) | Int() | EmptyList(),
    })
})
SUPPORTED_CUBES = [CubeCreator, RegularizersModifierCube]
SUPPORTED_STRATEGIES = [PerplexityStrategy, GreedyStrategy]


def choose_key(param):
    """
    Parameters
    ----------
    param : inspect.Parameter

    Returns
    -------
    str or strictyaml.Optional
    """
    if param.default is not Parameter.empty:
        return Optional(param.name)
    return param.name


def choose_validator(param):
    """
    Parameters
    ----------
    param : inspect.Parameter

    Returns
    -------
    instance of strictyaml.Validator
    """
    if param.annotation is int:
        return Int()
    if param.annotation is float:
        return Float()
    if param.annotation is bool:
        return Bool()
    if param.annotation is str:
        return Str()
    if param.name in ARTM_TYPES:
        return ARTM_TYPES[param.name]
    return Any()


def build_schema_from_signature(class_of_object, use_optional=True):
    """
    Parameters
    ----------
    class_of_object : class

    Returns
    -------
    dict
        each element is either str -> Validator or Optional(str) -> Validator
    """
    choose_key_func = choose_key if use_optional else (lambda param: param.name)
    return {choose_key_func(param): choose_validator(param)
            for param in signature(class_of_object.__init__).parameters.values()
            if param.name != 'self'}


def wrap_in_map(dictionary):
    could_be_empty = all(isinstance(key, Optional) for key in dictionary)
    if could_be_empty:
        return Map(dictionary) | EmptyDict()
    return Map(dictionary)


def build_schema_for_scores():
    """
    Returns
    -------
    strictyaml.Map
        schema used for validation and type-coercion
    """
    schemas = {}
    for elem in artm.scores.__all__:
        if "Score" in elem:
            class_of_object = getattr(artm.scores, elem)
            # TODO: check if every key is Optional. If it is, then "| EmptyDict()"
            # otherwise, just Map()
            res = wrap_in_map(build_schema_from_signature(class_of_object))

            specific_schema = Map({class_of_object.__name__: res})
            schemas[class_of_object.__name__] = specific_schema

    for elem in tnscores.__all__:
        if "Score" in elem:
            class_of_object = getattr(tnscores, elem)
            res = build_schema_from_signature(class_of_object)
            # res["name"] = Str()  # TODO: support custom names
            res = wrap_in_map(res)

            specific_schema = Map({class_of_object.__name__: res})
            schemas[class_of_object.__name__] = specific_schema

    return schemas


def build_schema_for_regs():
    """
    Returns
    -------
    strictyaml.Map
        schema used for validation and type-coercion
    """
    schemas = {}
    for elem in artm.regularizers.__all__:
        if "Regularizer" in elem:
            class_of_object = getattr(artm.regularizers, elem)
            res = build_schema_from_signature(class_of_object)
            if elem in ["SmoothSparseThetaRegularizer", "SmoothSparsePhiRegularizer",
                        "DecorrelatorPhiRegularizer"]:
                res[Optional("relative", default=None)] = Bool()
            res = wrap_in_map(res)

            specific_schema = Map({class_of_object.__name__: res})
            schemas[class_of_object.__name__] = specific_schema

    return schemas


def is_key_in_schema(key, schema):
    if key in schema:
        return True
    return any(
        key_val.key == key for key_val in schema
        if isinstance(key_val, Optional)
    )


def build_schema_for_cubes():
    """
    Returns
    -------
    dict
        each element is str -> strictyaml.Map
        where key is name of cube,
        value is a schema used for validation and type-coercion
    """
    schemas = {}
    for class_of_object in SUPPORTED_CUBES:
        res = build_schema_from_signature(class_of_object)

        # "selection" isn't used in __init__, but we will need it later
        res["selection"] = Seq(Str())

        # shortcut for strategy intialization
        if is_key_in_schema("strategy", res):
            signature_validation = {}
            for strategy_class in SUPPORTED_STRATEGIES:
                local_signature_validation = build_schema_from_signature(strategy_class)
                signature_validation.update(local_signature_validation)
            res[Optional("strategy_params")] = Map(signature_validation)

        # we will deal with "values" later, but we can check at least some simple things already
        if class_of_object.__name__ == "CubeCreator":
            element = Map({"name": Str(), "values": Seq(Any())})
            res["parameters"] = Seq(element)
        if class_of_object.__name__ == "RegularizersModifierCube":
            element = Map({
                Optional("name"): Str(),
                Optional("regularizer"): Any(),
                Optional("tau_grid"): Seq(Float())
            })
            res["regularizer_parameters"] = element | Seq(element)

        res = Map(res)

        specific_schema = Map({class_of_object.__name__: res})
        schemas[class_of_object.__name__] = specific_schema
    return schemas


def preprocess_parameters_for_cube_creator(elem_args):
    """
    This function does two things:
        1) convert class_ids from
            name: class_ids@text, values: [0, 1, 2, 3]
           to
            name: class_ids, values: {"@text": [0, 1, 2, 3]}
        2) type conversion for "values" field.

    Parameters
    ----------
    elem_args: strictyaml.YAML object
        (contains dict inside)

    Returns
    -------
    new_elem_args: dict
    """

    for param_portion in elem_args["parameters"]:
        name = str(param_portion["name"])
        if name.startswith("class_ids"):
            validator = Float() | Seq(Float())
        else:
            validator = Seq(ARTM_TYPES[name])
        param_schema = Map({
            "name": Str(),
            "values": validator
        })
        param_portion.revalidate(param_schema)


def handle_special_cases(elem_args, kwargs):
    """
    In-place fixes kwargs, handling special cases and shortcuts
    (only strategy for now)
    Parameters
    ----------
    elem_args: dict
    kwargs: dict
    """
    # special case: shortcut for strategy
    if "strategy" in elem_args:
        strategy = None
        for strategy_class in SUPPORTED_STRATEGIES:
            if strategy_class.__name__ == elem_args["strategy"]:
                strat_schema = build_schema_from_signature(strategy_class, use_optional=False)
                strat_kwargs = {}

                for key, value in elem_args["strategy_params"].items():
                    key = str(key)
                    value.revalidate(strat_schema[key])
                    strat_kwargs[key] = value.data

                strategy = strategy_class(**strat_kwargs)

        kwargs["strategy"] = strategy  # or None if failed to identify it


def build_score(elemtype, elem_args, is_artm_score):
    """
    Parameters
    ----------
    elemtype : str
        name of score
    elem_args: dict
    is_artm_score: bool

    Returns
    -------
    instance of artm.Score or topicnet.BaseScore
    """
    module = artm.scores if is_artm_score else tnscores
    class_of_object = getattr(module, elemtype)
    kwargs = {name: value
              for name, value in elem_args.items()}

    return class_of_object(**kwargs)


def build_regularizer(elemtype, elem_args, specific_topic_names, background_topic_names):
    """
    Parameters
    ----------
    elemtype : str
        name of regularizer
    elem_args: dict
    parsed: strictyaml.YAML object

    Returns
    -------
    instance of artm.Regularizer
    """
    class_of_object = getattr(artm.regularizers, elemtype)
    kwargs = {name: value
              for name, value in elem_args.items()}
    # special case: shortcut for topic_names
    if "topic_names" in kwargs:
        if kwargs["topic_names"] == "background_topics":
            kwargs["topic_names"] = background_topic_names
        if kwargs["topic_names"] == "specific_topics":
            kwargs["topic_names"] = specific_topic_names

    return class_of_object(**kwargs)


def build_cube_settings(elemtype, elem_args):
    """
    Parameters
    ----------
    elemtype : str
        name of regularizer
    elem_args: strictyaml.YAML object
        (contains dict inside)

    Returns
    -------
    list of dict
    """
    if elemtype == "CubeCreator":
        preprocess_parameters_for_cube_creator(elem_args)

    kwargs = {name: value
              for name, value in elem_args.data.items()
              if name not in ['selection', 'strategy', 'strategy_params']}

    handle_special_cases(elem_args, kwargs)
    return {elemtype: kwargs,
            "selection": elem_args['selection'].data}


def _add_parsed_scores(parsed, topic_model):
    """ """
    for score in parsed.data.get('scores', []):
        for elemtype, elem_args in score.items():
            is_artm_score = elemtype in artm.scores.__all__
            score_object = build_score(elemtype, elem_args, is_artm_score)
            if is_artm_score:
                topic_model._model.scores.add(score_object, overwrite=True)
            else:
                topic_model.custom_scores[elemtype] = score_object


def _add_parsed_regularizers(
    parsed, model, specific_topic_names, background_topic_names, data_stats
):
    """ """
    regularizers = []
    for stage in parsed.data['regularizers']:
        for elemtype, elem_args in stage.items():
            should_be_relative = None
            if "relative" in elem_args:
                should_be_relative = elem_args["relative"]
                elem_args.pop("relative")

            regularizer_object = build_regularizer(
                elemtype, elem_args, specific_topic_names, background_topic_names
            )
            handle_regularizer(should_be_relative, model, regularizer_object, data_stats)
            regularizers.append(model.regularizers[regularizer_object.name])
    return regularizers


def parse_modalities_data(parsed):
    has_modalities_to_use = is_key_in_schema("modalities_to_use", parsed["model"])
    has_weights = is_key_in_schema("modalities_weights", parsed["model"])
    main_modality = parsed["model"]["main_modality"]

    # exactly one should be specified
    if has_modalities_to_use == has_weights:
        raise ValueError(f"Either 'modalities_to_use' or 'modalities_weights' should be specified")

    if has_weights:
        modalities_to_use = list(parsed["model"]["modalities_weights"].data)
        if main_modality not in modalities_to_use:
            modalities_to_use.append(main_modality)
        local_schema = Map({
            key: Float() for key in modalities_to_use
        })
        parsed["model"]["modalities_weights"].revalidate(local_schema)
        modalities_weights = parsed["model"]["modalities_weights"].data
    else:
        modalities_weights = None
        modalities_to_use = parsed.data["model"]["modalities_to_use"]
    return modalities_to_use, modalities_weights


def parse(yaml_string, force_single_thread=False):
    """
    Parameters
    ----------
    yaml_string : str
    force_single_thread : bool

    Returns
    -------
    cube_settings: list of dict
    regularizers: list
    topic_model: TopicModel
    dataset: Dataset
    """
    parsed = dirty_load(yaml_string, base_schema, allow_flow_style=True)

    specific_topic_names, background_topic_names = create_default_topics(
        parsed.data["topics"]["specific_topics"],
        parsed.data["topics"]["background_topics"]
    )

    revalidate_section(parsed, "stages")
    revalidate_section(parsed, "regularizers")
    if "scores" in parsed:
        revalidate_section(parsed, "scores")

    cube_settings = []

    dataset = Dataset(parsed.data["model"]["dataset_path"])

    modalities_to_use, modalities_weights = parse_modalities_data(parsed)

    data_stats = count_vocab_size(dataset.get_dictionary(), modalities_to_use)
    model = init_simple_default_model(
        dataset=dataset,
        modalities_to_use=modalities_to_use,
        main_modality=parsed.data["model"]["main_modality"],
        specific_topics=parsed.data["topics"]["specific_topics"],
        background_topics=parsed.data["topics"]["background_topics"],
        modalities_weights=modalities_weights
    )

    regularizers = _add_parsed_regularizers(
        parsed, model, specific_topic_names, background_topic_names, data_stats
    )
    topic_model = TopicModel(model)
    _add_parsed_scores(parsed, topic_model)

    for stage in parsed['stages']:
        for elemtype, elem_args in stage.items():
            settings = build_cube_settings(elemtype.data, elem_args)
            if force_single_thread:
                settings[elemtype]["separate_thread"] = False
            cube_settings.append(settings)

    return cube_settings, regularizers, topic_model, dataset


def revalidate_section(parsed, section):
    """
    Perofrms in-place type coercion and validation

    Parameters
    ----------
    parsed : strictyaml.YAML object
        (half-parsed, half-validated chunk of config)
    section: str
    """
    if section == "stages":
        schemas = build_schema_for_cubes()
    elif section == "regularizers":
        schemas = build_schema_for_regs()
    elif section == "scores":
        schemas = build_schema_for_scores()
    else:
        raise ValueError(f"Unknown section name '{section}'")

    for i, stage in enumerate(parsed[section]):
        assert len(stage) == 1
        name = list(stage.data)[0]

        if name not in schemas:
            raise ValueError(f"Unsupported {section} value: {name} at line {stage.start_line}")
        local_schema = schemas[name]

        stage.revalidate(local_schema)


def build_experiment_environment_from_yaml_config(yaml_string, experiment_id,
                                                  save_path, force_single_thread=False):
    settings, regs, model, dataset = parse(yaml_string, force_single_thread)
    # TODO: handle dynamic addition of regularizers
    experiment = Experiment(experiment_id=experiment_id, save_path=save_path, topic_model=model)
    experiment.build(settings)
    return experiment, dataset
