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
from .models.topic_model import TopicModel
from .cubes import RegularizersModifierCube, CubeCreator
from .experiment import Experiment
from .dataset import Dataset


from .cubes import PerplexityStrategy, GreedyStrategy
from .model_constructor import init_simple_default_model

import artm

from inspect import signature, Parameter
from strictyaml import Map, Str, Int, Seq, Any, Optional, Float, EmptyNone, Bool
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


element = Any()
base_schema = Map({
    'regularizers': Seq(element),
    'stages': Seq(element),
    'model': Map({
        "dataset_path": Str(),
        "modalities_to_use": Seq(Str()),
        "main_modality": Str()
    }),
    'topics': Map({
        "background_topics": Seq(Str()),
        "specific_topics": Seq(Str()),
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
            res = Map(build_schema_from_signature(class_of_object))

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


def build_regularizer(elemtype, elem_args, parsed):
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
            kwargs["topic_names"] = parsed.data["topics"]["background_topics"]
        if kwargs["topic_names"] == "specific_topics":
            kwargs["topic_names"] = parsed.data["topics"]["specific_topics"]

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


def parse(yaml_string):
    """
    Parameters
    ----------
    yaml_string : str

    Returns
    -------
    cube_settings: list of dict
    regularizers: list
    topic_model: TopicModel
    dataset: Dataset
    """
    parsed = dirty_load(yaml_string, base_schema, allow_flow_style=True)
    revalidate_section(parsed, "stages")
    revalidate_section(parsed, "regularizers")

    cube_settings = []
    regularizers = []

    dataset = Dataset(parsed.data["model"]["dataset_path"])
    model = init_simple_default_model(
        dataset=dataset,
        modalities_to_use=parsed.data["model"]["modalities_to_use"],
        main_modality=parsed.data["model"]["main_modality"],
        specific_topics=parsed.data["topics"]["specific_topics"],
        background_topics=parsed.data["topics"]["background_topics"],
    )
    for stage in parsed.data['regularizers']:
        for elemtype, elem_args in stage.items():

            regularizer_object = build_regularizer(elemtype, elem_args, parsed)

            regularizers.append(regularizer_object)
            model.regularizers.add(regularizer_object, overwrite=True)

    topic_model = TopicModel(model)
    for stage in parsed['stages']:
        for elemtype, elem_args in stage.items():
            settings = build_cube_settings(elemtype.data, elem_args)
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
    else:
        raise ValueError(f"Unknown section name '{section}'")

    for i, stage in enumerate(parsed[section]):
        assert len(stage) == 1
        name = list(stage.data)[0]

        if name not in schemas:
            raise ValueError(f"Unsupported stage ID: {name} at line {stage.start_line}")
        local_schema = schemas[name]

        stage.revalidate(local_schema)


def build_experiment_environment_from_yaml_config(yaml_string, experiment_id, save_path):
    settings, regs, model, dataset = parse(yaml_string)
    # TODO: handle dynamic addition of regularizers
    experiment = Experiment(experiment_id=experiment_id, save_path=save_path, topic_model=model)
    experiment.build(settings)
    return experiment, dataset
