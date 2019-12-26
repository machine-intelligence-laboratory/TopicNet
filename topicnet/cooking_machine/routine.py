import numpy as np
import hashlib
import json
import re
import warnings
from datetime import datetime
from statistics import mean, median
import numexpr as ne


W_TOO_STRICT = 'No models match criteria '
W_TOO_STRICT_DETAILS = '(The requirements on {} have eliminated all {} models)'
W_NOT_ENOUGH_MODELS_FOR_CHOICE = 'Not enough models'
W_NOT_ENOUGH_MODELS_FOR_CHOICE_DETAILS = 'for models_num = {}, only {} models will be returned.'
W_RETURN_FEWER_MODELS = 'Can\'t return the requested number of models:'
W_RETURN_FEWER_MODELS_DETAILS = ' \"{}\". Only \"{}\" satisfy the query'


def is_jsonable(x):
    """
    Check that x is jsonable

    Parameters
    ----------
    x : optional

    Returns
    -------
    bool

    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def is_saveable_model(model=None, model_id=None, experiment=None):
    """
    Little helpful function. May be extended later.

    """
    from .models import SUPPORTED_MODEL_CLASSES

    if model is None and experiment is not None:
        model = experiment.models.get(model_id)

    # hasattr(model, 'save') is not currently supported due to dummy save in BaseModel

    return isinstance(model, SUPPORTED_MODEL_CLASSES)


def get_public_instance_attributes(instance):
    """
    Get list of all instance public atrributes.

    Parameters
    ----------
    instance : optional

    Returns
    -------
    list of str

    """
    public_attributes = [
        attribute
        for attribute in instance.__dir__() if attribute[0] != '_'
    ]
    return public_attributes


def transform_complex_entity_to_dict(some_entity):
    """

    Parameters
    ----------
    some_entity : optional

    Returns
    -------
    dict
        jsonable entity

    """
    jsonable_reg_params = dict()

    jsonable_reg_params['name'] = some_entity.__class__.__name__
    public_attributes = get_public_instance_attributes(some_entity)
    for attribute in public_attributes:
        try:
            value = getattr(some_entity, attribute)
            if is_jsonable(value):
                jsonable_reg_params[attribute] = value
        except (AttributeError, KeyError):
            # TODO: need warning here
            jsonable_reg_params[attribute] = None

    return jsonable_reg_params


def get_timestamp_in_str_format():
    """
    Returns current timestamp.

    Returns
    -------
    str
        timestamp in "%Hh%Mm%Ss_%dd%mm%Yy" format

    """
    curr_tmsp = datetime.now().strftime("%Hh%Mm%Ss_%dd%mm%Yy")

    return curr_tmsp


def transform_topic_model_description_to_jsonable(obj):
    """
    Change object to handle serialization problems with json.

    Parameters
    ----------
    obj : object
        input object

    Returns
    -------
    int
        jsonable object

    """
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif re.search(r'artm.score_tracker', str(type(obj))) is not None:
        return obj._name
    elif re.search(r'score', str(type(obj))) is not None:
        return str(obj.__class__)
    elif re.search(r'Score', str(type(obj))) is not None:
        return str(obj.__class__)
    elif re.search(r'Cube', str(type(obj))) is not None:
        return str(obj.__class__)
    elif re.search(r'protobuf', str(type(obj))) is not None:
        try:
            return str(list(obj))
        except:  # noqa: E722
            return str(type(obj))
    else:
        warnings.warn(f'Object {obj} can not be dumped using json.' +
                      'Object class name will be returned.', RuntimeWarning)
        return str(obj.__class__)


def get_fix_string(input_string: str, length: int):
    """
    Transforms input_string to the string of the size length.

    Parameters
    ----------
    input_string : str
        input_string
    length : int
        length of output_string, if -1 then output_string is the same as input_string

    Returns
    -------
    str
        beautiful string of the size length

    """
    input_string = str(input_string)
    if length < 0:
        output_string = input_string
    elif len(input_string) > length:
        sep = (length - 3) // 2
        if length % 2 == 0:
            output_string = input_string[:sep + 1] + "..." + input_string[-sep:]
        else:
            output_string = input_string[:sep] + "..." + input_string[-sep:]
    else:
        output_string = input_string + " " * (length - len(input_string))

    return output_string


def get_fix_list(input_list: list, length: int, num: int):
    """
    Returns list with strings of size length that contains not more than num strings.

    Parameters
    ----------
    input_list : list
        list of input strings
    length : int
        length of output strings
    num : int
        maximal number of strings on output list

    Returns
    -------
    list
        list with no more than num of beautiful strings

    """
    if len(input_list) == 0:
        input_list = ["---"]
    output_list = []
    if (len(input_list) > num) and (num != -1):
        sep = (num - 1) // 2
        if num % 2 == 0:
            for elem in input_list[:sep + 1]:
                output_list.append(get_fix_string(elem, length - 1) + ",")
            output_list.append("...," + " " * (length - 4))
            for elem in input_list[-sep:]:
                output_list.append(get_fix_string(elem, length - 1) + ",")
            output_list[-1] = output_list[-1][:-1] + " "
        else:
            for elem in input_list[:sep]:
                output_list.append(get_fix_string(elem, length - 1) + ",")
            output_list.append("...," + " " * (length - 4))
            for elem in input_list[-sep:]:
                output_list.append(get_fix_string(elem, length - 1) + ",")
            output_list[-1] = output_list[-1][:-1] + " "
    else:
        for elem in input_list:
            output_list.append(get_fix_string(elem, length - 1) + ",")
        output_list[-1] = output_list[-1][:-1] + " "

    return output_list


def get_equal_strings(strings, min_len: int = 0, sep: str = " "):
    """
    Transforms all strings to strings with the same length, but not less that min_len.
    Fills strings with sep. Inplace.

    Parameters
    ----------
    strings : list
        list of strings
    min_len : int
        minimal length of the string (Default value = 0)
    sep : str
        filling symbol (Default value = " ")

    """
    max_string_len = np.array([len(string) for string in strings]).max()
    max_string_len = max(min_len, max_string_len)
    for id_string, string in enumerate(strings):
        if len(string) < max_string_len:
            strings[id_string] += sep * (max_string_len - len(string))


def get_equal_lists(one_dict, min_len: int = 0, sep: str = " ", sep_len="last"):
    """
    Transforms all lists to list with the same length, but not less that min_len.
    Fills lists with sep. Inplace.

    Parameters
    ----------
    one_dict : dict
        dict with lists
    min_len : int
        minimal length of the list (Default value = 0)
    sep : str
        filling symbol (Default value = " ")
    sep_len : int or "last"
        length of added strings, if "last" than length of added strings is equal
        to the length of the last string in the list (Default value = "last")

    """
    max_len = np.array([len(one_list) for one_list in one_dict.values()]).max()
    max_len = max(min_len, max_len)
    for id_list, one_list in one_dict.items():
        if sep_len == "last":
            one_dict[id_list] += [sep * len(one_list[-1])] * (max_len - len(one_list))
        elif isinstance(sep_len, int):
            one_dict[id_list] += [sep * sep_len] * (max_len - len(one_list))
        else:
            raise ValueError("Parameter sep_len can be int or \"last\".")


def extract_required_parameter(model, parameter):
    """
    Extracts necessary parameter from model.

    Parameters
    ----------
    model : TopicModel
    parameter : str

    Returns
    -------
    optional

    """
    value_to_return_as_none = float('nan')  # value needed for comparisons in is_acceptable

    if parameter.split('.')[0] == 'model':
        parameters = model.get_init_parameters()
        parameter_name = parameter.split('.')[1]

        if parameter_name in parameters.keys():
            parameter_value = parameters.get(parameter_name)

            if parameter_value is not None:
                return parameter_value
            else:
                return value_to_return_as_none
        else:
            raise ValueError(f'Unknown parameter {parameter_name} for model.')
    else:
        scores = model.scores.get(parameter, None)

        if scores is None and model.depth == 0:  # start model
            warnings.warn(f'Start model doesn\'t have score values for \"{parameter}\"')

            return value_to_return_as_none

        elif scores is None:
            raise ValueError(
                f'Model \"{model}\" doesn\'t have the score \"{parameter}\". '
                f'Expected score name {parameter} or model.parameter {parameter}')

        if len(scores) == 0:
            raise ValueError(f'Empty score {parameter}.')

        if scores[-1] is None:  # FrozenScore
            return value_to_return_as_none

        return scores[-1]


def is_acceptable(model, requirement_lesser, requirement_greater, requirement_equal):
    """
    Checks if model suits request.

    Parameters
    ----------
    model : TopicModel
    requirement_lesser : list of tuple
    requirement_greater : list of tuple
    requirement_equal : list of tuple

    Returns
    -------
    bool

    """
    from .models import TopicModel

    if not isinstance(model, TopicModel):
        warnings.warn(f'Model {model} isn\'t of type TopicModel.' +
                      ' Check your selection level and/or level models.')
        return False

    answer = (
        all(extract_required_parameter(model, req_parameter) < value
            for req_parameter, value in requirement_lesser)
        and
        all(extract_required_parameter(model, req_parameter) > value
            for req_parameter, value in requirement_greater)
        and
        all(extract_required_parameter(model, req_parameter) == value
            for req_parameter, value in requirement_equal)
    )
    return answer


def _select_acceptable_models(models,
                              requirement_lesser, requirement_greater, requirement_equal):
    """
    Selects necessary models with sanity check.

    Parameters
    ----------
    models : list of TopicModel
        list of models with .scores parameter.
    requirement_lesser : list of tuple
        list containing tuples of form
        (SCORE_NAME/model.PARAMETER_NAME, TARGET_NUMBER)
    requirement_greater : list of tuple
        list containing tuples of form
        (SCORE_NAME/model.PARAMETER_NAME, TARGET_NUMBER)
    requirement_equal : list of tuple
        list containing tuples of form
        (SCORE_NAME/model.PARAMETER_NAME, TARGET_NUMBER)

    Returns
    -------
    list of TopicModels
    """
    acceptable_models = [
        model for model in models if is_acceptable(
            model,
            requirement_lesser,
            requirement_greater,
            requirement_equal
        )
    ]
    if len(models) and not len(acceptable_models):
        all_requirements = [
            req_parameter for req_parameter, value
            in (requirement_lesser + requirement_greater + requirement_equal)
        ]
        warnings.warn(W_TOO_STRICT +
                      W_TOO_STRICT_DETAILS.format(", ".join(all_requirements), len(models)))

    return acceptable_models


def choose_value_for_models_num_and_check(
        models_num_as_parameter, models_num_from_query) -> int:

    models_num = None

    if models_num_as_parameter is not None and models_num_from_query is not None and \
            models_num_as_parameter != models_num_from_query:

        warnings.warn(
            f'Models number given as parameter \"{models_num_as_parameter}\" '
            f'not the same as models number specified after '
            f'COLLECT: \"{models_num_from_query}\". '
            f'Parameter value \"{models_num_as_parameter}\" will be used for select'
        )

        models_num = models_num_as_parameter

    elif models_num_as_parameter is not None:
        models_num = models_num_as_parameter

    elif models_num_from_query is not None:
        models_num = models_num_from_query

    if models_num is not None and int(models_num) < 0:
        raise ValueError(f"Cannot return negative number of models")

    return models_num


def _choose_models_by_metric(acceptable_models, metric, extremum, models_num):
    scores_models = {}

    for acceptable_model in acceptable_models:
        if len(acceptable_model.scores[metric]) == 0:
            warnings.warn(
                f'Model \"{acceptable_model}\" has empty value list for score \"{metric}\"')

            continue

        score = acceptable_model.scores[metric][-1]

        if score in scores_models.keys():
            scores_models[score].append(acceptable_model)
        else:
            scores_models[score] = [acceptable_model]

    scores_models = sorted(scores_models.items(), key=lambda kv: kv[0])

    if models_num is None:
        models_num = len(scores_models) if not metric else 1

    if extremum == "max":
        scores_models = list(reversed(scores_models))

    best_models = sum([models[1] for models in scores_models[:models_num]], [])
    result_models = best_models[:models_num]

    if models_num > len(acceptable_models):
        warnings.warn(
            W_NOT_ENOUGH_MODELS_FOR_CHOICE + ' ' +
            W_NOT_ENOUGH_MODELS_FOR_CHOICE_DETAILS.format(models_num, len(acceptable_models))
        )

    if len(result_models) < models_num:
        warnings.warn(W_RETURN_FEWER_MODELS.format(models_num, len(result_models)))

    return result_models


def choose_best_models(models: list, requirement_lesser: list, requirement_greater: list,
                       requirement_equal: list, metric: str, extremum="min", models_num=None):
    """
    Get best model according to specified metric.

    Parameters
    ----------
    models : list of TopicModel
        list of models with .scores parameter.
    requirement_lesser : list of tuple
        list containing tuples of form
        (SCORE_NAME/model.PARAMETER_NAME, TARGET_NUMBER)
    requirement_greater : list of tuple
        list containing tuples of form
        (SCORE_NAME/model.PARAMETER_NAME, TARGET_NUMBER)
    requirement_equal : list of tuple
        list containing tuples of form
        (SCORE_NAME/model.PARAMETER_NAME, TARGET_NUMBER)
    metric : str
        metric for selection.
    extremum : str
        "min" or "max" - comparison parameter (Default value = "min")
    models_num : int
        number of models to select
        (default value is None, which is mapped to "all" or 1 depending on whether 'metric' is set)

    Returns
    -------
    best_models : list of models
        models with best scores or matching request

    """
    acceptable_models = _select_acceptable_models(
        models,
        requirement_lesser,
        requirement_greater,
        requirement_equal
    )

    if metric is None and extremum is None:
        if models_num is None:
            result_models = acceptable_models
        else:
            result_models = acceptable_models[:models_num]

        if models_num is not None and len(result_models) < models_num:
            warnings.warn(W_RETURN_FEWER_MODELS + ' ' +
                          W_RETURN_FEWER_MODELS_DETAILS.format(models_num, len(result_models)))

        return result_models

    elif len(models) > 0 and metric not in models[0].scores:
        raise ValueError(f'There is no {metric} metric for model {models[0].model_id}.\n'
                         f'The following scores are available: {list(models[0].scores.keys())}')

    return _choose_models_by_metric(acceptable_models, metric, extremum, models_num)


def parse_query_string(query_string: str):
    """
    This function will parse query string and subdivide it into following parts:

    Parameters
    ----------
    query_string : str
        (see Experiment.select function for details)

    Returns
    -------
    requirement_lesser : list
    requirement_greater : list
    requirement_equal : list
    metric : str
    extremum : str

    """  # noqa: W291
    requirement = {
        ">": [],
        "<": [],
        "=": []
    }
    metric = None
    extremum = None

    for part in filter(None, re.split(r'\s+and\s+', query_string)):
        expression_parts = part.strip().split()

        if len(expression_parts) != 3:
            raise ValueError(f"Cannot understand '{part}'")

        first, middle, last = expression_parts

        if middle in [">", "<", "="]:
            requirement[middle] += [(first, float(last))]

        elif middle == "->":
            current_metric = first
            current_extremum = last

            if metric == current_metric and extremum == current_extremum:
                continue

            if metric is not None:
                raise ValueError(
                    f"Cannot process more than one target: "
                    f"previous \"{metric}\" with extremum \"{extremum}\" and "
                    f"current \"{current_metric}\" with extremum \"{current_extremum}\"")

            if current_extremum not in ["max", "min"]:
                raise ValueError(f"Cannot understand '{part}': "
                                 f"unknown requirement '{current_extremum}'")

            metric = current_metric
            extremum = current_extremum

        else:
            raise ValueError(f"Unknown connector '{middle}' in '{part}'")

    return requirement["<"], requirement[">"], requirement["="], metric, extremum


def compute_special_queries(special_models, special_queries):
    """
    Computes special queries with functions.

    """
    special_functions = {
        'MINIMUM': min,
        'MAXIMUM': max,
        'AVERAGE': mean,
        'MEDIAN': median,
    }
    if not special_models and special_queries:
        warnings.warn(f"Cannot evaluate '{special_queries}': list of candidate models is empty",
                      RuntimeWarning)

    processed_queries = []
    for query in special_queries:
        first, middle, *raw_last = query.strip().split()
        if middle not in ['>', '<', '=']:
            raise ValueError(f"Cannot understand '{query}': unknown format")

        last = []
        for subpart in raw_last:
            if subpart[0] in ['A', 'M']:
                split_subpart = re.split('[()]', subpart)
                special_function, metric = split_subpart[0].strip(), split_subpart[1].strip()
                scores = [model.scores[metric][-1] for model in special_models]
                last.append(str(special_functions.get(special_function, max)(scores)))
            else:
                last += subpart

        try:
            last = str(ne.evaluate(''.join(last)))
        except SyntaxError:
            raise ValueError(f"Cannot evaluate {last} expression")

        processed_queries.append(' '.join([first, middle, last]))

    return processed_queries


def blake2bchecksum(file_path):
    """
    Calculates hash of the file

    Parameters
    ----------
    file_path : str
        path to the file
    """
    with open(file_path, 'rb') as fh:
        m = hashlib.blake2b()
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
        return m.hexdigest()
