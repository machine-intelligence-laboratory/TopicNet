import numpy as np
import hashlib
import json
import re
import warnings
from datetime import datetime
from statistics import mean, median
import numexpr as ne
# from .models.base_score import BaseScore


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
    from .models import TopicModel

    if model is None and experiment is not None:
        model = experiment.models.get(model_id)

    # hasattr(model, 'save') is not currently supported due to dummy save in BaseModel
    return isinstance(model, TopicModel)


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
    elif re.search(r'artm.score_tracker', str(type(obj))) is not None:
        return obj._name
    elif re.search(r'score', str(type(obj))) is not None:
        return str(obj.__class__)
    elif re.search(r'Score', str(type(obj))) is not None:
        return str(obj.__class__)
    elif re.search(r'Cube', str(type(obj))) is not None:
        return str(obj.__class__)
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


def choose_best_models(models: list, requirement_lesser: list, requirement_greater: list,
                       requirement_equal: list, metric: str, extremum="min", models_num=1):
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
        number of models to select (Default value = 1)

    Returns
    -------
    TopicModel
        models with best scores or matching request

    """
    def required_parameter(model, parameter):
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
        return model.scores[parameter][-1] \
            if parameter.split('.')[0] != 'model' \
            else model.get_init_parameters().get(parameter.split('.')[1])

    def is_acceptable(model):
        """
        Checks if model suits request.

        Parameters
        ----------
        model : TopicModel

        Returns
        -------
        bool

        """
        # locals: requirement_lesser/greater/equal
        answer = (
            all(required_parameter(model, req_parameter) < value
                for req_parameter, value in requirement_lesser)
            and
            all(required_parameter(model, req_parameter) > value
                for req_parameter, value in requirement_greater)
            and
            all(required_parameter(model, req_parameter) == value
                for req_parameter, value in requirement_equal)
        )
        return answer

    acceptable_models = [model for model in models if is_acceptable(model)]
    if not acceptable_models:
        all_requirements = [
            req_parameter for req_parameter, value
            in (requirement_lesser + requirement_greater + requirement_equal)
        ]
        raise ValueError(f'No model match criteria'
                         f'(The requirements on {", ".join(all_requirements)}'
                         f' have eliminated all {len(models)} models)')

    if metric is None and extremum is None:
        return acceptable_models
    elif metric not in models[0].scores:
        raise ValueError(f'There is no {metric} metric for model {models[0].model_id}.')

    scores = [acceptable_model.scores[metric][-1] for acceptable_model in acceptable_models]
    if extremum == "max":
        best_models = np.array(acceptable_models)[np.argsort(-np.array(scores))[:models_num]]
    else:
        best_models = np.array(acceptable_models)[np.argsort(scores)[:models_num]]

    return best_models.tolist()


def parse_query_string(query_string: str):
    """
    This function will parse query string and subdivide it into following parts:

    Parameters
    ----------
    query_string : str
        string of following form:  
        QUERY = EXPR and EXPR and EXPR and ... and EXPR
        where EXPR could take any of these forms:  
        EXPR = SCORE_NAME/model.PARAMETER_NAME < NUMBER  
        EXPR = SCORE_NAME/model.PARAMETER_NAME > NUMBER  
        EXPR = SCORE_NAME/model.PARAMETER_NAME = NUMBER  
        EXPR = SCORE_NAME -> min  
        EXPR = SCORE_NAME -> max  
        Everything is separated by spaces.

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
    for part in query_string.split("and"):
        expression_parts = part.strip().split(" ")
        if len(expression_parts) != 3:
            raise ValueError(f"Cannot understand '{part}'")

        first, middle, last = expression_parts
        if middle in [">", "<", "="]:
            requirement[middle] += [(first, float(last))]
        elif middle == "->":
            if metric is not None:
                raise ValueError(f"Cannot process more than one target ({metric} and {first})")
            metric = first
            if last in ["max", "min"]:
                extremum = last
            else:
                raise ValueError(f"Cannot understand '{part}': unknown requirement '{last}'")
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

    processed_queries = []
    for query in special_queries:
        first, middle, *raw_last = query.strip().split(' ')
        if middle not in ['>', '<', '=']:
            raise ValueError(f"Cannot understand '{query}': unknown format")

        last = []
        for subpart in raw_last:
            if subpart[0] in ['A', 'M']:
                split_subpart = re.split('[()]', subpart)
                special_function, metric = split_subpart[0], split_subpart[1]
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
