import numpy as np

from datetime import datetime
from .routine import get_fix_string, get_fix_list
from .routine import get_equal_strings, get_equal_lists
from .models.base_model import MODEL_NAME_LENGTH

UP_END = "┌"
DOWN_END = "└"
MIDDLE = "├"
LAST = "┤"
EMPTY = "│"
START_END = "┐"
SPACE = " "

USELESS_SUBKEYS = {
    "type", "verbose", "config"
}
USELESS_KEYS = {
    "reuse_theta", "cache_theta", "num_document_passes", "theta_name",
    "parent_model_weight", "theta_columns_naming", "transaction_typenames",
    "score_tracker", "model_pwt", "model_nwt",
    "num_phi_updates", "num_online_processed_batches", "show_progress_bars",
    "library_version", "version", "regularizers",
}


def resize_value(key, value, tab: str = "  "):
    """

    Parameters
    ----------
    key : str
    value : optional
    tab: str
         (Default value = "  ")

    Returns
    -------
    list

    """
    if key in ["scores", "taus", "criteria"]:
        return [(tab + elem) for elem
                in get_fix_list(value, length=-1, num=-1)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [(tab + elem) for elem
                in get_fix_list(value, length=-1, num=5)]
    if isinstance(value, dict):
        def _trim(some_value):
            if isinstance(some_value, list):
                if len(some_value) > 4:
                    some_value = some_value[:2] + ["..."] + some_value[-2:]
                if len(some_value) > 0:
                    all_strings = all([isinstance(elem, str) for elem in some_value])
                    if all_strings:
                        some_value = "[" + ", ".join(some_value) + "]"
            return some_value
        pairs = [
            f"{key}={_trim(some_value)}"
            for key, some_value in value.items()
            if key not in USELESS_SUBKEYS and some_value is not None
        ]

        return [(tab + elem) for elem in get_fix_list(pairs, length=-1, num=15)]
    return [tab + get_fix_string(value, length=-1)]


def get_cube_strings(cubes, tab: str = "  ", min_len_per_cube: int = MODEL_NAME_LENGTH):
    """

    Parameters
    ----------
    cubes : list of dict
    tab : str
         (Default value = "  ")
    min_len_per_cube : int
         (Default value = MODEL_NAME_LENGTH defined in BaseModel)

    Returns
    -------
    dict

    """
    cube_strings = dict()
    for id_cube, cube in enumerate(cubes):
        cube_strings[id_cube] = []
        cube_strings[id_cube].append(f"{cube['action'].upper()}")
        for key, value in cube.items():
            if key not in ["action", "params"]:
                cube_strings[id_cube] += [get_fix_string(key, length=-1) + ":",
                                          tab + get_fix_string(value, length=-1)]
        cube_strings[id_cube].append("")
        for params in cube["params"]:
            for key, value in params.items():
                if (key[0] != "_") and (key[-1] != "_"):
                    if key not in USELESS_KEYS:
                        cube_strings[id_cube].append(get_fix_string(key, length=-1) + ":")
                        cube_strings[id_cube] += resize_value(key, value, tab)
            cube_strings[id_cube].append("   +   ")
        cube_strings[id_cube][-1] = ""
        get_equal_strings(cube_strings[id_cube], min_len=min_len_per_cube)
    get_equal_lists(cube_strings)
    return cube_strings


def get_criteria_strings(criteria, tab: str = "  ", min_len_per_cube: int = MODEL_NAME_LENGTH):
    """

    Parameters
    ----------
    criteria : list of str
    tab : str
         (Default value = "  ")
    min_len_per_cube : int
         (Default value = MODEL_NAME_LENGTH defined in BaseModel)

    Returns
    -------
    dict

    """
    criterion_strings = dict()
    for id_criterion, criterion in enumerate(criteria):
        criterion_strings[id_criterion] = []
        if criterion is None:
            criterion_strings[id_criterion].append(' ')
        else:
            for statement_id, statement in enumerate(criterion, 1):
                stage = statement.split(' and ')
                if len(stage) > 1:
                    heading = f'stage criteria {statement_id}:'
                else:
                    heading = f'stage criterion {statement_id}:'
                criterion_strings[id_criterion].append(heading)
                for rule in stage:
                    criterion_strings[id_criterion].append(tab * 2 + rule)
                criterion_strings[id_criterion].append("")
        get_equal_strings(criterion_strings[id_criterion], min_len=min_len_per_cube)
    get_equal_lists(criterion_strings)
    return criterion_strings


def add_non_tree_strings(strings, strings_to_add, add_separator=True):
    """
    Adding training stage strings
    to the experiment description

    Parameters
    ----------
    strings : list of strings
        description of the experiment
    strings_to_add : dict of lists of strings
        new information to add to the experiment
    add_separator : bool
        make pretty separation line
        at the end of this strings (Default value = True)

    Returns
    -------
    strings : list of strings
        description of the experiment
    """
    separation_string = ''
    for id_string in range(len(strings_to_add[0])):
        string = " "
        for id_stage, value in strings_to_add.items():
            string += value[id_string] + " | "
            if len(separation_string) < len(string):
                separation_string += "─" * (len(value[id_string]) + 2) + '+'
        string = string[:-3]
        strings.append(string)
    if add_separator:
        strings.append(separation_string[:-3])
    return strings


def give_strings_description(experiment,
                             tab: str = "  ",
                             min_len_per_cube: int = MODEL_NAME_LENGTH,
                             len_tree_step: int = MODEL_NAME_LENGTH + 1):
    """
    Gets strings description of the experiment.

    Parameters
    ----------
    tab : str
        tab symbol for margin (Default value = "  ")
    min_len_per_cube : int
        minimal length of one stage of description experiment
        (Default value = MODEL_NAME_LENGTH defined in BaseModel)
    len_tree_step : int
        length of the whole one stage description of experiment's tree
        (Default value = MODEL_NAME_LENGTH + 1 defined in BaseModel)

    Returns
    -------
    list
        strings description

    """
    version = 'not defined'
    for ind in range(len(experiment.cubes)):
        if experiment.cubes[ind]['params'][0].get('version', 'not defined') != 'not defined':
            version = experiment.cubes[ind]['params'][0]['version']

    strings = [f"Experiment {experiment.experiment_id}", "",
               f"Experiment was made with BigARTM {version}"]

    cube_strings = get_cube_strings(experiment.cubes, tab, min_len_per_cube)
    stage_strings = get_criteria_strings(experiment.criteria, tab, min_len_per_cube)
    tree_strings = experiment.tree.get_description()
    for key in range(len(cube_strings)):
        max_len_string = max(len(cube_strings[key][0]), len(stage_strings[key][0]))
        cube_strings[key] = [
            get_fix_string(string_cube, length=max_len_string)
            for string_cube in cube_strings[key]
        ]
        stage_strings[key] = [
            get_fix_string(string_stage, length=max_len_string)
            for string_stage in stage_strings[key]
        ]
    # merge strings together
    # st = test_len_tree_step - 1
    fi = -1
    st = len_tree_step - 1
    for id_cube, values in cube_strings.items():
        fi += len(values[-1]) + 3
        for id_string in range(len(tree_strings)):
            cur_string = tree_strings[id_string][:]
            if st < len(cur_string):
                if cur_string[st] == LAST or cur_string[st] == START_END:
                    tree_strings[id_string] = cur_string[:st] + "─" * (fi - st) \
                                              + cur_string[st:]
                else:
                    tree_strings[id_string] = cur_string[:st] + " " * (fi - st) \
                                              + cur_string[st:]
        st += fi - st + len_tree_step

    strings.append("Tree:")
    strings += tree_strings
    strings.append("Cubes:")
    strings = add_non_tree_strings(strings, cube_strings, add_separator=True)
    strings = add_non_tree_strings(strings, stage_strings, add_separator=False)
    return strings


def get_html(experiment, window_size: int = 1500):
    """
    Gets html text to save human-readable description of the experiment.

    Parameters
    ----------
    window_size : int
        pixels size of window in html description (Default value = 1500)

    Returns
    -------
    str
        description of the experiment in html format

    """
    # TODO разобраться с разной шириной пробела в шрифтах
    strings = give_strings_description(experiment)
    strings_html = ["<html>",
                    f"<p><font size='+5'>Experiment <b>{experiment.experiment_id}</b></font></p>",
                    f"<p><i>{strings[2]}</i></p>",
                    "<p></p>",
                    f"<td width=\"{window_size}px\" style=\"white-space:pre;\">",
                    f"<div style=\"width:{window_size}px;overflow:auto;white-space:pre;\">"]
    for string in strings[3:]:
        if string == "":
            strings_html.append("<p></p>")
        elif string in ["Tree:", "Cubes:"]:
            strings_html.append(f"<p><font size='+1'><b>{string}</b></font></p>")
        else:
            strings_html += ["<samp><font size='+1'>" + "&ensp;".join(string.split(" "))
                             + "</font></samp>"]
    strings_html += ["</div>", "</td>", "<p></p>",
                     "<p><i><font size='-1'>Page was generated at "
                     + str(datetime.now()) + ".</font></i></p>",
                     "</html>"]
    return "\n".join(strings_html)


def make_notebook_pretty():
    from IPython.core.display import display, HTML

    display(HTML("""<style>
    div.output_html {
        white-space: nowrap;
    }
    div .output_subarea > pre {
        white-space: pre;
        word-wrap: normal;
    }
    div .output_stdout > pre {
        white-space: pre-wrap !important;
        word-wrap:  break-word !important;
    }
    </style>"""))
