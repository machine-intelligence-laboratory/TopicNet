import os
import re
import json
import warnings

from .model_tracking import Tree, START
from typing import List

from .pretty_output import give_strings_description, get_html
from .routine import transform_topic_model_description_to_jsonable
from .routine import (
    parse_query_string,
    choose_best_models,
    compute_special_queries,
    choose_value_for_models_num_and_check
)
from .routine import is_saveable_model

from .models import BaseModel
from .models.base_model import MODEL_NAME_LENGTH

W_EMPTY_SPECIAL_1 = 'Unable to calculate special functions in query\n'
W_EMPTY_SPECIAL_2 = 'Process failed with following: {}'
EMPTY_ERRORS = [
    'mean requires at least one data point',
    'no median for empty data',
    'min() arg is an empty sequence',
    'max() arg is an empty sequence',
]


def _run_from_notebook():
    try:
        shell = get_ipython().__class__.__name__  # noqa: F821
        return shell == 'ZMQInteractiveShell'
    except:  # noqa: E722
        return False


class Experiment(object):
    """
    Contains experiment, its description and descriptions of all models in the experiment.

    """
    def __init__(self, topic_model, experiment_id: str, save_path: str,
                 save_model_history: bool = False, save_experiment: bool = True,
                 tree: dict = None, models_info: dict = None, cubes: List[dict] = None,
                 low_memory: bool = False):
        """
        Initialize stage, also used for loading and creating new experiments.

        Parameters
        ----------
        experiment_id : str
            experiment id
        save_path : str
            path to save the experiment
        topic_model : TopicModel or None
            if TopicModel - use initial topic_model or last topic_model
            if save_model_history is True 
            if None - create empty experiment
        save_model_history : bool
            if True - Experiment will save all information about previous
            models (before this topic_model). The typical use case than
            you want to apply cube that cannot be applied in old
            experiment, then you create new experiment that will save
            all necessary information and will be independent itself  
            if False - then topic model will be initial model (the first)
        tree : dict
            tree of the experiment. It is used for loading and creating non empty experiment
        models_info : dict
            keys are model ids, where values are model's description
        cubes : list of dict
            cubes that were used in the experiment
        low_memory : bool
            If true, models be transformed to dummies via `squeeze_models()`.
            Gradually, level by level.
            If false, models will be untouched, all data, including inner ARTM models,
            Phi, Theta matrices, stays.

        """  # noqa: W291

        if not isinstance(save_path, str):
            raise ValueError("Cannot create an Experiment with invalid save_path!")
        if not isinstance(experiment_id, str):
            raise ValueError("Cannot create an Experiment with invalid experiment_id!")

        self.experiment_id = experiment_id

        if os.path.exists(save_path) and save_experiment:
            folders = os.listdir(save_path)
            if experiment_id in folders:
                raise FileExistsError(
                    f"In /{save_path} experiment {experiment_id} already exists"
                )

        self.save_path = save_path

        # if you want to create an empty Experiment (only experiment_id and save_path must be known)
        if save_model_history:
            self._prune_experiment(topic_model)
        else:
            topic_model.model_id = START
            self.cubes = [
                {
                    'action': 'start',
                    'params': [topic_model.get_jsonable_from_parameters()],
                }
            ]
            self.criteria = [None]
            self.models_info = {
                START: topic_model.get_jsonable_from_parameters()
            }

            self.models = {
                START: topic_model,
            }
            topic_model.experiment = self
            self.tree = Tree()
            self.tree.add_model(topic_model)
            topic_model.save_parameters()

        if save_experiment:
            self.save()

        self.datasets = dict()

        self._low_memory = low_memory

    @property
    def depth(self):
        """
        Returns depth of the tree.  
        Be careful, depth of the tree may not be the real experiment depth.

        """  # noqa: W291
        return self.tree.get_depth()

    @property
    def root(self):
        """ """
        return self.models[START]

    def _move_models(self, load_path, old_experiment_id):
        """
        Moves models description to a new experiment.

        Parameters
        ----------
        load_path : str
            path to an old experiment
        old_experiment_id : str
            old experiment id

        """
        path_from = f"{load_path}/{old_experiment_id}"
        path_to = f"{self.save_path}/{self.experiment_id}"
        if not os.path.exists(path_to):
            os.makedirs(path_to)
        for model_id in self.models_info:
            os_code = os.system(f"cp -R {path_from}/{model_id} {path_to}/{model_id}")
            if os_code == 0:
                params = json.load(open(f"{path_to}/{model_id}/params.json", "r"))
                params["experiment_id"] = self.experiment_id
                json.dump(params, open(f"{path_to}/{model_id}/params.json", "w"))

    def _prune_experiment(self, topic_model):
        """
        Prunes old experiment. Creates new experiment with information from old experiment.

        Parameters
        ----------
        topic_model : TopicModel
            topic_model

        """
        experiment = topic_model.experiment
        self.cubes = experiment.cubes[:topic_model.depth + 1]
        self.criteria = experiment.criteria[:topic_model.depth + 1]
        self.tree = experiment.tree.clone()
        self.tree.prune(topic_model.depth)
        self.models_info = dict()
        self.models = dict()
        for model_id in self.tree.get_model_ids():
            self.models_info[model_id] = experiment.models_info[model_id]
            self.models[model_id] = experiment.models[model_id]
        self._move_models(topic_model.experiment.save_path,
                          topic_model.experiment.experiment_id)
        topic_model.experiment = self

    def _recover_consistency(self, load_path):
        """
        Recovers removed files and models descriptions.

        Parameters
        ----------
        load_path : str
            path to the experiment

        """
        if load_path[-1] == "/":
            load_path = load_path[:-1]
        if self.save_path != "/".join(load_path.split("/")[:-1]):
            print(f"This Experiment was replaced from {self.save_path}.", end=" ")
            self.save_path = "/".join(load_path.split("/")[:-1])
            print("Parameter is updated.")
        if self.experiment_id != load_path.split("/")[-1]:
            print(f"This Experiment was renamed to {load_path.split('/')[-1]}.", end=" ")
            self.experiment_id = load_path.split("/")[-1]
            for model_id in self.models_info.keys():
                self.models_info[model_id]["experiment_id"] = self.experiment_id
                model_save_path = f"{self.save_path}/{self.experiment_id}/{model_id}"
                if os.path.exists(model_save_path) \
                        and ("params.json" in os.listdir(model_save_path)):
                    params = self.models_info[model_id]
                    json.dump(params, open(f"{model_save_path}/params.json", "w"),
                              default=transform_topic_model_description_to_jsonable)
            print("Parameter is updated.")

        experiment_save_path = f"{self.save_path}/{self.experiment_id}"
        files = os.listdir(experiment_save_path)
        if "params.html" not in files:
            print("The file params.html was removed. Recover...", end=" ")
            html = get_html(self,)
            with open(f"{experiment_save_path}/params.html", "w", encoding='utf-8') as f:
                f.write(html)
            print("Recovered.")
        for model_id in self.models_info:
            model_save_path = f"{experiment_save_path}/{model_id}"
            if model_id not in files:
                print(f"The folder with {model_id} model was removed. "
                      f"Recover...",
                      end=" ")
                os.makedirs(model_save_path)
                params = self.models_info[model_id]
                json.dump(params, open(f"{model_save_path}/params.json", "w"),
                          default=transform_topic_model_description_to_jsonable)
                print("Recovered.")
            else:
                model_files = os.listdir(model_save_path)
                if "params.json" not in model_files:
                    print(f"The file params.json in {model_id} folder was removed. "
                          f"Recover...",
                          end=" ")
                    params = self.models_info[model_id]
                    json.dump(params, open(f"{model_save_path}/params.json", "w"),
                              default=transform_topic_model_description_to_jsonable)
                    print("Recovered.")

    def get_params(self):
        """
        Gets params of the experiment.

        Returns
        -------
        parameters : dict

        """
        params = {"save_path": self.save_path,
                  "experiment_id": self.experiment_id,
                  "models_info": self.models_info,
                  "criteria": self.criteria,
                  "tree": self.tree.tree,
                  "depth": self.depth,
                  "cubes": self.cubes}

        return params

    def add_model(self, topic_model):
        """
        Adds model to the experiment.

        Parameters
        ----------
        topic_model : TopicModel
            topic model

        """
        topic_model.experiment = self
        self.tree.add_model(topic_model)
        self.models_info[topic_model.model_id] = topic_model.get_parameters()
        self.models[topic_model.model_id] = topic_model
        self.save()

    def add_cube(self, cube):
        """
        Adds cube to the experiment.

        Parameters
        ----------
        cube : dict
            cube's params

        """
        self.cubes.append(cube)
        self.criteria.append(None)
        self.save()

    def add_dataset(self, dataset_id, dataset):
        """
        Adds dataset to storage.

        Parameters
        ----------
        dataset_id : str
            id of dataset to save
        dataset : Dataset

        """
        if dataset_id not in self.datasets:
            self.datasets[dataset_id] = dataset
        else:
            raise NameError(f'Dataset with name {dataset_id} already exists in the experiment.')

    def remove_dataset(self, dataset_id):
        """
        Removes dataset from storage.

        Parameters
        ----------
        dataset_id : str
            id of dataset to remove

        """
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]
        else:
            raise NameError(f'There is no dataset with name {dataset_id} in this experiment.')

    @staticmethod
    def _load(load_path, experiment_id: str, save_path: str, tree: dict = None,
              models_info: dict = None, cubes: List[dict] = None,
              criteria: List[List] = [None]):
        """
        Load helper.

        """
        from.experiment import Experiment
        from .models import TopicModel

        root_model_save_path = os.path.join(load_path, START)
        root_model = TopicModel.load(root_model_save_path)
        experiment = Experiment(
            root_model,
            experiment_id=experiment_id,
            save_path=save_path,
            save_experiment=False)
        experiment.tree = Tree(tree=tree)
        experiment.models_info = models_info
        experiment.models = dict.fromkeys(experiment.tree.get_model_ids())
        experiment.models[START] = root_model
        experiment.cubes = cubes
        experiment.criteria = criteria

        return experiment

    def save_models(self, mode='all'):
        """
        Saves experiment models with respect to selected way of saving.

        Parameters
        ----------
        mode : str
            defines saving mode
            'all' - save all models in experiment  
            'tree' - save only stem and leaves from the last level  
            'last' save only leaves from the last level

        """  # noqa: W291
        experiment_save_path = os.path.join(self.save_path, self.experiment_id)

        save_models = set()
        if mode == 'all':
            save_models.update([
                (tmodel, tmodel.model_id)
                for tmodel in self.models.values()
                if is_saveable_model(tmodel)
            ])
        elif mode == 'tree':
            save_models.update([
                (self.models.get(getattr(tmodel, 'parent_model_id', None)),
                 getattr(tmodel, 'parent_model_id', None))
                for tmodel in self.models.values()
                if is_saveable_model(self.models.get(getattr(tmodel, 'parent_model_id', None)))
            ])
        else:
            save_models.update(set([
                (tmodel, tmodel.model_id)
                for tmodel in self.get_models_by_depth(self.depth)
                if is_saveable_model(tmodel)
            ]))

        for model, model_id in list(save_models):
            model_save_path = os.path.join(experiment_save_path, model_id)
            model.save(model_save_path=model_save_path)

    def squeeze_models(self, depth: int = None):
        """Transforms models to dummies so as to occupy less RAM memory

        Parameters
        ----------
        depth : int
            Models on what depth are to be squeezed, i.e. transformed to dummies
        """
        if depth == 0:
            return

        assert isinstance(depth, int) and depth > 0

        for m in self.get_models_by_depth(depth):
            m.make_dummy()

    def save(self, window_size: int = 1500, mode: str = 'all'):
        """
        Saves all params of the experiment to save_path/experiment_id.

        Parameters
        ----------
        window_size : int
            pixels size of window in html description (Default value = 1500)

        """
        experiment_save_path = os.path.join(self.save_path, self.experiment_id)
        if not os.path.exists(experiment_save_path):
            os.makedirs(experiment_save_path)

        self.save_models(mode=mode)

        params = self.get_params()
        json.dump(params, open(f'{experiment_save_path}/params.json', 'w'),
                  default=transform_topic_model_description_to_jsonable)
        html = get_html(self, window_size)
        html_path = os.path.join(experiment_save_path, 'params.html')
        with open(html_path, "w", encoding='utf-8') as f:
            f.write(html)

    @staticmethod
    def load(load_path):
        """
        Loads all params of the experiments. Recovers removed files if it is possible.

        Parameters
        ----------
        load_path : str
            path to the experiment folder.

        Returns
        -------
        Experiment

        """
        from .models import TopicModel

        files = os.listdir(load_path)
        if "params.json" not in files:
            raise FileExistsError("The main file params.json does not exist.")
        else:
            params = json.load(open(f"{load_path}/params.json", "r"))
            params.pop('depth', None)

            experiment = Experiment._load(load_path, **params)
            experiment._recover_consistency(load_path)

            for model_id in experiment.models.keys():
                if model_id != START:
                    model_save_path = os.path.join(load_path, model_id)
                    experiment.models[model_id] = TopicModel.load(
                        model_save_path, experiment
                    )

        return experiment

    def get_description(self,
                        min_len_per_cube: int = MODEL_NAME_LENGTH,
                        len_tree_step: int = MODEL_NAME_LENGTH + 1):
        """
        Creates description of the tree that you can print.
        Print is good when you use no more than 3 cubes at all.

        Parameters
        ----------
        min_len_per_cube : int
            minimal length of the one stage of experiment description
            (Default value = MODEL_NAME_LENGTH)
        len_tree_step : int
            length of the whole one stage description of experiment's tree
            (Default value = MODEL_NAME_LENGTH +1)

        Returns
        -------
        str
            description to print

        """
        strings = give_strings_description(
            self,
            min_len_per_cube=min_len_per_cube,
            len_tree_step=len_tree_step
        )
        description = "\n".join(strings)

        return description

    def show(self):
        """
        Shows description of the experiment.

        """
        nb_verbose = _run_from_notebook()
        string = self.get_description()
        Experiment._clear_and_print(string, nb_verbose)

    def get_models_by_depth(self, level=None):
        """ """
        if level is None:
            # level = self.depth
            level = len(self.cubes)

        return [
            tmodel
            for tmodel in self.models.values()
            if isinstance(tmodel, BaseModel) and tmodel.depth == int(level)
        ]

    def select(self, query_string='', models_num=None, level=None):
        """
        Selects all models satisfying the query string
        from all models on a particular depth.

        Parameters
        ----------
        query_string : str
            string of form "SCORE1 < VAL and SCORE2 > VAL and SCORE3 -> min"
        models_num : int
            number of models to select (Default value = None)
        level : int
            None represents "the last level of experiment" (Default value = None)

        Returns
        -------
        result_topic_models : list of restored TopicModels

        String Format
        -------------
        string of following form:  
        QUERY = EXPR and EXPR and EXPR and ... and EXPR [collect COLLECT_NUMERAL]
        where EXPR could take any of these forms:  
            EXPR = LITERAL < NUMBER  
            EXPR = LITERAL > NUMBER  
            EXPR = LITERAL = NUMBER  
            EXPR = LITERAL -> min  
            EXPR = LITERAL -> max  
        and LITERAL is one of the following:
            SCORE_NAME or model.PARAMETER_NAME
            (for complicated scores you can use '.': e.g. TopicKernelScore.average_purity)
        COLLECT clause is optional. COLLECT_NUMERAL could be integer or string "all"

        NUMBER is float / int or some expression involving special functions:
            MINIMUM, MAXIMUM, AVERAGE, MEDIAN
        Everything is separated by spaces.

        Notes
        -----

        If both models_num and COLLECT_NUMERAL is specified, COLLECT_NUMERAL takes priority.

        If optimization directive is specified, select() may return more models than requested
        (whether by models_num or by COLLECT_NUMERAL). This behaviour occurs when some scores
        are equal.

        For example, if we have 5 models with following scores:
            [model1: 100, model2: 95, model3: 95, model4: 95, model5: 80]
        and user asks experiment to provide 2 models with maximal score,
        then 4 models will be returned:
            [model1: 100, model2: 95, model3: 95, model4: 95]


        Examples
        --------

        >> experiment.select("PerplexityScore@words -> min COLLECT 2")

        >> experiment.select(
            "TopicKernelScore.average_contrast -> max and PerplexityScore@all < 100 COLLECT 2"
        )

        >> experiment.select(
            "PerplexityScore@words < 1.1 * MINIMUM(PerplexityScore@all) and model.num_topics > 12"
        )


        """  # noqa: W291
        from .models import DummyTopicModel
        models_num_as_parameter = models_num
        models_num_from_query = None
        candidate_tmodels = self.get_models_by_depth(level=level)

        if "COLLECT" in query_string:
            first_part, second_part = re.split(r'\s*COLLECT\s+', query_string)

            if second_part.lower() != 'all':
                try:
                    models_num_from_query = int(second_part)
                except ValueError:
                    raise ValueError(f"Invalid directive in COLLECT: {second_part}")
            else:
                models_num_from_query = len(candidate_tmodels)

            query_string = first_part

        models_num = choose_value_for_models_num_and_check(
            models_num_as_parameter, models_num_from_query
        )

        try:
            query_string = self.preprocess_query(query_string, level)
            req_lesser, req_greater, req_equal, metric, extremum = parse_query_string(query_string)

            result = choose_best_models(
                candidate_tmodels,
                req_lesser, req_greater, req_equal,
                metric, extremum,
                models_num
            )
            result_topic_models = [model.restore() if isinstance(model, DummyTopicModel)
                                   else model for model in result]
            return result_topic_models

        except ValueError as e:
            if e.args[0] not in EMPTY_ERRORS:
                raise e

            error_message = repr(e)
            warnings.warn(W_EMPTY_SPECIAL_1 + W_EMPTY_SPECIAL_2.format(error_message))

            return []

    def run(self, dataset, verbose=False, nb_verbose=False):
        """
        Runs defined pipeline and prints out the result.

        Parameters
        ----------
        dataset : Dataset
        verbose : bool
            parameter that determines if the output is produced (Default value = False)
        nb_verbose : bool
            parameter that determines where the output is produced 
            if False prints in console (Default value = False)

        """  # noqa: W291
        stage_models = self.root

        for cube_index, cube_description in enumerate(self.cubes):
            if cube_description['action'] == 'start':
                continue

            cube = cube_description['cube']
            cube(stage_models, dataset)

            # TODO: either delete this line completely
            #  or come up with a way to restore any cube using just info about it in self.cubes
            #  (need to restore cubes for upgrading dummy to topic model)
            # self.cubes[cube_index].pop('cube', None)

            stage_models = self._select_and_save_unique_models(
                self.criteria[cube_index], dataset, cube_index + 1
            )

            if verbose:
                tree_description = "\n".join(self.tree.get_description())
                Experiment._clear_and_print(tree_description, nb_verbose)

            if self._low_memory:
                self.squeeze_models(max(0, self.depth - 2))

        if verbose:
            Experiment._clear_and_print(self.get_description(), nb_verbose)

        if self._low_memory:
            self.squeeze_models(max(0, self.depth - 1))
            self.squeeze_models(self.depth)

        return stage_models

    @staticmethod
    def _clear_and_print(string, nb_verbose):
        if nb_verbose:
            from IPython.display import clear_output
            from IPython.core.display import display_pretty
            clear_output()
            display_pretty(string, raw=True)
        else:
            _ = os.system('cls' if os.name == 'nt' else 'clear')
            print(string)

    def _select_and_save_unique_models(self, templates, dataset, current_level):
        """
        Applies selection criteria to
        last stage models and save successful candidates.

        Parameters
        ----------
        templates : list of str
        dataset : Dataset
        current_level : int

        Returns
        -------
        selected_models : set of TopicModel

        """
        stage_models = sum(
            [self.select(template, level=current_level) for template in templates],
            []
        )
        number_models_selected = len(stage_models)
        stage_models = set(stage_models)
        if number_models_selected > len(stage_models):
            warnings.warn('Some models satisfy multiple criteria')
        for model in stage_models:
            model.save(theta=True, dataset=dataset)
        return stage_models

    def describe_model(self, model_id):
        """
        Returns all scores mentioned on the model stage criteria.

        Parameters
        ----------
        model_id : str
            string id of the model to examine

        Returns
        -------
        description_string : str
        """
        model = self.models[model_id]
        # criteria for selecting models for the following cube
        templates = self.criteria[model.depth - 1]

        score_names = []
        for template in templates:
            score_names += [statement.split()[0] for statement in re.split(r'\s+and\s+', template)]
        score_names = set(score_names)
        description_strings = ['model: ' + model_id]
        for score_name in score_names:
            if 'model.' in score_name:
                attr = score_name.split('.')[1]
                attr_val = getattr(model, attr)
                description_strings += [f'model attribute "{attr}" with value: {attr_val}']
            else:
                try:
                    description_strings += [f'{score_name}: {model.scores[score_name][-1]}']
                except KeyError:
                    raise ValueError(f'Model does not have {score_name} score.')

        description_string = "\n".join(description_strings)
        return description_string

    def preprocess_query(self, query_string: str, level):
        """
        Preprocesses special queries with functions inside.

        Parameters
        ----------
        query_string : str
            string for processing
        level : int
            model level

        """
        queries_list = re.split(r'\s+and\s+', query_string)
        special_functions = [
                    'MINIMUM',
                    'MAXIMUM',
                    'AVERAGE',
                    'MEDIAN',
                ]

        model_queries = []
        special_queries = []
        standard_queries = []
        for query in queries_list:
            if query.startswith('model.'):
                model_queries.append(query)
            elif any(special_function in query for special_function in special_functions):
                special_queries.append(query)
            else:
                standard_queries.append(query)

        if len(model_queries) != 0:
            inner_query_string = ' and '.join(model_queries)
            (req_lesser, req_greater,
             req_equal, metric, extremum) = parse_query_string(inner_query_string)

            if metric is not None or extremum is not None:
                warnings.warn(f'You try to optimize model parameters.')

            candidate_tmodels = self.get_models_by_depth(level=level)
            special_models = choose_best_models(
                candidate_tmodels,
                req_lesser, req_greater, req_equal,
                metric, extremum,
                models_num=None
            )
        else:
            special_models = self.get_models_by_depth(level=level)

        special_queries = compute_special_queries(special_models, special_queries)

        return ' and '.join(standard_queries + model_queries + special_queries)

    def build(self, settings):
        """
        Builds experiment pipeline from description.

        Parameters
        ----------
        settings: list of dicts
            list with cubes parameters for every pipeline step
        Returns
        -------
        Nothing

        """
        import topicnet.cooking_machine.cubes as tncubes

        self.criteria = [None]
        for stage in settings:
            for cube_name, cube_param in stage.items():
                if cube_name == 'selection':
                    stage_criteria = cube_param
                else:
                    try:
                        stage_cube = getattr(tncubes, cube_name)(**cube_param)
                    except Exception as e:
                        error_message = repr(e)
                        raise ValueError(f'Can not create {cube_name} '
                                         f'with parameters {cube_param}.\n'
                                         f'Process failed with following: {error_message}')
            try:
                self.cubes += [{
                    'action': stage_cube.action,
                    'params': stage_cube.get_jsonable_from_parameters(),
                    'cube': stage_cube
                }]
                self.criteria.append(stage_criteria)
                del(stage_cube, stage_criteria)
            except NameError:
                raise NameError('To define pipeline BOTH cube and selection criteria needed')

    def set_criteria(self, cube_index, criteria):
        """
        Allows to edit model selection criteria
        on each stage of the Experiment

        Parameters
        ----------
        cube_index : int
        selection_criteria: list of str or str
            the criteria to replacing current record

        Returns
        -------
        Nothing

        """
        if cube_index >= len(self.cubes):
            raise ValueError(f'Invalid cube_index. There are {len(self.cubes)} cubes.'
                             'You can check it using experiment.cubes')
        else:
            if isinstance(criteria, str):
                criteria = [criteria]
            self.criteria[cube_index] = criteria
