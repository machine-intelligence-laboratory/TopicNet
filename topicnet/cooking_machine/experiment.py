import os
import json
import shutil
import warnings

from .model_tracking import Tree
from typing import List

from .pretty_output import give_strings_description, get_html
from .routine import transform_topic_model_description_to_jsonable
from .routine import parse_query_string, choose_best_models, compute_special_queries
from .routine import is_saveable_model

from .models import BaseModel

import topicnet.cooking_machine.cubes as tncubes

START = '<'*8 + 'start' + '>'*8

UP_END = "┌"
DOWN_END = "└"
MIDDLE = "├"
LAST = "┤"
EMPTY = "│"
START_END = "┐"
SPACE = " "


class Experiment(object):
    """
    Contains experiment, its description and descriptions of all models in the experiment.

    """
    def __init__(self, experiment_id: str, save_path: str, topic_model=None,
                 save_model_history: bool = False, save_experiment: bool = True,
                 tree: dict = None, models_info: dict = None, cubes: List[dict] = None):
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

        """  # noqa: W291

        self.experiment_id = experiment_id

        if os.path.exists(save_path) and save_experiment:
            folders = os.listdir(save_path)
            if experiment_id in folders:
                raise FileExistsError(
                    f"In /{save_path} experiment {experiment_id} already exists"
                )

        self.save_path = save_path

        # if you want to create an empty Experiment (only experiment_id and save_path must be known)
        if topic_model is None:
            self.tree = Tree()
            self.models_info = {START: 'Start training point.'}
            self.cubes = [{
                'action': 'start',
                'params': [{'version': 'not defined'}]
            }]
            self.models = {START: BaseModel(model_id=START, experiment=self)}
            self.criteria = [None]
        else:
            if save_model_history:
                self._prune_experiment(topic_model)
            else:
                self.cubes = [
                    {
                        'action': 'start',
                        'params': [{'version':
                                    topic_model.get_jsonable_from_parameters()['version']}]
                    },
                    {
                        'action': 'init',
                        'params': [topic_model.get_jsonable_from_parameters()],
                    }
                ]
                self.criteria = [None] * 2
                self.models_info = {
                    START: 'Start training point.',
                    topic_model.model_id: topic_model.get_jsonable_from_parameters()
                }
                self.models = {
                    START: BaseModel(model_id=START, experiment=self),
                    topic_model.model_id: topic_model
                }
                topic_model.experiment = self
                topic_model.parent_model_id = START
                self.tree = Tree()
                self.tree.add_model(topic_model)
                topic_model.save_parameters()

        if save_experiment:
            self.save()
        self.datasets = dict()

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
            with open(f"{experiment_save_path}/params.html", "w") as f:
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
    def _load(experiment_id: str, save_path: str, tree: dict = None,
              models_info: dict = None, cubes: List[dict] = None):
        """
        Load helper.

        """
        from.experiment import Experiment

        experiment = Experiment(experiment_id=experiment_id, save_path=save_path,
                                save_experiment=False)
        experiment.tree = Tree(tree=tree)
        experiment.models_info = models_info
        experiment.models = dict.fromkeys(experiment.tree.get_model_ids())
        experiment.cubes = cubes

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
        save_models.update(set([
            (tmodel, tmodel.model_id)
            for tmodel in self.get_models_by_depth(self.true_depth)
            if is_saveable_model(tmodel)
        ]))

        for model in list(save_models):
            model_save_path = os.path.join(experiment_save_path, model[0].model_id)
            model[0].save(model_save_path=model_save_path)

        files = [
            file
            for file in os.listdir(experiment_save_path)
            if os.path.isdir(os.path.join(experiment_save_path, file))
        ]
        for file in files:
            model_delete_path = os.path.join(experiment_save_path, file, 'model')
            if file not in list(zip(*save_models))[1] and os.path.exists(model_delete_path):
                shutil.rmtree(model_delete_path)

    def save(self, window_size: int = 1500, mode: str = 'tree'):
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
        with open(html_path, "w") as f:
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

            experiment = Experiment._load(**params)
            experiment._recover_consistency(load_path)

            for model_id in experiment.models.keys():
                if model_id != START:
                    experiment.models[model_id] = TopicModel.load(
                        f'{load_path}/{model_id}', experiment
                    )

        return experiment

    def get_description(self, min_len_per_cube: int = 21, len_tree_step: int = 22):
        """
        Creates description of the tree that you can print.
        Print is good when you use no more than 3 cubes at all.

        Parameters
        ----------
        min_len_per_cube : int
            minimal length of the one stage of experiment description (Default value = 21)
        len_tree_step : int
            length of the whole one stage description of experiment's tree (Default value = 22)

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
        print(self.get_description())

    def get_models_by_depth(self, level=None):
        """ """
        if level is None:
            level = self.true_depth
        return [
            tmodel
            for tmodel in self.models.values()
            if isinstance(tmodel, BaseModel) and tmodel.depth == level
        ]

    @property
    def true_depth(self):
        """
        Returns true depth of the tree.

        """
        return max(tmodel.depth for tmodel in self.models.values() if isinstance(tmodel, BaseModel))

    def select(self, query_string, models_num=1, level=None):
        """
        Selects a best model according to the query string among all models on a particular depth.

        Parameters
        ----------
        query_string : str
            string of form "SCORE1 < VAL and SCORE2 > VAL and SCORE3 -> min"
            (see parse_query_string function for details)
        models_num : int
            number of models to select (Default value = 1)
        level : int
            None represents "the last level of experiment" (Default value = None)

        Returns
        -------
        TopicModel
            model - an element of experiment

        """
        if "COLLECT" in query_string:
            first_part, delim, second_part = query_string.partition(" COLLECT ")
            try:
                models_num = int(second_part)
            except ValueError:
                raise ValueError(f"Invalid directive in COLLECT: {second_part}")
            query_string = first_part

        query_string = self.preprocess_query(query_string, level)
        req_lesser, req_greater, req_equal, metric, extremum = parse_query_string(query_string)
        candidate_tmodels = self.get_models_by_depth(level=level)
        result = choose_best_models(
            candidate_tmodels,
            req_lesser, req_greater, req_equal,
            metric, extremum,
            models_num
        )
        return result

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

        def clear_and_print(string, nb_verbose):
            if nb_verbose:
                from IPython.display import clear_output
                from IPython.core.display import display_pretty
                clear_output()
                display_pretty(string, raw=True)
            else:
                _ = os.system('cls' if os.name == 'nt' else 'clear')
                print(string)

        for model_id in self.models.keys():
            if model_id is not START:
                stage_models = [self.models[model_id]]

        for cube_index, cube_description in enumerate(self.cubes):
            if cube_description['action'] == 'start':
                continue
            cube = cube_description['cube']
            if isinstance(cube, tncubes.CubeCreator):
                cube(self.root, dataset)
            else:
                cube(stage_models, dataset)
            self.cubes[cube_index].pop('cube', None)
            stage_models = self._select_and_save_unique_models(self.criteria[cube_index], dataset)

            if verbose:
                tree_description = "\n".join(self.tree.get_description())
                clear_and_print(tree_description, nb_verbose)

        if verbose:
            clear_and_print(self.get_description(), nb_verbose)
        return stage_models

    def _select_and_save_unique_models(self, templates, dataset):
        """
        Applies selection criteria to
        last stage models and save sucessfull candidates.

        Parameters
        ----------
        templates : list of str

        Returns
        -------
        selected_models : set of TopicModel

        """
        stage_models = sum([self.select(template) for template in templates], [])
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

        """
        model = self.models[model_id]
        # criteria for selecting models for the following cube
        templates = self.criteria.get(model.depth, [])

        score_names = []
        for template in templates:
            score_names += [statement.split()[0] for statement in template.split('and')]
        score_names = set(score_names)
        print('model: ', model_id)
        for score_name in score_names:
            if 'model.' in score_name:
                attr = score_name.split('.')[1]
                attr_val = getattr(model, attr)
                print(f'model attribute "{attr}" with value: {attr_val}')
            else:
                try:
                    print(score_name + ': ', model.scores[score_name][-1])
                except KeyError:
                    raise ValueError(f'Model does not have {score_name} score.')

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
        queries_list = query_string.split(' and ')
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

        """
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

    def set_criterion(self, cube_index, criterion):
        if cube_index >= len(self.cubes):
            raise ValueError(f'Invalid cube_index. There are {len(self.cubes)} cubes.'
                             'You can check it using experiment.cubes')
        else:
            if isinstance(criterion, str):
                criterion = [criterion]
            self.criteria[cube_index] = criterion
