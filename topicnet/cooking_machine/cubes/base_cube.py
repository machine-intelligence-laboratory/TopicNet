import os
from tqdm import tqdm
import warnings
from multiprocessing import SimpleQueue, Process

from .strategy import BaseStrategy
from ..models.base_model import padd_model_name
from ..routine import get_timestamp_in_str_format


def check_experiment_existence(topic_model):
    """
    Checks if topic_model has experiment.

    Parameters
    ----------
    topic_model : TopicModel
        topic model

    Returns
    -------
    bool
        True if experiment exists, in other case False.

    """
    is_experiment = topic_model.experiment is not None

    return is_experiment


class retrieve_score_for_strategy:
    def __init__(self, score_name):
        self.score_name = score_name

    def __call__(self, model):
        if isinstance(model, str):
            self.score_name = model
        else:
            return model.scores[self.score_name][-1]


class BaseCube:
    """
    Abstract class for all cubes.

    """
    def __init__(self, num_iter, action=None, reg_search="grid",
                 strategy=None, tracked_score_function=None, verbose=False):
        """
        Initialize stage.
        Checks params and update .parameters attribute.

        Parameters
        ----------
        num_iter : int
            number of iterations or method
        action : str
            stage of creation
        reg_search : str
            "grid" or "pair". "pair" for elementwise grid search in the case
            of several regularizers, "grid" for the fullgrid search in the
            case of several regularizers
        strategy : BaseStrategy
            optimization approach
        tracked_score_function : str or callable
            optimizable function for strategy
        verbose : bool
            visualization flag

        """
        self.num_iter = num_iter
        self.parameters = []
        self.action = action
        self.reg_search = reg_search
        if not strategy:
            strategy = BaseStrategy()
        self.strategy = strategy
        self.verbose = verbose

        if isinstance(tracked_score_function, str):
            tracked_score_function = retrieve_score_for_strategy(tracked_score_function)
        self.tracked_score_function = tracked_score_function

    def apply(self, topic_model, one_cube_parameter, dictionary=None, model_id=None):
        """
        "apply" method changes topic_model in way that is defined by one_cube_parameter.

        Parameters
        ----------
        topic_model : TopicModel
            topic model
        one_cube_parameter : optional
            parameters of one experiment
        dictionary : dict
            dictionary so that the it can be used
            on the basis of the model (Default value = None)
        model_id : str
            id of created model if necessary (Default value = None)

        Returns
        -------

        """
        raise NotImplementedError('must be implemented in subclass')

    # TODO: из-за метода get_description на эту фунцию налагется больше требований чем тут написано
    def get_jsonable_from_parameters(self):
        """
        Transform self.parameters to something that can be downloaded as json.

        Parameters
        ----------

        Returns
        -------
        optional
            something jsonable

        """
        return self.parameters

    def _train_models(self, queue, experiment, topic_model, dataset,
                      dataset_trainable, search_space, search_length):
        """
        This function trains models in separate thread, saves them
        and returns all paths for save with respect to train order.
        To preserve train order model number is also returned.

        """
        def verbose_enumerate(space, verbose, total_length):
            """

            Parameters
            ----------
            space : optional
            verbose : bool
            total_length : int

            Returns
            -------

            """
            if verbose:
                space = tqdm(space, total=total_length)
            for param_id, point in enumerate(space):
                yield param_id, point

        with warnings.catch_warnings(record=True) as caught_warnings:
            returned_paths = []
            experiment_save_path = getattr(experiment, 'save_path', None)
            experiment_id = getattr(experiment, 'experiment_id', None)
            save_folder = os.path.join(experiment_save_path, experiment_id)
            for parameter_id, search_point in verbose_enumerate(search_space,
                                                                self.verbose,
                                                                search_length):
                new_model_def_id = get_timestamp_in_str_format()
                new_model_id = padd_model_name(new_model_def_id)
                new_model_save_path = os.path.join(save_folder, new_model_id)
                model_index = 0
                while os.path.exists(new_model_save_path):
                    new_model_id = padd_model_name(new_model_def_id + '_' + str(model_index))
                    new_model_save_path = os.path.join(save_folder, new_model_id)
                    model_index += 1

                new_model = self.apply(topic_model, search_point,
                                       dataset.get_dictionary(), new_model_id)

                model_cube = {
                    "action": self.action,
                    "num_iter": self.num_iter,
                    "params": repr(search_point)
                }

                new_model._fit(
                    dataset_trainable=dataset_trainable,
                    num_iterations=self.num_iter
                )
                new_model.add_cube(model_cube)
                new_model.experiment = experiment
                new_model.save()

                returned_paths.append((parameter_id, new_model.model_default_save_path))

                # some strategies depend on previous train results, therefore scores must be updated
                if self.tracked_score_function:
                    current_score = self.tracked_score_function(new_model)
                else:
                    # we return number of iterations as a placeholder
                    current_score = len(returned_paths)
                self.strategy.update_scores(current_score)

            queue.put(len(returned_paths))
            for path_tuple in returned_paths:
                queue.put(path_tuple)

            # to work with strategy we recover consistency by sending important parameters
            strategy_parameters = self.strategy._get_strategy_parameters(saveable_only=True)
            queue.put(strategy_parameters)

            caught_warnings = [(warning.message, warning.category)
                               for warning in caught_warnings]
            queue.put(caught_warnings)

    def _run_cube(self, topic_model, dataset):
        """
        Apply cube to topic_model. Get new models and fit them on batch_vectorizer.
        Return list of all trained models.

        Parameters
        ----------
        topic_model : TopicModel
        dataset : Dataset

        Returns
        -------
        TopicModel

        """
        from ..models import TopicModel

        # create log
        # TODO: будет странно работать, если бесконечный список
        parameter_description = self.get_jsonable_from_parameters()
        cube_description = {
                'action': self.action,
                'params': parameter_description
        }

        # at one level only one cube can be implemented
        if not check_experiment_existence(topic_model):
            raise ValueError("TopicModel has no experiment. You should create Experiment.")
        experiment = topic_model.experiment
        topic_model_depth_in_tree = topic_model.depth
        if topic_model_depth_in_tree < len(experiment.cubes):
            existed_cube = experiment.cubes[topic_model_depth_in_tree]
            if existed_cube['params'] != cube_description['params'] or \
                    existed_cube['action'] != cube_description['action']:
                error_message = (
                    "\nYou can not change strategy to another on this level in "
                    "this experiment.\n"
                    "If you want you can create another experiment with this "
                    "model with parameter new_experiment=True."
                    f"the existing cube is \n {existed_cube['params']} \n, "
                    f"but the proposed cube is \n {cube_description['params']} \n"
                )
                raise ValueError(error_message)
            is_new_exp_cube = False
        else:
            is_new_exp_cube = True

        dataset_trainable = dataset._transform_data_for_training()

        # perform all experiments
        self.strategy.prepare_grid(self.parameters, self.reg_search)
        search_space = self.strategy.grid_visit_generator(self.parameters, self.reg_search)
        search_length = getattr(self.strategy, 'grid_len', None)
        queue = SimpleQueue()
        process = Process(
            target=self._train_models,
            args=(queue, experiment, topic_model, dataset,
                  dataset_trainable, search_space, search_length)
        )
        process.start()

        models_num = queue.get()
        topic_models_dict = {}
        for _ in range(models_num):
            path_tuple = queue.get()
            topic_models_dict[path_tuple[0]] = TopicModel.load(path_tuple[1], experiment=experiment)

        strategy_parameters = queue.get()
        caught_warnings = queue.get()

        process.join()

        for warning in caught_warnings:
            if issubclass(warning[1], UserWarning):
                warnings.warn(warning[0])

        topic_models = list(dict(sorted(topic_models_dict.items())).values())
        for topic_model in topic_models:
            experiment.add_model(topic_model)

        self.strategy._set_strategy_parameters(strategy_parameters)

        if is_new_exp_cube:
            experiment.add_cube(cube_description)

        return topic_models

    def __call__(self, topic_model_input, dataset):
        """
        Apply cube to topic_model. Get new models and fit them on batch_vectorizer.
        Return list of all trained models.

        Parameters
        ----------
        topic_model_input: TopicModel or list of TopicModel
        dataset: Dataset

        Returns
        -------
        list of TopicModel

        """
        if isinstance(topic_model_input, (list, set)):
            results = [
                self._run_cube(topic_model, dataset)
                for topic_model in topic_model_input
            ]
            return results
        return self._run_cube(topic_model_input, dataset)
