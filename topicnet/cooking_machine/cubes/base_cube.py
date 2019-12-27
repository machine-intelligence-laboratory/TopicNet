import os
from tqdm import tqdm
import warnings
from multiprocessing import Queue, Process
# from queue import Empty
from artm.wrapper.exceptions import ArtmException

from .strategy import BaseStrategy
from ..models.base_model import padd_model_name
from ..routine import get_timestamp_in_str_format

NUM_MODELS_ERROR = "Failed to retrive number of trained models"
MODEL_RETRIEVE_ERROR = "Retrieved only {0} models out of {1}"
STRATEGY_RETRIEVE_ERROR = 'Failed to retrieve strategy parameters'
WARNINGS_RETRIEVE_ERROR = 'Failed to return warnings'


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


# exists for multiprocessing debug
def put_to_queue(queue, puttable):
    queue.put(puttable)


# exists for multiprocessing debug
def get_from_queue_till_fail(queue,  error_message='',):
    while True:
        return queue.get()


class BaseCube:
    """
    Abstract class for all cubes.

    """
    def __init__(self, num_iter, action=None, reg_search="grid",
                 strategy=None, tracked_score_function=None,
                 verbose=False, separate_thread=True):
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
        separate_thread : bool
            will train models inside a separate thread if True

        """
        self.num_iter = num_iter
        self.parameters = []
        self.action = action
        self.reg_search = reg_search
        if not strategy:
            strategy = BaseStrategy()
        self.strategy = strategy
        self.verbose = verbose
        self.separate_thread = separate_thread

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

    def _train_models(self, experiment, topic_model, dataset, search_space):
        """
        This function trains models
        """
        dataset_trainable = dataset._transform_data_for_training()
        dataset_dictionary = dataset.get_dictionary()
        returned_paths = []
        experiment_save_path = experiment.save_path
        experiment_id = experiment.experiment_id
        save_folder = os.path.join(experiment_save_path, experiment_id)
        for search_point in search_space:
            candidate_name = get_timestamp_in_str_format()
            new_model_id = padd_model_name(candidate_name)
            new_model_save_path = os.path.join(save_folder, new_model_id)
            model_index = 0
            while os.path.exists(new_model_save_path):
                model_index += 1
                new_model_id = padd_model_name("{0}{1:_>5}".format(candidate_name, model_index))
                new_model_save_path = os.path.join(save_folder, new_model_id)

            model_cube = {
                "action": self.action,
                "num_iter": self.num_iter,
                "params": repr(search_point)
            }

            try:
                # alter the model according to cube parameters
                new_model = self.apply(topic_model, search_point, dataset_dictionary, new_model_id)
                # train new model for a number of iterations (might be zero)
                new_model._fit(
                    dataset_trainable=dataset_trainable,
                    num_iterations=self.num_iter
                )
            except ArtmException as e:
                error_message = repr(e)
                raise ValueError(
                    f'Cannot alter and fit artm model with parameters {search_point}.\n'
                    "ARTM failed with following: " + error_message

                )
            # add cube description to the model history
            new_model.add_cube(model_cube)
            new_model.experiment = experiment
            new_model.save()
            assert os.path.exists(new_model.model_default_save_path)

            returned_paths.append(new_model.model_default_save_path)

            # some strategies depend on previous train results, therefore scores must be updated
            if self.tracked_score_function:
                current_score = self.tracked_score_function(new_model)
                self.strategy.update_scores(current_score)
            # else:
                # we return number of iterations as a placeholder
                # current_score = len(returned_paths)

        return returned_paths

    def _retrieve_results_from_process(self, queue, experiment):
        from ..models import DummyTopicModel
        models_num = get_from_queue_till_fail(queue, NUM_MODELS_ERROR)
        topic_models = []
        for _ in range(models_num):
            path = get_from_queue_till_fail(queue,
                                            MODEL_RETRIEVE_ERROR.format(_, models_num))
            topic_models.append(DummyTopicModel.load(path, experiment=experiment))

        strategy_parameters = get_from_queue_till_fail(queue, STRATEGY_RETRIEVE_ERROR)
        caught_warnings = get_from_queue_till_fail(queue, WARNINGS_RETRIEVE_ERROR)
        self.strategy._set_strategy_parameters(strategy_parameters)

        for (warning_message, warning_class) in caught_warnings:
            # if issubclass(warning_class, UserWarning):
            warnings.warn(warning_message)

        return topic_models

    def _train_models_and_report_results(self, queue, experiment, topic_model, dataset,
                                         search_space, search_length):
        """
        This function trains models in separate thread, saves them
        and returns all paths for save with respect to train order.
        To preserve train order model number is also returned.

        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            returned_paths = self._train_models(experiment, topic_model, dataset, search_space)
            put_to_queue(queue, len(returned_paths))
            for path in returned_paths:
                put_to_queue(queue, path)

            # to work with strategy we recover consistency by sending important parameters
            strategy_parameters = self.strategy._get_strategy_parameters(saveable_only=True)
            put_to_queue(queue, strategy_parameters)

            caught_warnings = [(warning.message, warning.category)
                               for warning in caught_warnings]
            put_to_queue(queue, caught_warnings)

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

        from ..models import DummyTopicModel
        if isinstance(topic_model, DummyTopicModel):
            topic_model = topic_model.restore()

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

        # perform all experiments
        self.strategy.prepare_grid(self.parameters, self.reg_search)
        search_space = self.strategy.grid_visit_generator(self.parameters, self.reg_search)
        search_length = getattr(self.strategy, 'grid_len', None)

        if self.verbose:
            search_space = tqdm(search_space, total=search_length)

        if self.separate_thread:
            queue = Queue()
            process = Process(
                target=self._train_models_and_report_results,
                args=(queue, experiment, topic_model, dataset,
                      search_space, search_length),
                daemon=True
            )
            process.start()
            topic_models = self._retrieve_results_from_process(queue, experiment)
        else:
            returned_paths = self._train_models(experiment, topic_model, dataset, search_space)
            topic_models = [
                DummyTopicModel.load(path, experiment=experiment)
                for path in returned_paths
            ]

        for topic_model in topic_models:
            topic_model.data_path = dataset._data_path
            experiment.add_model(topic_model)

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
