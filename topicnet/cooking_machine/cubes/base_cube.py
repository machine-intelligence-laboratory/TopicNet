from tqdm import tqdm

from .strategy import BaseStrategy


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


def choose_best_model(models, metric):
    """
    Get best model according to specified metric.

    Parameters
    ----------
    models : list
        list of models with .scores parameter
    metric : str
        metric for selection

    Returns
    -------
    TopicModel
        model with best score
    float
        best score.

    """
    if metric not in models[0].scores:
        raise ValueError(f'There is no {metric} metric for model {models[0].model_id}.')
    best_model, best_score = models[0], models[0].scores[metric][-1]
    for model in models[1:]:
        current_score = model.scores[metric][-1]
        if best_score < current_score:
            best_score = current_score
            best_model = model
    return best_model, best_score


def retrieve_perplexity_score(topic_model):
    """

    Parameters
    ----------
    topic_model : TopicModel

    Returns
    -------

    """
    score_name = "PerplexityScore"
    value_name = "value"
    model = topic_model._model
    values = getattr(model.score_tracker[score_name], value_name)
    return values


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
            stage of creation (Defatult value = None)
        reg_search : str
            "grid" or "pair". "pair" for elementwise grid search in the case
            of several regularizers, "grid" for the fullgrid search in the
            case of several regularizers (Defatult value = "grid")
        strategy : BaseStrategy
            optimization approach (Defatult value = None)
        tracked_score_function : retrieve_score_for_strategy
            optimizable function for strategy (Defatult value = None)
        verbose : bool
            visualization flag (Defatult value = False)

        """
        self.num_iter = num_iter
        self.parameters = []
        self.action = action
        self.reg_search = reg_search
        if not strategy:
            strategy = BaseStrategy()
        self.strategy = strategy
        self.tracked_score_function = tracked_score_function
        self.verbose = verbose

    def apply(self, topic_model, one_cube_parameter, dictionary=None):
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
        # print(is_new_exp_cube, experiment.depth, experiment.true_depth, topic_model_depth_in_tree)

        dataset_trainable = dataset._transform_data_for_training()

        # perform all experiments
        topic_models = []
        strategy = self.strategy
        strategy.prepare_grid(self.parameters, self.reg_search)
        search_space = strategy.grid_visit_generator(self.parameters, self.reg_search)
        search_length = getattr(strategy, 'grid_len', None)

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

        for parameter_id, search_point in verbose_enumerate(search_space,
                                                            self.verbose,
                                                            search_length):
            new_model = self.apply(topic_model, search_point, dataset.get_dictionary())

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
            experiment.add_model(new_model)
            topic_models.append(new_model)

            if self.tracked_score_function:
                current_score = self.tracked_score_function(new_model)
            else:
                # we return number of iterations as a placeholder
                current_score = len(topic_models)
            strategy.update_scores(current_score)

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
