import json
import os
from copy import deepcopy

from ..routine import get_timestamp_in_str_format
from ..routine import transform_topic_model_description_to_jsonable

MODEL_NAME_LENGTH = 26


def padd_model_name(model_id):
    padding = MODEL_NAME_LENGTH - len(model_id)
    if padding > 0:
        add = padding // 2
        odd = padding % 2
        return '#' * add + model_id + '#' * (add + odd)
    else:
        return model_id[:MODEL_NAME_LENGTH]


class BaseModel(object):
    def __init__(self, model_id=None, parent_model_id=None, experiment=None, *args, **kwargs):
        """
        Initialize stage, also used for loading previously saved experiments.

        Parameters
        ----------
        model_id : str
            model id (Default value = None)
        parent_model_id : str
            model id from which current model was created (Default value = None)
        experiment : Experiment
            the experiment to which the model is bound (Default value = None)

        """
        self._parent_model_id = parent_model_id
        self.experiment = experiment

        # set unique model_id in the experiment
        if self.experiment is None:
            if model_id is None:
                self.set_model_id_as_timestamp()
            else:
                self.model_id = padd_model_name(model_id)
        else:
            experiment_save_path = getattr(experiment, 'save_path', None)
            experiment_id = getattr(experiment, 'experiment_id', None)
            save_folder = os.path.join(experiment_save_path, experiment_id)
            if model_id is None:
                candidate_name = get_timestamp_in_str_format()
            else:
                candidate_name = model_id
            model_index = 0
            new_model_id = padd_model_name(candidate_name)
            new_model_save_path = os.path.join(save_folder, new_model_id)
            while os.path.exists(new_model_save_path):
                model_index += 1
                new_model_id = padd_model_name("{0}{1:_>5}".format(candidate_name, model_index))

            self.model_id = new_model_id

        self._description = []
        self._scores = dict()
        self._score_functions = dict()
        self._custom_scores = []

    def __str__(self):
        return f'id={self.model_id}, ' \
               f'parent_id={self.parent_model_id}, ' \
               f'experiment_id={self.experiment.experiment_id if self.experiment is not None else None}'  # noqa long line

    def _fit(self, dataset_trainable, num_iterations):
        """
        Fitting stage.

        Parameters
        ----------
        dataset_trainable : optional
            TODO: describe after dataset implementation
        num_iterations : int
            number of iteration for fitting.

        """
        raise NotImplementedError

    def get_phi(self, *args, **kwargs):
        """ """
        raise NotImplementedError

    def get_theta(self, dataset=None, *args, **kwargs):
        """

        Parameters
        ----------
        dataset : Dataset
             (Default value = None)

        """
        raise NotImplementedError

    def save(self, path, *args, **kwargs):
        """

        Parameters
        ----------
        path : str

        """
        raise NotImplementedError

    @staticmethod
    def load(path, *args, **kwargs):
        """

        Parameters
        ----------
        path : str

        """
        raise NotImplementedError

    def clone(self):
        """ """
        return deepcopy(self)

    def get_jsonable_from_parameters(self):
        """ """
        raise NotImplementedError

    @property
    def score_functions(self):
        """ """
        return self._score_functions

    @property
    def scores(self):
        """ """
        return self._scores

    def add_cube(self, cube):
        """
        Adds cube to the model.

        Parameters
        ----------
        cube : dict
            training cube params.

        """
        self.description.append(cube)
        self.save_parameters()

    @property
    def depth(self):
        """
        Returns depth of the model.

        """
        return len(self.description)

    @property
    def description(self):
        """ """
        return self._description

    @property
    def parent_model_id(self):
        """ """
        return self._parent_model_id

    @parent_model_id.setter
    def parent_model_id(self, new_id):
        """
        Returns parent model id.

        Parameters
        ----------
        new_id : str

        """
        if self._check_is_model_id_in_experiment(new_id):
            self._parent_model_id = new_id
        else:
            raise ValueError(f'Model with id: {new_id} does not exist.')

    def save_parameters(self, model_save_path=None):
        """
        Saves params of the model.

        """
        if model_save_path is None:
            model_save_path = self.model_default_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        parameters = self.get_parameters()
        json.dump(parameters, open(f"{model_save_path}/params.json", "w"),
                  default=transform_topic_model_description_to_jsonable)

    def get_parameters(self):
        """
        Gets all params of the model.

        Returns
        -------
        dict
            parameters of the model

        """
        parameters = {
            "model_id": self.model_id,
            "init_parameters": self.get_init_parameters(),
            "parent_model_id": self.parent_model_id,
            "data_path": self.data_path,
            "description": self.description,
            "depth": self.depth,
            "scores": self.scores
        }
        if self.experiment is None:
            parameters["experiment_id"] = None
        else:
            parameters["experiment_id"] = self.experiment.experiment_id

        return parameters

    @property
    def model_default_save_path(self):
        """ """
        # Experiment may be None. If so, AttributeError is raised
        # __getattr__ catches it in case of TopicModel and redirects to artm_model
        experiment_save_path = getattr(self.experiment, 'save_path', None)
        experiment_id = getattr(self.experiment, 'experiment_id', None)

        assert self.model_id is not None

        path_components = [
            experiment_save_path,
            experiment_id,
            self.model_id
        ]

        path_possible = all(path_components)

        if path_possible:
            path_to_save = os.path.join(*path_components)
        else:
            path_to_save = self.model_id

        return path_to_save

    @property
    def model_id(self):
        """ """
        return self._model_id

    @model_id.setter
    def model_id(self, new_id):
        """

        Parameters
        ----------
        new_id : str

        """
        if self._check_is_model_id_in_experiment(new_id):
            raise ValueError(f'Model with id: {new_id} already exists.')
        else:
            self._model_id = new_id

    def set_model_id_as_timestamp(self):
        """ """
        self._model_id = padd_model_name(get_timestamp_in_str_format())

    def _check_is_model_id_in_experiment(self, model_id):
        """

        Parameters
        ----------
        model_id : str

        """
        if self.experiment is None:
            return False
        if model_id in self.experiment.models_info.keys():
            return True
        return False
