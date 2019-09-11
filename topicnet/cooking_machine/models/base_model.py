import json
import os
from copy import deepcopy

from ..routine import get_timestamp_in_str_format
from ..routine import transform_topic_model_description_to_jsonable

MODEL_NAME_LENGTH = 21


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
        def padd_model_name(model_id):
            padding = MODEL_NAME_LENGTH - len(model_id)
            if padding > 0:
                add = padding // 2
                odd = padding % 2
                return '#' * add + model_id + '#' * (add + odd)
            else:
                return model_id[:MODEL_NAME_LENGTH]
        if self.experiment is None:
            if model_id is None:
                self.set_model_id_as_timestamp()
            else:
                self.model_id = padd_model_name(model_id)
        else:
            if model_id is None:
                candidate_name = get_timestamp_in_str_format()
            else:
                candidate_name = model_id

            version = 0
            while True:
                if candidate_name in self.experiment.models_info:
                    version += 1
                    candidate_name = candidate_name.split('__')[0] + '__' + str(version)
                else:
                    self.model_id = candidate_name
                    break

        self._description = []
        self._scores = dict()
        self._score_functions = dict()

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
            "parent_model_id": self.parent_model_id,
            "data_path": self.data_path,
            "description": self.description,
            "depth": self.depth,
        }
        if self.experiment is None:
            parameters["experiment_id"] = None
        else:
            parameters["experiment_id"] = self.experiment.experiment_id

        return parameters

    @property
    def model_default_save_path(self):
        """ """
        path_possible = all([
            self.experiment.save_path,
            self.experiment.experiment_id,
            self.model_id
        ])
        if path_possible:
            path_to_save = os.path.join(
                self.experiment.save_path,
                self.experiment.experiment_id,
                self.model_id
            )
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
        self._model_id = get_timestamp_in_str_format()

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
