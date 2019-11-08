from .base_model import BaseModel
from ..routine import transform_complex_entity_to_dict

import os
import json
import glob
import shutil
import dill as pickle
import pandas as pd
import warnings
import inspect

import artm
from artm.wrapper.exceptions import ArtmException

from six import iteritems
from copy import deepcopy

from inspect import signature

ARTM_NINE = artm.version().split(".")[1] == "9"

SUPPORTED_SCORES_WITHOUT_VALUE_PROPERTY = (
    artm.score_tracker.TopTokensScoreTracker,
    artm.score_tracker.ThetaSnippetScoreTracker,
    artm.score_tracker.TopicKernelScoreTracker,
)


class EmptyScore:
    def __init__(self):
        self.value = []


class TopicModel(BaseModel):
    """
    Topic Model contains artm model and all necessary information: scores, training pipeline, etc.

    """
    def __init__(self, artm_model=None, model_id=None,
                 parent_model_id=None, data_path=None,
                 description=None, experiment=None,
                 callbacks=list(), depth=0, scores=dict(),
                 custom_scores=dict(), *args, **kwargs):
        """
        Initialize stage, also used for loading previously saved experiments.

        Parameters
        ----------
        artm_model : artm model or None
            model to use, None if you want to create model (Default value = None)
        model_id : str
            model id (Default value = None)
        parent_model_id : str
            model id from which current model was created (Default value = None)
        data_path : str
            path to the data (Default value = None)
        description : list of dict
            description of the model (Default value = None)
        experiment : Experiment
            the experiment to which the model is bound (Default value = None)
        callbacks : list of objects with invoke() method
            function called inside _fit which alters model parameters
            mainly used for fancy regularizer coefficients manipulation
        custom_scores : dict
            dictionary with score names as keys and score classes as functions
            (score class with functionality like those of BaseScore)

        """
        super().__init__(model_id=model_id, parent_model_id=parent_model_id,
                         experiment=experiment, *args, **kwargs)
        self.callbacks = list(callbacks)

        if artm_model is None:
            artm_ARTM_args = inspect.getfullargspec(artm.ARTM).args
            kwargs = {k: v for k, v in kwargs.items() if k in artm_ARTM_args}

            try:
                self._model = artm.ARTM(**kwargs)
            except ArtmException as e:
                error_message = repr(e)
                raise ValueError(
                    f'Cannot create artm model with parameters {kwargs}.\n'
                    "ARTM failed with following: " + error_message
                )
        else:
            self._model = artm_model

        self.data_path = data_path
        self.custom_scores = custom_scores

        self._score_caches = None  # returned by model.score, reset by model._fit

        self._description = []
        if description is None and self._model._initialized:
            init_params = self.get_jsonable_from_parameters()
            self._description = [{"action": "init",
                                  "params": [init_params]}]
        else:
            self._description = description

    def __getattr__(self, attr_name):
        return getattr(self._model, attr_name)

    def _get_all_scores(self):
        if len(self._model.score_tracker.items()) == 0:
            yield from {key: EmptyScore() for key in self._model.scores.data.keys()}.items()
        yield from self._model.score_tracker.items()

        if self.custom_scores is not None:  # default is dict(), but maybe better to set None?
            yield from self.custom_scores.items()

    def _reset_score_caches(self):
        self._score_caches = None

    def _compute_score_values(self):
        def get_score_properties_and_values(score_name, score_object):
            for internal_name in dir(score_object):
                if internal_name.startswith('_') or internal_name.startswith('last'):
                    continue

                score_property_name = score_name + '.' + internal_name

                yield score_property_name, getattr(score_object, internal_name)

        score_values = dict()

        for score_name, score_object in self._get_all_scores():
            try:
                score_values[score_name] = getattr(score_object, 'value')
            except AttributeError:
                if not isinstance(score_object, SUPPORTED_SCORES_WITHOUT_VALUE_PROPERTY):
                    warnings.warn(f'Score "{str(score_object.__class__)}" is not supported')
                    continue

                for score_property_name, value in get_score_properties_and_values(
                        score_name, score_object):

                    score_values[score_property_name] = value

        return score_values

    def _fit(self, dataset_trainable, num_iterations):
        for cur_iter in range(num_iterations):
            self._model.fit_offline(batch_vectorizer=dataset_trainable,
                                    num_collection_passes=1)

            for name, custom_score in self.custom_scores.items():
                try:
                    score = custom_score.call(self)
                    custom_score.update(score)
                    self._model.score_tracker[name] = custom_score
                except AttributeError:  # TODO: means no "call" attribute?
                    raise AttributeError(f'Score {name} doesn\'t have a desired attribute')

            # TODO: think about performance issues
            for callback_agent in self.callbacks:
                callback_agent.invoke(self, cur_iter)

            self._reset_score_caches()

    def get_jsonable_from_parameters(self):
        """
        Gets artm model params.

        Returns
        -------
        dict
            artm model parameters

        """
        parameters = transform_complex_entity_to_dict(self._model)

        regularizers = {}
        for name, regularizer in iteritems(self._model._regularizers.data):
            tau = None
            gamma = None
            try:
                tau = regularizer.tau
                gamma = regularizer.gamma
            except KeyError:
                pass
            regularizers[name] = [str(regularizer.config), tau, gamma]
        parameters['regularizers'] = regularizers

        parameters['version'] = artm.version()

        return parameters

    def get_init_parameters(self, not_include=list()):
        init_artm_parameter_names = [
            p.name for p in list(signature(artm.ARTM.__init__).parameters.values())
        ][1:]
        parameters = transform_complex_entity_to_dict(self._model)
        filtered = dict()
        for parameter_name, parameter_value in parameters.items():
            if parameter_name not in not_include and parameter_name in init_artm_parameter_names:
                filtered[parameter_name] = parameter_value
        return filtered

    def save(self,
             model_save_path=None,
             phi=True,
             theta=False,
             dataset=None,):
        """
        Saves model description and dumps artm model.
        Use this method if you want to dump the model.

        Parameters
        ----------
        model_save_path : str
            path to the folder with dumped info about model
        phi : bool
            save phi in csv format if True
        theta : bool
            save theta in csv format if True
        dataset : Dataset
             dataset

        """
        if model_save_path is None:
            model_save_path = self.model_default_save_path

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if phi:
            self._model.get_phi().to_csv(f"{model_save_path}/phi.csv")
        if theta:
            self.get_theta(dataset=dataset).to_csv(f"{model_save_path}/theta.csv")

        model_itself_save_path = f"{model_save_path}/model"

        if os.path.exists(model_itself_save_path):
            shutil.rmtree(model_itself_save_path)
        self._model.dump_artm_model(model_itself_save_path)
        self.save_parameters(model_save_path)

        for score_name, score_object in self.custom_scores.items():
            save_path = os.path.join(model_save_path, score_name + '.p')
            pickle.dump(score_object, open(save_path, 'wb'))

        for i, agent in enumerate(self.callbacks):
            save_path = os.path.join(model_save_path, f"callback_{i}.pkl")
            pickle.dump(agent, open(save_path, 'wb'))

    @staticmethod
    def load(path, experiment=None):
        """
        Loads the model.

        Parameters
        ----------
        path : str
            path to the model's folder
        experiment : Experiment

        Returns
        -------
        TopicModel

        """

        if "model" in os.listdir(f"{path}"):
            model = artm.load_artm_model(f"{path}/model")
        else:
            model = None
            print("There is no dumped model. You should train it again.")
        params = json.load(open(f"{path}/params.json", "r"))
        topic_model = TopicModel(model, **params)
        topic_model.experiment = experiment

        custom_scores = {}
        for score_path in glob.glob(os.path.join(path, '*.p')):
            score_name = os.path.basename(score_path).split('.')[0]
            custom_scores[score_name] = pickle.load(open(score_path, 'rb'))
        topic_model.custom_scores = custom_scores

        all_agents = glob.glob(os.path.join(path, 'callback*.pkl'))
        topic_model.callbacks = [None for _ in enumerate(all_agents)]
        for agent_path in all_agents:
            filename = os.path.basename(agent_path).split('.')[0]
            original_index = int(filename.partition("_")[2])
            topic_model.callbacks[original_index] = pickle.load(open(agent_path, 'rb'))

        topic_model._reset_score_caches()
        return topic_model

    def clone(self, model_id=None):
        """
        Creates a copy of the model except model_id.

        Parameters
        ----------
        model_id : str
            (Default value = None)

        Returns
        -------
        TopicModel

        """
        topic_model = TopicModel(artm_model=self._model.clone(),
                                 model_id=model_id,
                                 parent_model_id=self.parent_model_id,
                                 description=deepcopy(self.description),
                                 custom_scores=deepcopy(self.custom_scores),
                                 experiment=self.experiment)
        topic_model._score_functions = deepcopy(topic_model.score_functions)
        topic_model._scores = deepcopy(topic_model.scores)
        topic_model.callbacks = deepcopy(self.callbacks)

        return topic_model

    def get_phi(self, topic_names=None, class_ids=None, model_name=None):
        """
        Gets custom Phi matrix of model.

        Parameters
        ----------
        topic_names : list of str or str
            list with topics or single topic to extract,
            None value means all topics (Default value = None)
        class_ids : list of str or str
            list with class_ids or single class_id to extract,
            None means all class ids (Default value = None)
        model_name : str
            self.model.model_pwt by default, self.model.model_nwt is also
            reasonable to extract unnormalized counters

        Returns
        -------
        pd.DataFrame
            phi matrix

        """
        if ARTM_NINE:
            phi_parts_array = []
            if isinstance(class_ids, str):
                class_ids = [class_ids]
            class_ids_iter = class_ids or self._model.class_ids
            # TODO: this workaround seems to be a correct solution to this problem
            if not class_ids_iter:
                valid_model_name = self._model.model_pwt
                info = self._model.master.get_phi_info(valid_model_name)
                class_ids_iter = list(set(info.class_id))

            for class_id in class_ids_iter:
                phi_part = self._model.get_phi(topic_names, class_id, model_name)
                phi_part.index.rename("token", inplace=True)
                phi_part.reset_index(inplace=True)
                phi_part["modality"] = class_id
                phi_parts_array.append(phi_part)
            phi = pd.concat(phi_parts_array).set_index(['modality', 'token'])
        else:
            phi = self._model.get_phi(topic_names, class_ids, model_name)
            phi.index = pd.MultiIndex.from_tuples(phi.index, names=('modality', 'token'))

        return phi

    def get_phi_dense(self, topic_names=None, class_ids=None, model_name=None):
        """
        Gets custom Phi matrix of model.

        Parameters
        ----------
        topic_names : list of str or str
            list with topics or single topic to extract,
            None value means all topics (Default value = None)
        class_ids : list of str or str
            list with class_ids or single class_id to extract,
            None means all class ids (Default value = None)
        model_name : str
            self.model.model_pwt by default, self.model.model_nwt is also
            reasonable to extract unnormalized counters

        Returns
        -------
        3-tuple
            dense phi matrix

        """
        return self._model.get_phi_dense(topic_names, class_ids, model_name)

    def get_phi_sparse(self, topic_names=None, class_ids=None, model_name=None, eps=None):
        """
        Gets custom Phi matrix of model as sparse scipy matrix.

        Parameters
        ----------
        topic_names : list of str or str
            list with topics or single topic to extract,
            None value means all topics (Default value = None)
        class_ids : list of str or str
            list with class_ids or single class_id to extract,
            None means all class ids (Default value = None)
        model_name : str
            self.model.model_pwt by default, self.model.model_nwt is also
            reasonable to extract unnormalized counters
        eps : float
            threshold to consider values as zero (Default value = None)

        Returns
        -------
        3-tuple
            sparse phi matrix

        """
        return self._model.get_phi_sparse(topic_names, class_ids, model_name, eps)

    def get_theta(self, topic_names=None,
                  dataset=None,
                  theta_matrix_type='dense_theta',
                  predict_class_id=None,
                  sparse=False,
                  eps=None,):
        """
        Gets Theta matrix as pandas DataFrame
        or sparse scipy matrix.

        Parameters
        ----------
        topic_names : list of str or str
            list with topics or single topic to extract,
            None value means all topics (Default value = None)
        dataset : Dataset
            an instance of Dataset class (Default value = None)
        theta_matrix_type : str
            type of matrix to be returned, possible values:
            ‘dense_theta’, ‘dense_ptdw’, ‘cache’, None (Default value = ’dense_theta’)
        predict_class_id : str
            class_id of a target modality to predict. When this option
            is enabled the resulting columns of theta matrix will
            correspond to unique labels of a target modality. The values
            will represent p(c|d), which give the probability of class
            label c for document d (Default value = None)
        sparse : bool
            if method returns sparse representation of the data (Default value = False)
        eps : float
            threshold to consider values as zero. Required for sparse matrix.
            depends on the collection (Default value = None)

        Returns
        -------
        pd.DataFrame
            theta matrix

        """
        # assuming particular case of BigARTM library that user can't get theta matrix
        # without cache_theta == True. This also covers theta_name == None case
        if self._model._cache_theta:
            # TODO wrap sparse in pd.SparseDataFrame and check that viewers work with that output
            if sparse:
                return self._model.get_theta_sparse(topic_names, eps)
            else:
                return self._model.get_theta(topic_names)
        else:
            if dataset is None:
                raise ValueError("To get theta a dataset is required")
            else:
                batch_vectorizer = dataset.get_batch_vectorizer()
                if sparse:
                    return self._model.transform_sparse(batch_vectorizer, eps)
                else:
                    theta = self._model.transform(batch_vectorizer,
                                                  theta_matrix_type,
                                                  predict_class_id)
                    return theta

    def to_dummy(self):
        """Creates dummy model

        Returns
        -------
        DummyTopicModel
            Dummy model: without inner ARTM model,
            but with scores and init parameters of calling TopicModel

        Notes
        -----
        Dummy model has the same model_id as the original model,
        but "model_id" key in experiment.models contains original model, not dummy
        """
        from .dummy_topic_model import DummyTopicModel
        # python crashes if place this import on top of the file
        # import circle: TopicModel -> DummyTopicModel -> TopicModel

        dummy = DummyTopicModel(
            init_parameters=self.get_init_parameters(),
            scores=self.scores,
            model_id=self.model_id,
            parent_model_id=self.parent_model_id,
            description=self.description,
            experiment=self.experiment
        )

        # BaseModel spoils model_id trying to make it unique
        dummy._model_id = self.model_id  # accessing private field instead of public property

        return dummy

    def make_dummy(self, save_to_drive=True, save_path=None, dataset=None):
        """Makes topic model dummy in-place

        Parameters
        ----------
        save_to_drive : bool
            Whether to save model to drive or not. If not, the info will be lost
        save_path : str (or None)
            Path to folder to dump info to
        dataset : Dataset
            Dataset with text collection on which the model was trained.
            Needed for saving Theta matrix

        Notes
        -----
        After calling the method, the model is still of type TopicModel,
        but all its attributes are now like DummyTopicModel's
        """
        from .dummy_topic_model import DummyTopicModel
        from .dummy_topic_model import WARNING_ALREADY_DUMMY

        if hasattr(self, DummyTopicModel._dummy_attribute):
            warnings.warn(WARNING_ALREADY_DUMMY)

            return

        if not save_to_drive:
            save_path = None
        else:
            save_path = save_path or self.model_default_save_path
            save_theta = self._model._cache_theta or (dataset is not None)
            self.save(save_path, phi=True, theta=save_theta, dataset=dataset)

        dummy = self.to_dummy()
        dummy._original_model_save_folder_path = save_path

        self._model.dispose()
        self._model = dummy._model

        del dummy

        setattr(self, DummyTopicModel._dummy_attribute, True)

    @property
    def scores(self):
        """
        Gets score values by name.

        Returns
        -------
        score_values : dict : string -> list
            dictionary with scores and corresponding values

        """
        if self._score_caches is None:  # assume users won't try to corrupt _score_caches
            self._score_caches = self._compute_score_values()

        assert self._score_caches is not None  # maybe empty dict, but not None

        return self._score_caches

    @property
    def description(self):
        """ """
        return self._description

    @property
    def regularizers(self):
        """ """
        return self._model.regularizers

    @property
    def class_ids(self):
        """ """
        return self._model.class_ids

    def describe_scores(self):
        data = []
        for score_name, score in self.scores.items():
            data.append([self.model_id, score_name, score[-1]])
        result = pd.DataFrame(columns=["model_id", "score_name", "last_value"], data=data)
        return result.set_index(["model_id", "score_name"])

    def describe_regularizers(self):
        data = []
        for reg_name, reg in self.regularizers._data.items():
            data.append([self.model_id, reg_name, reg.tau, reg.gamma])
        result = pd.DataFrame(columns=["model_id", "regularizer_name", "tau", "gamma"], data=data)
        return result.set_index(["model_id", "regularizer_name"])
