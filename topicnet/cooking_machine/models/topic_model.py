from .base_model import BaseModel
from .frozen_score import FrozenScore
from ..routine import transform_complex_entity_to_dict

import os
import json
import glob
import dill
import pickle
import shutil
import pandas as pd
import warnings
import inspect

import artm
from artm.wrapper.exceptions import ArtmException

from six import iteritems
from copy import deepcopy

from inspect import signature

# change log style
lc = artm.messages.ConfigureLoggingArgs()
lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)

ARTM_NINE = artm.version().split(".")[1] == "9"

SUPPORTED_SCORES_WITHOUT_VALUE_PROPERTY = (
    artm.score_tracker.TopTokensScoreTracker,
    artm.score_tracker.ThetaSnippetScoreTracker,
    artm.score_tracker.TopicKernelScoreTracker,
)


class TopicModel(BaseModel):
    """
    Topic Model contains artm model and all necessary information: scores, training pipeline, etc.

    """
    def __init__(self, artm_model=None, model_id=None,
                 parent_model_id=None, data_path=None,
                 description=None, experiment=None,
                 callbacks=list(), depth=0, scores=dict(),
                 custom_scores=dict(), custom_regularizers=dict(),
                 *args, **kwargs):
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
        custom_regularizers : dict
            dictionary with regularizer names as keys and regularizer classes as values

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
        self.custom_regularizers = custom_regularizers

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
            yield from {
                key: FrozenScore(list())
                for key in self._model.scores.data.keys()
            }.items()
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

    def _fit(self, dataset_trainable, num_iterations, custom_regularizers=dict()):
        """

        Parameters
        ----------
        dataset_trainable : BatchVectorizer
            Data for model fit
        num_iterations : int
            Amount of fit steps
        custom_regularizers : dict of BaseRegularizer
            Regularizers to apply to model

        """
        all_custom_regularizers = deepcopy(custom_regularizers)
        all_custom_regularizers.update(self.custom_regularizers)

        if len(all_custom_regularizers) != 0:
            for regularizer in all_custom_regularizers.values():
                regularizer.attach(self._model)

            base_regularizers_name = [regularizer.name
                                      for regularizer in self._model.regularizers.data.values()]
            base_regularizers_tau = [regularizer.tau
                                     for regularizer in self._model.regularizers.data.values()]

        for cur_iter in range(num_iterations):
            self._model.fit_offline(batch_vectorizer=dataset_trainable,
                                    num_collection_passes=1)

            if len(all_custom_regularizers) != 0:
                self._apply_custom_regularizers(
                    dataset_trainable, all_custom_regularizers,
                    base_regularizers_name, base_regularizers_tau
                )

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

    def _apply_custom_regularizers(self, dataset_trainable, custom_regularizers,
                                   base_regularizers_name, base_regularizers_tau):
        """

        Parameters
        ----------
        dataset_trainable : BatchVectorizer
            Data for model fit
        custom_regularizers : dict of BaseRegularizer
            Regularizers to apply to model
        base_regularizers_name : list of str
            List with all artm.regularizers names, applied to model
        base_regularizers_tau : list of float
            List with tau for all artm.regularizers, applied to model

        """
        pwt = self._model.get_phi(model_name=self._model.model_pwt)
        nwt = self._model.get_phi(model_name=self._model.model_nwt)
        rwt_name = 'rwt'

        self._model.master.regularize_model(pwt=self._model.model_pwt,
                                            nwt=self._model.model_nwt,
                                            rwt=rwt_name,
                                            regularizer_name=base_regularizers_name,
                                            regularizer_tau=base_regularizers_tau)

        (meta, nd_array) = self._model.master.attach_model(rwt_name)
        attached_rwt = pd.DataFrame(data=nd_array, columns=meta.topic_name, index=meta.token)

        for regularizer in custom_regularizers.values():
            attached_rwt.values[:, :] += regularizer.grad(pwt, nwt)

        self._model.master.normalize_model(pwt=self._model.model_pwt,
                                           nwt=self._model.model_nwt,
                                           rwt=rwt_name)

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
        for name, regularizer in iteritems(self.custom_regularizers):
            tau = getattr(regularizer, 'tau', None)
            gamma = getattr(regularizer, 'gamma', None)
            config = str(getattr(regularizer, 'config', ''))
            regularizers[name] = [config, tau, gamma]

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

    def save_custom_regularizers(self, model_save_path=None):
        if model_save_path is None:
            model_save_path = self.model_default_save_path

        for regularizer_name, regularizer_object in self.custom_regularizers.items():
            try:
                save_path = os.path.join(model_save_path, regularizer_name + '.rd')
                with open(save_path, 'wb') as reg_f:
                    dill.dump(regularizer_object, reg_f)
            except (TypeError, AttributeError):
                try:
                    save_path = os.path.join(model_save_path, regularizer_name + '.rp')
                    with open(save_path, 'wb') as reg_f:
                        pickle.dump(regularizer_object, reg_f)
                except (TypeError, AttributeError):
                    warnings.warn(f'Cannot save {regularizer_name} regularizer.')

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
            with open(save_path, 'wb') as score_f:
                try:
                    dill.dump(score_object, score_f)
                except pickle.PicklingError:
                    warnings.warn(
                        f'Failed to save custom score "{score_object}" correctly! '
                        f'Freezing score (saving only its value)'
                    )

                    frozen_score_object = FrozenScore(score_object.value)
                    dill.dump(frozen_score_object, score_f)

        self.save_custom_regularizers(model_save_path)

        for i, agent in enumerate(self.callbacks):
            save_path = os.path.join(model_save_path, f"callback_{i}.pkl")
            with open(save_path, 'wb') as agent_f:
                dill.dump(agent, agent_f)

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

        with open(f"{path}/params.json", "r", encoding='utf-8') as params_f:
            params = json.load(params_f)

        topic_model = TopicModel(model, **params)
        topic_model.experiment = experiment

        custom_scores = {}

        for score_path in glob.glob(os.path.join(path, '*.p')):
            score_name = os.path.basename(score_path).split('.')[0]
            with open(score_path, 'rb') as score_f:
                custom_scores[score_name] = dill.load(score_f)

        topic_model.custom_scores = custom_scores

        custom_regularizers = {}

        for regularizer_path in glob.glob(os.path.join(path, '*.rd')):
            regularizer_name = os.path.basename(regularizer_path).split('.')[0]
            with open(regularizer_path, 'rb') as reg_f:
                custom_regularizers[regularizer_name] = dill.load(reg_f)

        for regularizer_path in glob.glob(os.path.join(path, '*.rp')):
            regularizer_name = os.path.basename(regularizer_path).split('.')[0]
            with open(regularizer_path, 'rb') as reg_f:
                custom_regularizers[regularizer_name] = pickle.load(reg_f)

        topic_model.custom_regularizers = custom_regularizers

        all_agents = glob.glob(os.path.join(path, 'callback*.pkl'))
        topic_model.callbacks = [None for _ in enumerate(all_agents)]

        for agent_path in all_agents:
            filename = os.path.basename(agent_path).split('.')[0]
            original_index = int(filename.partition("_")[2])
            with open(agent_path, 'rb') as agent_f:
                topic_model.callbacks[original_index] = dill.load(agent_f)

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
                                 custom_regularizers=deepcopy(self.custom_regularizers),
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
        if self._cache_theta:
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
        """
        Gets regularizers from model.

        """
        return self._model.regularizers

    @property
    def all_regularizers(self):
        """
        Gets all regularizers with custom regularizers.

        Returns
        -------
        regularizers_dict : dict
            dict with artm.regularizer and BaseRegularizer instances

        """
        regularizers_dict = dict()
        for custom_regularizer_name, custom_regularizer in self.custom_regularizers.items():
            regularizers_dict[custom_regularizer_name] = custom_regularizer
        regularizers_dict.update(self._model.regularizers.data)

        return regularizers_dict

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
        for custom_reg_name, custom_reg in self.custom_regularizers.items():
            data.append([self.model_id, custom_reg_name, custom_reg.tau, custom_reg.gamma])
        result = pd.DataFrame(columns=["model_id", "regularizer_name", "tau", "gamma"], data=data)
        return result.set_index(["model_id", "regularizer_name"])
