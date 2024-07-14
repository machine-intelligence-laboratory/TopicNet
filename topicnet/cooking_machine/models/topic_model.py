import artm
import dill
import glob
import inspect
import json
import os
import pandas as pd
import pickle
import shutil
import warnings

from copy import deepcopy
from inspect import signature
from numbers import Number
from six import iteritems
from typing import (
    Any,
    Dict,
    List,
    Union,
)

from artm.wrapper.exceptions import ArtmException

from . import scores as tn_scores
from .base_model import BaseModel
from .base_regularizer import BaseRegularizer
from .base_score import BaseScore
from .frozen_score import FrozenScore
from ..cubes.controller_cube import ControllerAgent
from ..routine import transform_complex_entity_to_dict

# TODO: can't import Experiment from here (to specify type in init)
#  probably need to rearrange imports
#  (Experiment and Models are kind of in one bunch: one should be able to know about the other)

from .scores_wrapper import ScoresWrapper


LIBRARY_VERSION = artm.version()
ARTM_NINE = LIBRARY_VERSION.split(".")[1] == "9"

SUPPORTED_SCORES_WITHOUT_VALUE_PROPERTY = (
    artm.score_tracker.TopTokensScoreTracker,
    artm.score_tracker.ThetaSnippetScoreTracker,
    artm.score_tracker.TopicKernelScoreTracker,
)


class TopicModel(BaseModel):
    """
    Topic Model contains artm model and all necessary information: scores, training pipeline, etc.

    """
    def __init__(
            self,
            artm_model: artm.ARTM = None,
            model_id: str = None,
            parent_model_id: str = None,
            data_path: str = None,
            description: List[Dict[str, Any]] = None,
            experiment=None,
            callbacks: List[ControllerAgent] = None,
            custom_scores: Dict[str, BaseScore] = None,
            custom_regularizers: Dict[str, BaseRegularizer] = None,
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

        if callbacks is None:
            callbacks = list()
        if custom_scores is None:
            custom_scores = dict()
        if custom_regularizers is None:
            custom_regularizers = dict()

        self.callbacks = list(callbacks)

        if artm_model is not None:
            self._model = artm_model
        else:
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

        self.data_path = data_path
        self.custom_scores = custom_scores
        self.custom_regularizers = custom_regularizers
        self.library_version = LIBRARY_VERSION

        self._description = []

        if description is None and self._model._initialized:
            init_params = self.get_jsonable_from_parameters()
            self._description = [{"action": "init",
                                  "params": [init_params]}]
        else:
            self._description = description

        self._scores_wrapper = ScoresWrapper(
            topicnet_scores=self.custom_scores,
            artm_scores=self._model.scores
        )

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

    def _prepare_custom_regularizers(self, custom_regularizers):
        if custom_regularizers is None:
            custom_regularizers = dict()

        all_custom_regularizers = deepcopy(custom_regularizers)
        all_custom_regularizers.update(self.custom_regularizers)
        base_regularizers_name, base_regularizers_tau = None, None

        if len(all_custom_regularizers) != 0:
            for regularizer in all_custom_regularizers.values():
                regularizer.attach(self._model)

            base_regularizers_name = [regularizer.name
                                      for regularizer in self._model.regularizers.data.values()]
            base_regularizers_tau = [regularizer.tau
                                     for regularizer in self._model.regularizers.data.values()]

        return base_regularizers_name, base_regularizers_tau, all_custom_regularizers

    def _fit(self, dataset_trainable, num_iterations, custom_regularizers=None):
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
        (base_regularizers_name,
         base_regularizers_tau,
         all_custom_regularizers) = self._prepare_custom_regularizers(custom_regularizers)

        for cur_iter in range(num_iterations):
            precomputed_data = dict()
            iter_is_last = cur_iter == num_iterations - 1

            self._model.fit_offline(batch_vectorizer=dataset_trainable,
                                    num_collection_passes=1)

            if len(all_custom_regularizers) != 0:
                self._apply_custom_regularizers(
                    dataset_trainable, all_custom_regularizers,
                    base_regularizers_name, base_regularizers_tau
                )

            for name, custom_score in self.custom_scores.items():
                try:
                    should_compute_now = iter_is_last or custom_score._should_compute(cur_iter)

                    if not should_compute_now:
                        continue

                    # TODO: this check is probably should be refined somehow...
                    #  what if some new parameter added to BaseScore.call -> new check?..
                    call_parameters = signature(custom_score.call).parameters

                    # if-else instead of try-catch: to speed up
                    if (BaseScore._PRECOMPUTED_DATA_PARAMETER_NAME not in call_parameters
                            and not any(str(p).startswith('**') for p in call_parameters.values())):

                        score = custom_score.call(self)
                    else:
                        score = custom_score.call(self, precomputed_data=precomputed_data)

                    custom_score.update(score)
                    self._model.score_tracker[name] = custom_score

                except AttributeError:  # TODO: means no "call" attribute?
                    raise AttributeError(f'Score {name} doesn\'t have a desired attribute')

            # TODO: think about performance issues
            for callback_agent in self.callbacks:
                callback_agent.invoke(self, cur_iter)

            self._scores_wrapper._reset_score_caches()

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
        attached_rwt = pd.DataFrame(data=nd_array, columns=list(meta.topic_name), index=list(meta.token))

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
        parameters['version'] = self.library_version

        return parameters

    def get_init_parameters(self, not_include=None):
        if not_include is None:
            not_include = list()

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
            # If not do this, there may be problems with pickling:
            # `model` is an ARTM-C-like thing, and it may cause problems
            # This is safe, because `model` appears in attach(),
            # which is called before each iteration
            # P.S. and the `model` itself may be needed for a regularizer inside `grad()`
            regularizer_object._model = None

            managed_to_pickle = False

            for (pickler, extension) in zip([dill, pickle], ['.rd', '.rp']):
                save_path = os.path.join(model_save_path, regularizer_name + extension)

                try:
                    with open(save_path, 'wb') as reg_f:
                        pickler.dump(regularizer_object, reg_f)
                except (TypeError, AttributeError):
                    if os.path.isfile(save_path):
                        os.remove(save_path)
                else:
                    managed_to_pickle = True

                if managed_to_pickle:
                    break

            if not managed_to_pickle:
                warnings.warn(f'Cannot save {regularizer_name} regularizer!')

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
            self._model.get_phi().to_csv(os.path.join(model_save_path, 'phi.csv'))
        if theta:
            self.get_theta(dataset=dataset).to_csv(os.path.join(model_save_path, 'theta.csv'))

        model_itself_save_path = os.path.join(model_save_path, 'model')

        if os.path.exists(model_itself_save_path):
            shutil.rmtree(model_itself_save_path)

        self._model.dump_artm_model(model_itself_save_path)
        self.save_parameters(model_save_path)

        for score_name, score_object in self.custom_scores.items():
            class_name = score_object.__class__.__name__
            save_path = os.path.join(
                model_save_path,
                '.'.join([score_name, class_name, 'p'])
            )

            try:
                score_object.save(save_path)
            except pickle.PicklingError:
                warnings.warn(
                    f'Failed to save custom score "{score_object}" correctly!'
                    f' Freezing score (saving only its value)'
                )

                frozen_score_object = FrozenScore(
                    score_object.value,
                    original_score=score_object
                )
                frozen_score_object.save(save_path)

        self.save_custom_regularizers(model_save_path)

        for i, agent in enumerate(self.callbacks):
            save_path = os.path.join(model_save_path, f"callback_{i}.pkl")

            with open(save_path, 'wb') as agent_file:
                dill.dump(agent, agent_file)

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

        with open(os.path.join(path, 'params.json'), 'r', encoding='utf-8') as params_file:
            params = json.load(params_file)

        topic_model = TopicModel(model, **params)
        topic_model.experiment = experiment

        for score_path in glob.glob(os.path.join(path, '*.p')):
            # TODO: file '..p' is not included, so score with name '.' will be lost
            #  Need to validate score name?
            score_file_name = os.path.basename(score_path)
            *score_name, score_cls_name, _ = score_file_name.split('.')
            score_name = '.'.join(score_name)

            score_cls = getattr(tn_scores, score_cls_name)
            loaded_score = score_cls.load(score_path)
            # TODO check what happens with score name
            loaded_score._name = score_name
            topic_model.scores.add(loaded_score)

        for reg_file_extension, loader in zip(['.rd', '.rp'], [dill, pickle]):
            for regularizer_path in glob.glob(os.path.join(path, f'*{reg_file_extension}')):
                regularizer_file_name = os.path.basename(regularizer_path)
                regularizer_name = os.path.splitext(regularizer_file_name)[0]

                with open(regularizer_path, 'rb') as reg_file:
                    topic_model.custom_regularizers[regularizer_name] = loader.load(reg_file)

        all_agents = glob.glob(os.path.join(path, 'callback*.pkl'))
        topic_model.callbacks = [None for _ in enumerate(all_agents)]

        for agent_path in all_agents:
            file_name = os.path.basename(agent_path).split('.')[0]
            original_index = int(file_name.partition("_")[2])

            with open(agent_path, 'rb') as agent_file:
                topic_model.callbacks[original_index] = dill.load(agent_file)

        topic_model._scores_wrapper._reset_score_caches()
        _ = topic_model.scores

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
            # maybe the next for-loop could be replaced with these three lines
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

    def to_dummy(self, save_path=None):
        """Creates dummy model

        Parameters
        ----------
        save_path : str (or None)
            Path to folder with dumped info about topic model

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

        if save_path is None:
            save_path = self.model_default_save_path

        dummy = DummyTopicModel(
            init_parameters=self.get_init_parameters(),
            scores=dict(self.scores),
            model_id=self.model_id,
            parent_model_id=self.parent_model_id,
            description=self.description,
            experiment=self.experiment,
            save_path=save_path,
        )

        # BaseModel spoils model_id trying to make it unique
        dummy._model_id = self.model_id  # accessing private field instead of public property

        return dummy

    def make_dummy(self, save_to_drive=True, save_path=None, dataset=None):
        """Makes topic model dummy in-place.

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
        but there is no ARTM model inside! (so `model.get_phi()` won't work!)
        If one wants to use the topic model as before,
        this ARTM model should be restored first:
        >>> save_path = topic_model.model_default_save_path
        >>> topic_model._model = artm.load_artm_model(f'{save_path}/model')
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

        dummy = self.to_dummy(save_path=save_path)
        dummy._original_model_save_folder_path = save_path

        self._model.dispose()
        self._model = dummy._model

        del dummy

        setattr(self, DummyTopicModel._dummy_attribute, True)

    @property
    def scores(self) -> Dict[str, List[float]]:
        """
        Gets score values by name.

        Returns
        -------
        dict : string -> list
            dictionary with scores and corresponding values
        """
        if self._scores_wrapper._score_caches is None:
            self._scores_wrapper._score_caches = self._compute_score_values()

        return self._scores_wrapper

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

    def select_topics(self, substrings, invert=False):
        """
        Gets all topics containing specified substring

        Returns
        -------
        list
        """
        return [
            topic_name for topic_name in self.topic_names
            if invert != any(
                substring.lower() in topic_name.lower() for substring in substrings
            )
        ]

    @property
    def background_topics(self):
        return self.select_topics(["background", "bcg"])

    @property
    def specific_topics(self):
        return self.select_topics(["background", "bcg"], invert=True)

    @property
    def class_ids(self):
        """ """
        return self._model.class_ids

    def describe_scores(self, verbose=False):
        data = []
        for score_name, score in self.scores.items():
            data.append([self.model_id, score_name, score[-1]])
        result = pd.DataFrame(columns=["model_id", "score_name", "last_value"], data=data)
        if not verbose:
            printable_types = result.last_value.apply(lambda x: isinstance(x, Number))
            result = result.loc[printable_types]

        return result.set_index(["model_id", "score_name"])

    def describe_regularizers(self):
        data = []
        for reg_name, reg in self.regularizers._data.items():
            entry = [self.model_id, reg_name, reg.tau,
                     reg.gamma, getattr(reg, "class_ids", None)]
            data.append(entry)
        for custom_reg_name, custom_reg in self.custom_regularizers.items():
            entry = [self.model_id, custom_reg_name, custom_reg.tau,
                     custom_reg.gamma, getattr(custom_reg, "class_ids", None)]
            data.append(entry)
        result = pd.DataFrame(
            columns=["model_id", "regularizer_name", "tau", "gamma", "class_ids"], data=data
        )
        return result.set_index(["model_id", "regularizer_name"]).sort_values(by="regularizer_name")

    def get_regularizer(
            self, reg_name: str) -> Union[BaseRegularizer, artm.regularizers.BaseRegularizer]:
        """
        Retrieves the regularizer specified, no matter is it custom or "classic"

        Returns
        -------
        regularizer

        """
        # TODO: RegularizersWrapper?

        if reg_name in self.custom_regularizers:
            return self.custom_regularizers[reg_name]
        elif reg_name in self._model.regularizers.data:
            return self._model.regularizers.data[reg_name]
        else:
            raise KeyError(
                f'There is no such regularizer "{reg_name}"'
                f' among custom and ARTM regularizers!'
            )
