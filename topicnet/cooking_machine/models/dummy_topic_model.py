import artm
import json
import os
import re
import warnings

from ..dataset import Dataset
from .topic_model import TopicModel

# change log style
lc = artm.messages.ConfigureLoggingArgs()
lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)


class InvalidOperationError(RuntimeError):
    def __init__(self, message='Dummy model can\'t do this'):
        super().__init__(message)


SIMPLE_ARTM_MODEL = artm.ARTM(num_topics=1, num_processors=1)
JSON_KEY_REGULARIZERS = 'regularizers'
JSON_KEY_CLASS_IDS = 'class_ids'
WARNING_ALREADY_DUMMY = 'Already dummy'


class DummyTopicModel(TopicModel):
    _save_path_suffix = '__dummy'
    _dummy_attribute = '_is_dummy'

    def __init__(self,
                 scores,
                 init_parameters=None,
                 model_id=None,
                 parent_model_id=None,
                 description=None,
                 experiment=None,
                 *args,
                 **kwargs):
        """
        Notes
        -----
        Only TopicModel supposed to be able to create DummyTopicModel
        ("private" < access < "public")
        """
        super().__init__(
            artm_model=SIMPLE_ARTM_MODEL,
            model_id=model_id,
            parent_model_id=parent_model_id,
            description=description,
            experiment=experiment,
            **kwargs,
        )

        self._model.dispose()
        self._save_folder_path = None
        self._model = _DummyArtmModel(self._save_folder_path)
        self._init_parameters = init_parameters

        self._scores = scores

        setattr(self, DummyTopicModel._dummy_attribute, True)

        self._original_model_save_folder_path = None

    def __getattr__(self, name):
        # Don't redirect the stuff to artm_model (as TopicModel does)
        if name in self._init_parameters:
            return self._init_parameters[name]
        raise AttributeError(f'Dummy model has no attribute "{name}"')

    def get_init_parameters(self, not_include=None):
        """"""
        return self._init_parameters

    @property
    def model_default_save_path(self):
        """"""
        return super().model_default_save_path + DummyTopicModel._save_path_suffix

    @property
    def scores(self):
        """"""
        return self._scores

    @property
    def regularizers(self):
        """"""
        return self._model.regularizers

    @property
    def class_ids(self):
        """"""
        return self._model.class_ids

    def save(self, model_save_path=None, **kwargs):
        """"""
        # kwargs - for compatibility with super()'s method

        # TODO: a bit copy-paste from TopicModel:
        #  can't call super()'s, because artm_model is being saved by default there

        self._save_folder_path = model_save_path or self.model_default_save_path

        if not os.path.exists(self._save_folder_path):
            os.makedirs(self._save_folder_path)

        self.save_parameters(self._save_folder_path)

    @staticmethod
    def load(path, experiment=None):
        """"""
        params = json.load(open(f'{path}/params.json', 'r'))
        model = DummyTopicModel(**params)
        model.experiment = experiment
        model._original_model_save_folder_path = path

        return model

    def restore(self, dataset: Dataset = None):
        """Restores dummy to original TopicModel

        Tries to load the data from drive (if model was saved).
        Otherwise tries to train the model using parent model, experiment and dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset on which the model was trained.
            If the original model was saved to drive, the parameter won't be used.
            If not, dataset should be provided for training.

        Returns
        -------
        TopicModel
            Restored topic model
        """
        # Not in-place, as TopicModel's make_dummy() because (seems like) TopicModel can be empty
        # But it would be really strange if DummyTopicModel actually had all the stuff inside

        topic_model = None

        if self._original_model_save_folder_path is not None:
            topic_model = self._load_original_model()

        if topic_model is None:
            topic_model = self._train_to_original_model(dataset)

        return topic_model

    def to_dummy(self, save_to_drive=True, save_path=None, **kwargs):
        warnings.warn(WARNING_ALREADY_DUMMY, UserWarning)

        if save_to_drive:
            self.save(save_path, **kwargs)

        return self

    def make_dummy(self, save_to_drive=True, save_path=None, **kwargs):
        warnings.warn(WARNING_ALREADY_DUMMY, UserWarning)

        if save_to_drive:
            self.save(save_path, **kwargs)

    def _load_original_model(self):
        # TODO: custom_scores not restored currently
        #  modify model's save()-load() methods?
        topic_model = None

        try:
            topic_model = super().load(
                self._original_model_save_folder_path,
                self.experiment
            )
        except FileNotFoundError as e:
            warnings.warn(f'Failed to read data from drive: "{e.args}"')

        return topic_model

    def _train_to_original_model(self, dataset: Dataset):
        # TODO: refactor: big bunch of code, a lot of obscure and highly-likely-fo-fail places
        #  (parsing params, connecting one params with other params, restoring cube, running cube)

        if len(self.description) == 0:
            raise RuntimeError(
                'Dummy model has empty description. So seems like nothing to restore'
            )

        if self.parent_model_id is None:
            raise ValueError(
                'Dummy model has no parent. Can\'t restore model in such a case'
            )

        if self.parent_model_id not in self.experiment.models:
            raise ValueError(
                f'Parent model "{self.parent_model_id}" not found in models '
                f'associated with the experiment'
            )

        if dataset is None:
            raise ValueError('Can\'t restore the model via training without dataset')

        parent_model = self.experiment.models[self.parent_model_id]

        if hasattr(parent_model, DummyTopicModel._dummy_attribute):
            assert hasattr(parent_model, 'restore')

            parent_model.restore(True, dataset)  # also restore in experiment.models

            delattr(parent_model, DummyTopicModel._dummy_attribute)

        last_cube_description = self.description[-1]
        # {
        #   'action': 'reg_modifier',
        #   'num_iter': 1,
        #   'params': <some string with some description of regularizers>
        # }
        #
        # Example of 'params' (it is string):
        #   "([<artm.regularizers.SmoothSparseThetaRegularizer object at 0x7faba8363ac8>,
        #     'tau', 10.0],)"

        # Currently need to parse the string with params
        cube_parameters_from_description = last_cube_description['params']
        cube_parameters_from_description = re.findall(
            '\\[.*?\\]',
            cube_parameters_from_description
        )
        cube_parameters_from_description = list(map(
            lambda p: p[1:-1].split(', '),
            cube_parameters_from_description
        ))
        cube_parameters_from_description = list(map(
            lambda p: dict(zip(['object', 'field', 'value'], p)),
            cube_parameters_from_description
        ))

        assert len(self.experiment.cubes) >= len(self.description)

        last_cube_parameters = self.experiment.cubes[len(self.description) - 1]
        # {
        #   'action': 'reg_modifier',
        #   'params': [
        #     {
        #       'tau_grid': [0, 0.0],
        #       'regularizer': { 'name': 'smooth_theta_bcg', 'tau': 1, ... }
        #     },
        #     ...
        #  ],
        #  'cube': <Cube object>
        # }

        # For some reason some cubes seemed to not have this 'cube' parameter
        # and not just the first two cubes
        assert 'cube' in last_cube_parameters

        cube = last_cube_parameters['cube']

        # Example of cube.parameters:
        # [
        #   { 'object': <Regularizer object>, 'field': 'tau', 'values': [0, 0.0] }
        # ]

        # TODO: assume order in cube.parameters is the same as in self.description[-1]['params]
        #  otherwise need to sort both lists?
        for i in range(len(cube.parameters)):
            assert str(cube.parameters[i]['object']) == \
                   cube_parameters_from_description[i]['object']
            # one is object, another is string

            cube.parameters[i]['values'] = float(
                cube_parameters_from_description[i]['value']
            )

        cube_parameters_for_apply = list(
            map(lambda p: list(p.values()), cube.parameters)
        )

        being_restored_model = cube.apply(
            parent_model,
            cube_parameters_for_apply,
            dataset.get_dictionary()
        )
        being_restored_model._fit(
            dataset_trainable=dataset.get_batch_vectorizer(),
            num_iterations=cube.num_iter
        )
        model_cube = {
            'action': cube.action,
            'num_iter': cube.num_iter,
            'params': repr(tuple(cube_parameters_for_apply))  # trying to make it look like before
        }
        being_restored_model.add_cube(model_cube)  # restoring description
        being_restored_model._model_id = self.model_id  # using private field

        return being_restored_model


class _DummyArtmModel:
    def __init__(self, save_folder_path):
        self.master = None

        self._save_folder_path = save_folder_path
        self._artm_params = None

    def __getattr__(self, attr):
        raise AttributeError(f'Dummy ARTM model doesn\'t have such attribute "{attr}"')

    def dispose(self):
        pass

    @property
    def regularizers(self):
        """ """
        assert JSON_KEY_REGULARIZERS in self._artm_parameters

        return self._artm_parameters[JSON_KEY_REGULARIZERS]

    @property
    def class_ids(self):
        """ """
        assert JSON_KEY_CLASS_IDS in self._artm_parameters

        return self._artm_parameters[JSON_KEY_CLASS_IDS]

    def _load_artm_parameters(self):
        if self._save_folder_path is None:
            raise ValueError('Model has never been saved. Can\'t load parameters')

        artm_parameters_file_path = os.path.join(
            self._save_folder_path,
            'model',  # TODO: need some const-s for these names
            'parameters.json'
        )

        if not os.path.isfile(artm_parameters_file_path):
            raise FileNotFoundError(
                f'File with artm model parameters not found on path "{artm_parameters_file_path}"')

        return json.loads(
            open(artm_parameters_file_path, 'r').read()
        )

    @property
    def _artm_parameters(self):
        if self._artm_params is None:
            self._artm_params = self._load_artm_parameters()

        return self._artm_params

    def _fit(self, dataset_trainable, num_iterations):
        raise InvalidOperationError()

    def get_jsonable_from_parameters(self):
        raise InvalidOperationError()

    def clone(self):
        raise InvalidOperationError()

    def get_phi(self, *args, **kwargs):
        raise InvalidOperationError()

    def get_phi_dense(self, *args, **kwargs):
        raise InvalidOperationError()

    def get_phi_sparse(self, *args, **kwargs):
        raise InvalidOperationError()

    def get_theta(self, *args, **kwargs):
        raise InvalidOperationError()

    def add_cube(self, cube):
        raise InvalidOperationError()
