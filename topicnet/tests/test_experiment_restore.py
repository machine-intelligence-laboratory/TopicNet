import os
import pytest
import shutil

from typing import (
    Dict,
    List,
)

import artm

from ..cooking_machine.dataset import Dataset
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.models.base_score import BaseScore
from ..cooking_machine.models.topic_model import TopicModel

_MAIN_MODALITY = "@text"

_ONE_CUBE_NUM_ITERATIONS = 10
_NUM_CUBES = 3
_INTERRUPT_CUBE_ITERATION = 5
_ONCE_INTERRUPTED = [False]

_TAU_GRID = [0.1, 0.5, 1.0]  # better three or more values (for testing purposes)

_SCORE_NAME = 'bad_score'
_REGULARIZER_NAME = 'reglurellriser'
_SELECT_CRITERION_FOR_ALL_MODELS = ''

_DEBUG_MODE = False


class InterruptingScore(BaseScore):
    def __init__(self, name: str, interrupt_cube: int, interrupt_tau: float):
        super().__init__(name=name)

        self._iteration = 0
        self._interrupt_cube = interrupt_cube
        self._interrupt_tau = interrupt_tau

    def call(self, model: TopicModel) -> float:
        regularizer_tau = model.regularizers[_REGULARIZER_NAME].tau
        current_cube = model.depth
        current_cube_iteration = (
            len(model.scores[_SCORE_NAME]) - (current_cube - 1) * _ONE_CUBE_NUM_ITERATIONS
        )

        if (current_cube == self._interrupt_cube
                and regularizer_tau == self._interrupt_tau
                and current_cube_iteration >= _INTERRUPT_CUBE_ITERATION
                and not _ONCE_INTERRUPTED[0]):

            _ONCE_INTERRUPTED[0] = True

            raise KeyboardInterrupt()

        self._iteration += 1

        return self._iteration


class TestExperimentRestore:
    dataset = None
    dictionary = None
    experiments_save_path = None

    @classmethod
    def setup_class(cls):
        cls.dataset = Dataset('tests/test_data/test_dataset.csv')
        cls.dictionary = cls.dataset.get_dictionary()
        cls.experiments_save_path = 'tests/experiments'

    def setup_method(self):
        _ONCE_INTERRUPTED[0] = False

        os.makedirs(self.experiments_save_path, exist_ok=True)

    def teardown_method(self):
        if os.path.isdir(self.experiments_save_path):
            shutil.rmtree(self.experiments_save_path)

    @classmethod
    def teardown_class(cls):
        if os.path.isdir(cls.experiments_save_path):
            shutil.rmtree(cls.experiments_save_path)

        if cls.dataset is not None:
            cls.dataset.clear_folder()

    @pytest.mark.parametrize(
        'interrupt_cube_index, interrupt_model_index',
        [(1, 0), (0, -1), (-1, 1)]
    )
    def test_ctrl_c_and_proceed(self, interrupt_cube_index, interrupt_model_index):
        self._test_ctrl_c_and_proceed(
            interrupt_cube_index=interrupt_cube_index,
            interrupt_model_index=interrupt_model_index,
            thread_flag=False,
            load_experiment=False,
        )

    # TODO: something happens in multiprocess, and it takes infinity to wait till the end
    @pytest.mark.xfail
    @pytest.mark.timeout(10)
    @pytest.mark.parametrize(
        'interrupt_cube_index, interrupt_model_index',
        [(1, 0)]  # , (0, -1), (-1, 1)]
    )
    def test_ctrl_c_and_proceed_multiprocess(self, interrupt_cube_index, interrupt_model_index):
        self._test_ctrl_c_and_proceed(
            interrupt_cube_index=interrupt_cube_index,
            interrupt_model_index=interrupt_model_index,
            thread_flag=True,
            load_experiment=False,
        )

    # TODO: cubes are loaded as strings, not as Python objects -> experiment.run fails
    @pytest.mark.xfail
    @pytest.mark.parametrize(
        'interrupt_cube_index, interrupt_model_index',
        [(1, 0), (0, -1), (-1, 1)]
    )
    def test_ctrl_c_and_load(self, interrupt_cube_index, interrupt_model_index):
        self._test_ctrl_c_and_proceed(
            interrupt_cube_index=interrupt_cube_index,
            interrupt_model_index=interrupt_model_index,
            thread_flag=False,
            load_experiment=True,
        )

    @pytest.mark.xfail
    @pytest.mark.timeout(10)
    @pytest.mark.parametrize(
        'interrupt_cube_index, interrupt_model_index',
        [(1, 0)]  # , (0, -1), (-1, 1)]
    )
    def test_ctrl_c_and_load_multiprocess(self, interrupt_cube_index, interrupt_model_index):
        self._test_ctrl_c_and_proceed(
            interrupt_cube_index=interrupt_cube_index,
            interrupt_model_index=interrupt_model_index,
            thread_flag=True,
            load_experiment=True,
        )

    def _test_ctrl_c_and_proceed(
            self,
            interrupt_cube_index: int,
            interrupt_model_index: int,
            thread_flag: bool,
            load_experiment: bool) -> None:

        experiment = self._initialize_experiment(
            experiment_id=f'Experiment_{thread_flag}',
            interrupt_cube_index=interrupt_cube_index,
            interrupt_model_index=interrupt_model_index,
        )
        cube_settings = self._initialize_cube_settings(thread_flag)
        experiment.build(cube_settings)

        models: List[TopicModel] = None
        is_interrupt_detected = False

        try:
            experiment.run(
                self.dataset, verbose=False, nb_verbose=False
            )
        except KeyboardInterrupt:
            is_interrupt_detected = True

            if load_experiment:
                experiment = Experiment.load(
                    os.path.join(experiment.save_path, experiment.experiment_id)
                )  # TODO: need to concatenate?

            models = experiment.run(
                self.dataset, verbose=False, nb_verbose=False,
                restore_mode=True,
            )
        finally:
            self._print_debug_info(experiment)

        assert is_interrupt_detected, 'No KeyboardInterrupt detected!'

        self._check_result(cube_settings, experiment, models)

    def _initialize_experiment(
            self,
            experiment_id: str,
            interrupt_cube_index: int,
            interrupt_model_index: int) -> Experiment:

        artm_model = artm.ARTM(
            num_processors=1,
            num_topics=5,
            cache_theta=True,
            num_document_passes=1,
            dictionary=self.dictionary,
            scores=[
                artm.PerplexityScore(
                    name='PerplexityScore'
                ),
                artm.SparsityPhiScore(
                    name='SparsityPhiScore', class_id=_MAIN_MODALITY
                )
            ]
        )

        topic_model = TopicModel(artm_model, model_id='start_id')
        interrupt_cube = list(range(_NUM_CUBES))[interrupt_cube_index] + 1
        interrupt_tau = _TAU_GRID[interrupt_model_index]
        topic_model.scores.add(
            InterruptingScore(
                name=_SCORE_NAME,
                interrupt_cube=interrupt_cube,
                interrupt_tau=interrupt_tau,
            )
        )

        return Experiment(
            topic_model,
            experiment_id=experiment_id,
            save_path=self.experiments_save_path,
        )

    def _initialize_cube_settings(self, separate_thread: bool) -> List[Dict]:
        return [
            self._one_cube_description(
                num_iter=_ONE_CUBE_NUM_ITERATIONS,
                separate_thread=separate_thread,
            )
            for _ in range(_NUM_CUBES)
        ]

    def _one_cube_description(self, num_iter: int, separate_thread: bool) -> dict:
        return {
            'RegularizersModifierCube':
            {
                'num_iter': num_iter,
                'regularizer_parameters':
                {
                    "regularizer": artm.regularizers.SmoothSparsePhiRegularizer(
                        name=_REGULARIZER_NAME
                    ),
                    "tau_grid": _TAU_GRID,
                },
                'reg_search': 'grid',
                'use_relative_coefficients': False,
                'separate_thread': separate_thread,
            },
            'selection': [_SELECT_CRITERION_FOR_ALL_MODELS]
        }

    def _check_result(
            self,
            cube_settings: List[Dict],
            experiment: Experiment,
            models: List[TopicModel]) -> None:

        assert experiment.depth > 0
        assert len(models) == len(_TAU_GRID) ** (experiment.depth - 1)
        assert experiment.depth == len(experiment.cubes)
        assert experiment.depth == len(cube_settings) + 1

        assert len(experiment.get_models_by_depth(0)) == 0
        assert len(experiment.get_models_by_depth(1)) == 1
        assert len(experiment.get_models_by_depth(2)) == 1 * len(_TAU_GRID)

        for d in range(3, experiment.depth + 1):
            assert len(experiment.get_models_by_depth(d)) == len(_TAU_GRID) ** (d - 1)

        assert len(experiment.models) == sum(
            len(_TAU_GRID) ** (d - 1) for d in range(1, experiment.depth + 1)
        )

    def _print_debug_info(self, experiment: Experiment) -> None:
        if not _DEBUG_MODE:
            return

        print(f'Experiment save path: {experiment.save_path}')
        print(f'Experiment depth: {experiment.depth}')
        print(f'Num cubes: {len(experiment.cubes)}')

        print('Cubes:' + '\n')

        for c in experiment.cubes:
            print(c)
            print()

        last_model = list(experiment.models.values())[-1]
        score_names = last_model.scores.keys()

        print(score_names)

        if _SCORE_NAME in score_names:
            print(last_model.scores[_SCORE_NAME])
