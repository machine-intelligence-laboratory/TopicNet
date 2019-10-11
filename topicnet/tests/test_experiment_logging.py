import artm
import shutil
import pytest
import warnings

from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset, W_DIFF_BATCHES_1
from ..cooking_machine.model_constructor import init_simple_default_model
from ..cooking_machine.cubes import RegularizersModifierCube
from ..cooking_machine.cubes.perplexity_strategy import PerplexityStrategy

MAIN_MODALITY = "@text"


class TestLogging:
    @classmethod
    def setup_class(cls):
        """ """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message=W_DIFF_BATCHES_1)
            cls.experiment_path = 'tests/test_data/test_experiment/'
            cls.dataset = Dataset('tests/test_data/test_dataset.csv')
            cls.model_artm = init_simple_default_model(
                dataset=cls.dataset,
                modalities_to_use={MAIN_MODALITY},
                main_modality=MAIN_MODALITY,
                specific_topics=14,
                background_topics=1,
            )
            cls.topic_model = TopicModel(cls.model_artm, model_id='Groot')

    @classmethod
    def teardown_class(cls):
        """ """
        shutil.rmtree('tests/test_data/test_dataset_batches')
        shutil.rmtree(cls.experiment_path)

    def test_experiment_exists(cls):
        """ """
        experiment = Experiment(
            cls.topic_model,
            experiment_id="rewrite_experiment",
            save_path=cls.experiment_path,
        )
        with pytest.raises(FileExistsError, match="already exists"):
            tm = TopicModel(cls.model_artm, model_id='Groot')
            experiment = Experiment(  # noqa: F841
                tm,
                experiment_id="rewrite_experiment",
                save_path=cls.experiment_path,
            )

    def test_experiment_prune(cls):
        """ """
        cls.topic_model.experiment = None
        experiment_run = Experiment(
            cls.topic_model,
            experiment_id="run_experiment",
            save_path=cls.experiment_path,
            )
        test_cube = RegularizersModifierCube(
            num_iter=5,
            regularizer_parameters={
                'regularizer': artm.DecorrelatorPhiRegularizer(name='decorrelation_phi', tau=1),
                'tau_grid': [],
            },
            strategy=PerplexityStrategy(0.001, 10, 25, threshold=1.0),
            tracked_score_function='PerplexityScore@all',
            reg_search='mul',
            relative_coefficients=False,
            verbose=True
        )

        test_cube(cls.topic_model, cls.dataset)
        experiment_run.set_criteria(1, 'some_criterion')

        new_seed = experiment_run.get_models_by_depth(level=1)[0]
        experiment = Experiment(
            topic_model=new_seed,
            experiment_id="prune_experiment",
            save_path=cls.experiment_path,
            save_model_history=True,
            )
        assert len(experiment.models) == 1

    def test_work_with_dataset(cls):
        """ """
        cls.topic_model.experiment = None
        experiment = Experiment(
            cls.topic_model,
            experiment_id="dataset_experiment",
            save_path=cls.experiment_path,
            )
        experiment.add_dataset('dataset', cls.dataset)
        with pytest.raises(NameError, match=r"Dataset with name *"):
            experiment.add_dataset('dataset', cls.dataset)

        experiment.remove_dataset('dataset')
        with pytest.raises(NameError, match=r"There is no dataset *"):
            experiment.remove_dataset('dataset')
