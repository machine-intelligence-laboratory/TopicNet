import artm
import shutil
import pytest

from ..cooking_machine.models.topic_model import TopicModel
from ..cooking_machine.experiment import Experiment
from ..cooking_machine.dataset import Dataset
from ..cooking_machine.model_constructor import init_simple_default_model
from ..cooking_machine.cubes import RegularizersModifierCube


class TestLogging:
    @classmethod
    def setup_class(cls):
        """ """
        cls.experiment_path = 'tests/test_data/test_experiment/'
        cls.dataset = Dataset('tests/test_data/test_dataset.csv')
        dictionary = cls.dataset.get_dictionary()
        model_artm = init_simple_default_model(
            dictionary=dictionary,
            modalities_to_use={'@text'},
            main_modality='@text',
            n_specific_topics=14,
            n_background_topics=1,
        )
        cls.topic_model = TopicModel(model_artm, model_id='Groot')

    @classmethod
    def teardown_class(cls):
        """ """
        shutil.rmtree('tests/test_data/test_dataset_batches')
        shutil.rmtree(cls.experiment_path)

    def test_experiment_exists(cls):
        """ """
        experiment = Experiment(
            experiment_id="rewrite_experiment",
            save_path=cls.experiment_path,
        )
        with pytest.raises(FileExistsError, match="already exists"):
            experiment = Experiment(  # noqa: F841
                experiment_id="rewrite_experiment",
                save_path=cls.experiment_path,
            )

    def test_experiment_prune(cls):
        """ """
        experiment_run = Experiment(
            topic_model=cls.topic_model,
            experiment_id="run_experiment",
            save_path=cls.experiment_path,
            )
        test_cube = RegularizersModifierCube(
            num_iter=3,
            regularizer_parameters={
                'regularizer': artm.DecorrelatorPhiRegularizer(name='decorrelation_phi', tau=1),
                'tau_grid': [1, 10, 100, 1000],
            },
            reg_search='mul',
            verbose=True
        )
        test_cube(cls.topic_model, cls.dataset)

        new_seed = experiment_run.get_models_by_depth()[0]
        experiment = Experiment(
            topic_model=new_seed,
            experiment_id="prune_experiment",
            save_path=cls.experiment_path,
            save_model_history=True,
            )
        assert len(experiment.models) == 2

    def test_work_with_dataset(cls):
        """ """
        experiment = Experiment(
            experiment_id="dataset_experiment",
            save_path=cls.experiment_path,
            )
        experiment.add_dataset('dataset', cls.dataset)
        with pytest.raises(NameError, match=r"Dataset with name *"):
            experiment.add_dataset('dataset', cls.dataset)

        experiment.remove_dataset('dataset')
        with pytest.raises(NameError, match=r"There is no dataset *"):
            experiment.remove_dataset('dataset')
