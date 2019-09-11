import pytest
import shutil
import pandas as pd

from glob import glob
from ..cooking_machine.dataset import BaseDataset
from ..cooking_machine.dataset import Dataset


DATA_PATH = glob('tests/test_data/test_*.*')
BAD_DATA_PATH = (glob('tests/test_data/wrong_*')
                 + ['tests/test_data/no_file.txt'])
KEEP_DATA = [True, False]


def test_base_dataset():
    """ """
    with pytest.raises(NotImplementedError):
        BaseDataset().get_source_document('1')
        BaseDataset()._transform_data_for_training()


class TestDataset:
    @classmethod
    def setup_class(cls):
        """ """
        cls.dataset_path = 'tests/test_data/test_dataset.csv'
        cls.files = ['doc_1a', 'doc_9b', 'doc_1', 'doc_9', 'doc_none']

    @classmethod
    def teardown_class(cls):
        """ """
        for path in DATA_PATH:
            try:
                shutil.rmtree(path.split('.')[0] + "_batches")
            except(FileNotFoundError):
                continue

    @pytest.mark.parametrize("small", KEEP_DATA)
    def test_get_dict(cls, small):
        """ """
        dataset = Dataset(cls.dataset_path, keep_in_memory=small)
        with pytest.warns(UserWarning):
            dataset.get_dictionary()

    @pytest.mark.parametrize("path", BAD_DATA_PATH)
    def test_read_wrong_data(cls, path):
        """ """
        if '.' in path:
            with pytest.raises(ValueError):
                _ = Dataset(path)
        else:
            with pytest.raises(TypeError):
                _ = Dataset(path)

    @pytest.mark.parametrize("path", DATA_PATH)
    def test_read_data(cls, path):
        """ """
        _ = Dataset(path)

    @pytest.mark.parametrize("small", KEEP_DATA)
    def test_return_data_both_cases(cls, small):
        """ """
        dataset = Dataset(cls.dataset_path, keep_in_memory=small)
        source_raw = dataset.get_source_document(cls.files)
        source_vw = dataset.get_vw_document(cls.files)
        assert isinstance(source_raw, pd.DataFrame)
        assert isinstance(source_vw, pd.DataFrame)
        assert len(cls.files) == len(source_raw)
        assert len(cls.files) == len(source_vw)
        for index, data in source_raw.iterrows():
            if len(index) == 5:
                assert data['raw_text'] != 'Not Found'
            else:
                assert data['raw_text'] == 'Not Found'
        for index, data in source_vw.iterrows():
            if len(index) == 5:
                assert data['vw_text'] != 'Not Found'
            else:
                assert data['vw_text'] == 'Not Found'
