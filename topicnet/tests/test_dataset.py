import pytest
import shutil
import pandas as pd

from glob import glob
from ..cooking_machine.dataset import BaseDataset
from ..cooking_machine.dataset import Dataset


DATA_PATH = glob('tests/test_data/test_*.*')
BAD_DATA_PATH = glob('tests/test_data/wrong_*')
NONEXISTENT_DATA_PATH = 'tests/test_data/no_file.txt'
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
        cls.files = ['doc_1', 'doc_9']
        cls.nonexistent_files = ['doc_1a', 'doc_9b', 'doc_none', 'doc_all']

    def teardown_method(self):
        """ """
        for path in DATA_PATH:
            try:
                dataset = Dataset(path)
                shutil.rmtree(dataset._internals_folder_path)
            except(FileNotFoundError):
                continue

    @pytest.mark.parametrize("small", KEEP_DATA)
    def test_get_dict(self, small):
        """ """
        dataset = Dataset(self.dataset_path, keep_in_memory=small)

        with pytest.warns(None) as record:
            dataset.get_dictionary()

        assert len(record) == 0

    @pytest.mark.parametrize("small", KEEP_DATA)
    def test_get_dict_two_times(self, small):
        """ """
        dataset = Dataset(self.dataset_path, keep_in_memory=small)
        dataset.get_batch_vectorizer()

        dataset = Dataset(self.dataset_path, keep_in_memory=small)

        with pytest.warns(None) as record:
            dataset.get_dictionary()

        assert len(record) == 0

    @pytest.mark.xfail
    @pytest.mark.parametrize("small", KEEP_DATA)
    def test_get_dict_two_times_alternating(self, small):
        """ """
        dataset = Dataset(self.dataset_path, keep_in_memory=small)
        dataset.get_batch_vectorizer()

        dataset = Dataset(self.dataset_path, keep_in_memory=not small)

        with pytest.warns(None) as record:
            dataset.get_dictionary()

        assert len(record) == 0

    @pytest.mark.parametrize("small", KEEP_DATA)
    def test_change_dict(self, small):
        """ """
        dataset = Dataset(self.dataset_path, keep_in_memory=small)

        dictionary = dataset.get_dictionary()
        original_num_entries = Dataset._get_dictionary_num_entries(dictionary)

        dictionary.filter(max_df_rate=0.0)
        changed_num_entries = Dataset._get_dictionary_num_entries(dictionary)

        assert original_num_entries > changed_num_entries

        dictionary = dataset.get_dictionary()
        second_time_num_entries = Dataset._get_dictionary_num_entries(dictionary)

        assert second_time_num_entries == original_num_entries

    @pytest.mark.parametrize("path", BAD_DATA_PATH)
    def test_read_wrong_data(self, path):
        """ """
        if '.' in path:
            with pytest.raises(ValueError):
                _ = Dataset(path)
        else:
            with pytest.raises(TypeError):
                _ = Dataset(path)

    def test_read_nonexistent_data(self):
        """ """
        with pytest.raises(FileNotFoundError):
            _ = Dataset(NONEXISTENT_DATA_PATH)

    @pytest.mark.parametrize("path", DATA_PATH)
    def test_read_data(self, path):
        """ """
        _ = Dataset(path)

    @pytest.mark.parametrize("small", KEEP_DATA)
    def test_fail_on_absent_id(self, small):
        """ """
        dataset = Dataset(self.dataset_path, keep_in_memory=small)

        with pytest.raises(KeyError):
            dataset.get_source_document(self.nonexistent_files)
        with pytest.raises(KeyError):
            dataset.get_vw_document(self.nonexistent_files)
        with pytest.raises(KeyError):
            dataset.get_source_document(self.nonexistent_files[:3])
        with pytest.raises(KeyError):
            dataset.get_vw_document(self.nonexistent_files[:3])

    @pytest.mark.parametrize("small", KEEP_DATA)
    def test_return_data_both_cases(self, small):
        """ """
        dataset = Dataset(self.dataset_path, keep_in_memory=small)
        source_raw = dataset.get_source_document(self.files)
        source_vw = dataset.get_vw_document(self.files)

        assert isinstance(source_raw, pd.DataFrame)
        assert isinstance(source_vw, pd.DataFrame)
        assert len(self.files) == len(source_raw)
        assert len(self.files) == len(source_vw)

        for index, data in source_raw.iterrows():
            assert isinstance(data['raw_text'], str)
        for index, data in source_vw.iterrows():
            assert isinstance(data['vw_text'], str)
