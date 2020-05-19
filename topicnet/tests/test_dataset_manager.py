import glob
import os
import pytest
import time

import topicnet

from ..dataset_manager import load_dataset
from ..dataset_manager.api import (
    _ARCHIVE_EXTENSION,
    _DEFAULT_DATASET_FILE_EXTENSION,
)


class TestDatasetManager:
    dataset_manager_folder_path = os.path.join(
        os.path.dirname(topicnet.__file__),
        'dataset_manager',
    )

    @classmethod
    def teardown_class(cls):
        cls._clear_dataset_manager_folder()

    def setup_method(self):
        self._clear_dataset_manager_folder()

        assert not self._is_any_dataset_exists()

    @classmethod
    def _is_any_dataset_exists(cls) -> bool:
        csv_file_paths = glob.glob(
            os.path.join(
                cls.dataset_manager_folder_path,
                f'*{_DEFAULT_DATASET_FILE_EXTENSION}',
            )
        )

        return len(csv_file_paths) > 0

    @classmethod
    def _clear_dataset_manager_folder(cls) -> None:
        for file_name in os.listdir(cls.dataset_manager_folder_path):
            if (file_name.endswith(_DEFAULT_DATASET_FILE_EXTENSION) or
                    file_name.endswith(_ARCHIVE_EXTENSION)):

                os.remove(os.path.join(cls.dataset_manager_folder_path, file_name))

    @pytest.mark.parametrize('dataset_name', ['postnauka', '20NG'])
    def test_download_once_and_again(self, dataset_name):
        dataset = load_dataset(dataset_name)
        dataset.get_dictionary()
        dataset.get_batch_vectorizer()

        assert len(os.listdir(dataset._batches_folder_path)) > 0

        dataset.clear_folder()

        assert os.path.isfile(dataset._data_path)
        assert self._is_any_dataset_exists()

        dataset = load_dataset(dataset_name)
        dataset.get_dictionary()
        dataset.get_batch_vectorizer()

        assert len(os.listdir(dataset._batches_folder_path)) > 0

    @pytest.mark.parametrize('keep_in_memory', [True, False])
    def test_specify_dataset_param(self, keep_in_memory):
        dataset_name = 'postnauka'

        dataset = load_dataset(dataset_name, keep_in_memory=keep_in_memory)

        assert dataset._small_data == keep_in_memory

    def test_no_load_if_already_download(self):
        dataset_name = 'postnauka'

        dataset = load_dataset(dataset_name)
        first_load_time = os.path.getmtime(dataset._data_path)

        time.sleep(1)

        dataset = load_dataset(dataset_name)
        second_load_time = os.path.getmtime(dataset._data_path)

        assert second_load_time == first_load_time
