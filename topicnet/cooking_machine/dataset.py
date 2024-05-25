import csv
import os
import pandas as pd
import shutil
import sys
import warnings

from glob import glob
from typing import (
    List,
    Optional,
)
from collections import Counter

import artm

from .routine import blake2bchecksum

VW_TEXT_COL = 'vw_text'
RAW_TEXT_COL = 'raw_text'

W_DIFF_BATCHES_1 = "Attempted to use batches for different dataset."
W_DIFF_BATCHES_2 = "Overwriting batches in {0}"
ERROR_NO_DATA_ENTRY = 'Requested documents with ids: {0} not found in the dataset'

DEFAULT_ARTM_MODALITY = '@default_class'  # TODO: how to get this value from artm library?
MODALITY_START_SYMBOL = '|'


def _increase_csv_field_max_size():
    """Makes document entry in dataset as big as possible

    References
    ----------
    https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072

    """
    max_int = sys.maxsize

    while True:
        try:
            csv.field_size_limit(max_int)

            break

        except OverflowError:
            max_int = int(max_int / 10)


def get_modality_names(vw_string):
    """
    Gets modality names from vw_string.

    Parameters
    ----------
    vw_string : str
        string in vw format

    Returns
    -------
    str
        document id
    list of str
        modalities in document

    """
    modalities = vw_string.split(MODALITY_START_SYMBOL)
    modality_names = [mod.split(' ')[0] for mod in modalities]
    doc_id = modality_names[0]
    modality_names = list(set(modality_names[1:]))

    return doc_id, modality_names


def get_modality_vw(vw_string, modality_name):
    """
    Gets modality string from document vw string.

    Parameters
    ----------
    vw_string : str
        string in vw format
    modality_name : str
        name of the modality

    Returns
    -------
    str
        content of modality_name modality

    """
    modality_contents = vw_string.split(MODALITY_START_SYMBOL)

    for one_modality_content in modality_contents:
        if one_modality_content[:len(modality_name)] == modality_name:
            return one_modality_content[len(modality_name):]

    return ""


def dataset2counter(dataset):
    result = {}
    for i, row in dataset._data.iterrows():
        doc_id, *text_info = row['vw_text'].split('|@')
        doc_id = doc_id.strip()
        result[doc_id] = Counter()
        # TODO: use get_content_of_modalty here
        vw_line = text_info[0]
        for token_with_counter in vw_line.split()[1:]:
            token, _, counter = token_with_counter.partition(':')
            result[doc_id][token] += int(counter or '1')
    return result


class BaseDataset:
    """ """
    def get_source_document(self, document_id):
        """

        Parameters
        ----------
        document_id : str

        """
        raise NotImplementedError

    def _transform_data_for_training(self):
        """ """
        raise NotImplementedError


class Dataset(BaseDataset):
    """
    Class for keeping training data and documents for creation models.

    """
    _internals_folder_name_suffix = 'internals'
    _dictionary_name = 'dict.dict'
    _vowpal_wabbit_file_name = 'vw.txt'
    _batches_folder_name = 'batches'

    def __init__(self,
                 data_path: str,
                 keep_in_memory: bool = True,
                 batch_vectorizer_path: str = None,
                 internals_folder_path: str = None,
                 batch_size: int = 1000):
        """
        Parameters
        ----------
        data_path : str
            path to a .csv file with input data for training models;
            file should have the following columns: id, raw_text, vw_text:

            * id (str) — document identificator
            * raw_text (str) — raw document text (maybe preprocessed somehow)
            * vw_text (str) — vowpal wabbit text (with modalities; either in bag-of-words format
                with specified word frequencies or in natural order)

            For an example, one may look at the test dataset here:
            topicnet/tests/test_data/test_dataset.csv
        keep_in_memory: bool
            flag determining if the collection is small enough to
            be kept in memory.
        batch_vectorizer_path : str
            path to the directory with collection batches
        internals_folder_path : str
            path to the directory with dataset internals, which includes:

            * vowpal wabbit file
            * dictionary file
            * batches directory

            The parameter is optional:
            the folder will be created by the dataset if not specified.
            This is a part of Dataset internal functioning.
            When working with any text collection `data_path` for the first time,
            there is no such folder: it will be created by Dataset.
        batch_size : int
            number of documents in one batch

        Warnings
        --------
        This class contains method to determine dataset modalities which
        relies on BigARTM library methods to work efficiently.
        However, we strongly advice against using modality name as is
        in `DEFAULT_ARTM_MODALITY` variable (currently `@default_class`)
        because it could cause incorrect behaviour from other parts of the library.

        It is also not recommended to use such symbols as comma ','
        and newline character '\\n' in `raw_text` and `vw_text` columns of ones dataset.
        This is because datasets are stored as .csv files which are to be read
        by `pandas` or `dask.dataframe` libraries.
        Mentioned symbols have special meaning for .csv file format,
        and, if used in plain text, may lead to errors.

        Notes
        -----
        Default way of training models in TopicNet is using :func:`artm.ARTM.fit_offline()`.
        However, if a dataset is really big
        (when `keep_in_memory` should definitely be set `False`),
        model training with big `num_iterations` may take a lot of time.
        ARTM library has another fit method for such cases: :func:`artm.ARTM.fit_online()`.
        It is worth trying to use exactly this method when working with huge document collections
        or collections which grow dynamically over time.
        However, as was mentioned,
        TopicNet is currently using only :func:`artm.ARTM.fit_offline()` under the hood.

        Below are some links,
        where one can fine some information about :func:`artm.ARTM.fit_online()`:

        * `RU text 1
        <http://www.machinelearning.ru/wiki/images/f/fb/Voron-ML-TopicModels.pdf>`_
        * `RU text 2
        <http://www.machinelearning.ru/wiki/index.php?title=ARTM>`_
        * `Documentation
        <bigartm.readthedocs.io/en/stable/api_references/python_interface/artm_model.html>`_

        It is also worth emphasizing that, if the text collection is big,
        `Theta` matrix may not fit in memory.
        So, in this case, some BigARTM scores (which depend on `Theta`) will stop working.
        """
        self._data_path = data_path
        self._small_data = keep_in_memory

        # If not do so, some really long documents may be lost/or error may be raised
        _increase_csv_field_max_size()

        self._data_hash = None

        self._dictionary: Optional[artm.Dictionary] = None
        self._dictionary_num_entries: Optional[int] = None

        if os.path.exists(data_path):
            self._data = self._read_data(data_path)
        else:
            raise FileNotFoundError('File {!r} doesn\'t exist'.format(data_path))

        if batch_vectorizer_path is not None:
            warnings.warn(
                'Parameter name `batch_vectorizer_path` is obsolete,'
                ' use `internals_folder_path` instead'
            )

            self._internals_folder_path = batch_vectorizer_path

            os.makedirs(self._batches_folder_path, exist_ok=True)

            for batch_file_path in glob(os.path.join(self._internals_folder_path, '*.batch')):
                shutil.move(batch_file_path, self._batches_folder_path)

        elif internals_folder_path is not None:
            self._internals_folder_path = internals_folder_path

        else:
            data_file_name = os.path.splitext(os.path.basename(self._data_path))[0]

            self._internals_folder_path = os.path.join(
                os.path.dirname(self._data_path),
                f'{data_file_name}__{self._internals_folder_name_suffix}',
            )
        self.batch_size = batch_size
        self.get_batch_vectorizer()
        self._modalities = self._extract_possible_modalities()

        if self._small_data:
            self._data_index = self._data.index
        else:
            self._data_index = self._data.index.compute()

    @property
    def documents(self) -> List[str]:
        return list(self._data_index)

    @property
    def _batch_vectorizer_path(self) -> str:
        warnings.warn(
            'Field `_batch_vectorizer_path` is obsolete,'
            ' use `_batches_folder_path` instead as path to batches folder'
            ' and `_internals_folder_path` as path to base dataset folder'
            ' (where there is also the batches folder)'
        )

        return self._batches_folder_path

    @property
    def _dictionary_file_path(self) -> str:
        return os.path.join(self._internals_folder_path, self._dictionary_name)

    @property
    def _vowpal_wabbit_file_path(self) -> str:
        return os.path.join(self._internals_folder_path, self._vowpal_wabbit_file_name)

    @property
    def _batches_folder_path(self) -> str:
        return os.path.join(self._internals_folder_path, self._batches_folder_name)

    @property
    def _cached_dict(self) -> Optional[artm.Dictionary]:
        if self._dictionary is None:
            return None

        if self._get_dictionary_num_entries(self._dictionary) != self._dictionary_num_entries:
            self._dictionary = None

        return self._dictionary

    @_cached_dict.setter
    def _cached_dict(self, dictionary: artm.Dictionary) -> None:
        self._dictionary = dictionary
        self._dictionary_num_entries = self._get_dictionary_num_entries(dictionary)

    @staticmethod
    def _get_dictionary_num_entries(dictionary: artm.Dictionary) -> int:
        """

        Notes
        -----
        See `__repr__`
        https://github.com/bigartm/bigartm/blob/master/python/artm/dictionary.py

        """
        description = next(
            x for x in dictionary._master.get_info().dictionary
            if x.name == dictionary.name
        )
        return description.num_entries

    def _read_data(self, data_path):
        """

        Parameters
        ----------
        data_path : str

        Returns
        -------
        pd.DataFrame
            data from data_path

        """
        _, file_type = os.path.splitext(data_path)

        if len(file_type) == 0:
            raise TypeError(f'Can\'t define file type: "{data_path}"')

        if self._small_data:
            import pandas as data_handle
        else:
            import dask.dataframe as data_handle

        if file_type == '.csv':
            data = data_handle.read_csv(
                data_path,
                engine='python',
                on_bad_lines='warn',
            )

        elif file_type == '.pkl':
            try:
                data = data_handle.read_pickle(
                    data_path,
                    engine='python',
                    on_bad_lines='warn',
                )
            except AttributeError:
                raise RuntimeError('Can\'t handle big *.pkl files!')

        elif file_type == '.txt' or file_type == '.vw':
            data = data_handle.read_csv(
                data_path,
                engine='python',
                on_bad_lines='warn',
                sep='HELLO_WORLD!',
                header=None,
                names=[VW_TEXT_COL]
            )

            data[RAW_TEXT_COL] = ''
            data['id'] = data[VW_TEXT_COL].str.partition(' ')[0]

        else:
            raise TypeError('Unknown file type')

        if VW_TEXT_COL not in data.columns:
            raise ValueError('data should contain VW field')

        data['id'] = data['id'].astype('str')
        data = data.set_index('id', drop=False)

        return data

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        save_dataset_path: str,
        dataframe_name: str = 'dataset',
        **kwargs
    ) -> 'Dataset':
        """
        Creates dataset from pd.DataFrame
        reuqires to specify technical folder for dataset files

        Parameters
        ----------
        dataset
            pandas DataFrame dataset
        save_dataset_path
            a folder where to store data.csv of your DataFrame
        dataframe_name:
            name for the dataset file to be saved in csv format
        Another Parameters
        ------------------
        **kwargs
            *kwargs* are optional init `topicnet.Dataset` parameters
        """
        data_path = os.path.join(save_dataset_path, dataframe_name + '.csv')
        dataframe.to_csv(data_path)

        return cls(data_path=data_path, **kwargs)

    def get_dataset(self):
        """ """
        return self._data

    def _prepare_no_entry_error_message(self, document_id, in_index):
        missing_ids = [
                    doc_id
                    for doc_id in document_id
                    if doc_id not in in_index
                ]
        if len(missing_ids) > 3:
            missing_ids = ', '.join(missing_ids[:3]) + ', ...'
        else:
            missing_ids = ', '.join(missing_ids[:3])
        return ERROR_NO_DATA_ENTRY.format(missing_ids)

    def get_vw_document(self, document_id: str or List[str]) -> pd.DataFrame:
        """
        Get 'vw_text' for the document with `document_id`.

        Parameters
        ----------
        document_id
            document name or list of document names

        Returns
        -------
        pd.DataFrame
            `document_id` and content of 'vw_text' column
        """
        if not isinstance(document_id, list):
            document_id = [document_id]
        if self._small_data:
            in_index = self._data.index.intersection(document_id)
            if len(in_index) < len(document_id):
                error_message = self._prepare_no_entry_error_message(
                    document_id,
                    in_index
                )
                raise KeyError(error_message)
            return pd.DataFrame(
                self._data.loc[in_index, VW_TEXT_COL]
                .reindex(document_id)
            )

        else:
            in_index = [
                doc_id for doc_id in document_id
                if doc_id in self._data_index
            ]
            if len(in_index) < len(document_id):
                error_message = self._prepare_no_entry_error_message(
                    document_id,
                    in_index
                )
                raise KeyError(error_message)
            return pd.DataFrame(
                self._data.loc[in_index, VW_TEXT_COL].compute()
                .reindex(document_id)
            )

    def get_source_document(self, document_id: str or List[str]) -> pd.DataFrame:
        """
        Get 'raw_text' for the document with `document_id`.

        Parameters
        ----------
        document_id
            document name or list of document names

        Returns
        -------
        pd.DataFrame
            `document_id` and content of 'raw_text' column
        """
        if not isinstance(document_id, list):
            document_id = [document_id]
        if self._small_data:
            in_index = self._data.index.intersection(document_id)
            if len(in_index) < len(document_id):
                error_message = self._prepare_no_entry_error_message(
                    document_id,
                    in_index
                )
                raise KeyError(error_message)
            return pd.DataFrame(
                self._data.loc[in_index, RAW_TEXT_COL]
                .reindex(document_id)
            )

        else:
            in_index = [
                doc_id for doc_id in document_id
                if doc_id in self._data_index
            ]
            if len(in_index) < len(document_id):
                error_message = self._prepare_no_entry_error_message(
                    document_id,
                    in_index
                )
                raise KeyError(error_message)
            return pd.DataFrame(
                self._data.loc[in_index, RAW_TEXT_COL].compute()
                .reindex(document_id)
            )

    def write_vw(self, file_path: str) -> None:
        """
        Saves dataset as text file in Vowpal Wabbit format

        """
        save_kwargs = {
            'header': False,
            'columns': [VW_TEXT_COL],
            'index': False,
            'sep': '\n',
            'quoting': csv.QUOTE_NONE,
            'quotechar': '',
        }
        if not self._small_data:
            save_kwargs['single_file'] = True
        try:
            self._data.to_csv(
                file_path,
                **save_kwargs
            )
        except csv.Error as e:
            raise RuntimeError(
                f'Failed to write Vowpal Wabbit file!'
                f' This might happen due to data containing'
                f' special symbol "\\n" that needed to be replaced.'
                f' Make sure that text values in {VW_TEXT_COL} column'
                f' do not contain new line symbols'
            ) from e

    def _check_collection(self):
        """
        Checks if folder with collection:
        1) Exists
        2) Same as the one this dataset holds

        Returns
        -------
        same_collection : bool
        """
        path_to_collection = self._vowpal_wabbit_file_path

        if not os.path.exists(self._internals_folder_path):
            os.mkdir(self._internals_folder_path)

            return False, path_to_collection

        if self._data_hash is None:
            temp_file_path = os.path.join(
                self._internals_folder_path, 'temp_vw.txt'
            )

            try:
                self.write_vw(temp_file_path)
                self._data_hash = blake2bchecksum(temp_file_path)
            finally:
                if os.path.isfile(temp_file_path):
                    os.remove(temp_file_path)

        if os.path.isfile(path_to_collection):
            same_collection = blake2bchecksum(path_to_collection) == self._data_hash
        else:
            same_collection = False

        return same_collection, path_to_collection

    def get_batch_vectorizer(self) -> artm.BatchVectorizer:
        """
        Gets batch vectorizer.

        Returns
        -------
        artm.BatchVectorizer

        """
        same_collection, path_to_collection = self._check_collection()

        if same_collection:
            batches_exist = len(glob(os.path.join(self._batches_folder_path, '*.batch'))) > 0

            if not batches_exist:
                self.write_vw(path_to_collection)

                return artm.BatchVectorizer(
                    data_path=path_to_collection,
                    data_format='vowpal_wabbit',
                    target_folder=self._batches_folder_path,
                    batch_size=self.batch_size
                )
            else:
                return artm.BatchVectorizer(
                    data_path=self._batches_folder_path,
                    data_format='batches'
                )

        if os.path.isdir(self._batches_folder_path):
            warnings.warn(W_DIFF_BATCHES_1 + W_DIFF_BATCHES_2.format(self._batches_folder_path))
            self.clear_batches_folder()

        self.write_vw(path_to_collection)

        return artm.BatchVectorizer(
            data_path=path_to_collection,
            data_format='vowpal_wabbit',
            target_folder=self._batches_folder_path,
            batch_size=self.batch_size
        )

    def get_dictionary(self) -> artm.Dictionary:
        """
        Gets dataset's dictionary.

        Returns
        -------
        artm.Dictionary

        """
        if self._cached_dict is not None:
            return self._cached_dict

        dictionary = artm.Dictionary()

        same_collection, path_to_collection = self._check_collection()

        if same_collection:
            if not os.path.isfile(self._dictionary_file_path):
                dictionary.gather(data_path=self._batches_folder_path)
                dictionary.save(dictionary_path=self._dictionary_file_path)

            dictionary.load(dictionary_path=self._dictionary_file_path)
            self._cached_dict = dictionary
        else:
            _ = self.get_batch_vectorizer()
            dictionary.gather(data_path=self._batches_folder_path)

            if os.path.isfile(self._dictionary_file_path):
                os.remove(self._dictionary_file_path)

            dictionary.save(dictionary_path=self._dictionary_file_path)
            dictionary.load(dictionary_path=self._dictionary_file_path)
            self._cached_dict = dictionary

        return self._cached_dict

    def _transform_data_for_training(self):
        """ """
        return self.get_batch_vectorizer()

    def _extract_possible_modalities(self):
        """
        Extracts all modalities from data.

        Returns
        -------
        set
            all modalities in Dataset

        """
        artm_dict = self.get_dictionary()
        modalities = set(artm_dict._master.get_dictionary(artm_dict._name).class_id)
        # ARTM fills modality name if none is present
        modalities.discard(DEFAULT_ARTM_MODALITY)
        return modalities

    def get_possible_modalities(self):
        """
        Returns extracted modalities.

        Returns
        -------
        set
            all modalities in Dataset

        """
        return self._modalities

    def clear_folder(self):
        """
        Clear internals_folder_path
        """
        if not os.path.isdir(self._internals_folder_path):
            print(f'Failed to delete non-existent folder: {self._internals_folder_path}')
        else:
            shutil.rmtree(self._internals_folder_path)
            os.makedirs(self._internals_folder_path)
            os.makedirs(self._batches_folder_path)

    def clear_batches_folder(self):
        """
        Clear batches folder
        """
        if not os.path.isdir(self._batches_folder_path):
            print(f'Failed to delete non-existent folder: {self._batches_folder_path}')
        else:
            shutil.rmtree(self._batches_folder_path)
            os.makedirs(self._batches_folder_path)
