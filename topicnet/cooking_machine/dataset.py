import os
import sys
import csv
import artm
import shutil
import warnings

import pandas as pd
from glob import glob
from .routine import blake2bchecksum

W_DIFF_BATCHES_1 = "Attempted to use batches for different dataset."
W_DIFF_BATCHES_2 = "Overwriting batches in {0}"

DEFAULT_ARTM_MODALITY = '@default_class'  # TODO: how to get this value from artm library?
MODALITY_START_SYMBOL = '|'

# change log style
lc = artm.messages.ConfigureLoggingArgs()
lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)


def _fix_max_string_size():
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)


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


VW_TEXT_COL = 'vw_text'
RAW_TEXT_COL = 'raw_text'


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
    def __init__(self, data_path,
                 keep_in_memory=True,
                 batch_vectorizer_path=None,
                 batch_size=1000):
        """

        Parameters
        ----------
        data_path : str
            path to a CSV file with input data for training models,
            format of the file is <id>,<raw text>,<Vowpal Wabbit text>
        keep_in_memory: bool
            a flag determining if the collection is small enough to
            be kept in memory.
        batch_vectorizer_path : str
            path to the directory with collection batches
        batch_size : int
            number of documents in one batch

        """
        # set main data
        self._data_path = data_path
        self._small_data = keep_in_memory
        # making document entry as big as possible
        _fix_max_string_size()
        self._data_hash = None
        self._cached_dict = None
        if os.path.exists(data_path):
            self._data = self._read_data(data_path)
        else:
            raise ValueError('File {!r} doesn\'t exist'.format(data_path))

        # set batch vectorizer path
        if batch_vectorizer_path is not None:
            self._batch_vectorizer_path = batch_vectorizer_path
        else:
            dot_position = data_path.rfind('.')
            dot_position = dot_position if dot_position > 0 else len(data_path)
            self._batch_vectorizer_path = data_path[:dot_position] + '_batches'

        self._modalities = self._extract_possible_modalities()
        self.batch_size = batch_size

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
        dot_position = data_path.rfind('.')
        if dot_position == -1:
            raise TypeError('filename should contain . to define type')

        file_type = data_path[dot_position + 1:]
        if self._small_data:
            import pandas as data_handle
        else:
            import dask.dataframe as data_handle

        if file_type == 'csv':
            data = data_handle.read_csv(
                data_path,
                engine='python',
                error_bad_lines=False,
            )
        elif file_type == 'pkl':
            try:
                data = data_handle.read_pickle(
                    data_path,
                    engine='python',
                    error_bad_lines=False,
                )
            except AttributeError:
                raise('Cant handle big *.pkl files')
        elif file_type == 'txt' or file_type == 'vw':
            data = data_handle.read_csv(
                data_path,
                engine='python',
                error_bad_lines=False,
                sep='\n',
                header=None,
                names=[VW_TEXT_COL]
            )
            data[RAW_TEXT_COL] = ''

            data['id'] = data[VW_TEXT_COL].str.partition(' ')[0]
        else:
            raise TypeError('Unknown file type')
        data['id'] = data['id'].astype('str')
        data = data.set_index('id')
        if VW_TEXT_COL not in data.columns:
            raise ValueError('data should contain VW field')
        return data

    def get_dataset(self):
        """ """
        return self._data

    def get_vw_document(self, document_id):
        """
        Get 'vw_text' for the document with document_id.

        Parameters
        ----------
        document_id : str

        Returns
        -------
        list of str
            document id and content of 'vw_text' column

        """
        if not isinstance(document_id, list):
            document_id = [document_id]
        if self._small_data:
            in_index = self._data.index.intersection(document_id)
            back_pd = pd.DataFrame(
                self._data.loc[in_index, VW_TEXT_COL]
                .reindex(document_id)
                .fillna('Not Found')
            )
            return back_pd
        else:
            data_indices = self._data.index.compute()
            in_index = [
                doc_id for doc_id in document_id
                if doc_id in data_indices
            ]
            back_pd = pd.DataFrame(
                self._data.loc[in_index, VW_TEXT_COL].compute()
                .reindex(document_id)
                .fillna('Not Found')
            )
            return back_pd

    def get_source_document(self, document_id):
        """
        Get 'raw_text' for the document with document_id.

        Parameters
        ----------
        document_id : str

        Returns
        -------
        list of str
            document id and content of 'raw_text' column

        """
        if not isinstance(document_id, list):
            document_id = [document_id]
        if self._small_data:
            in_index = self._data.index.intersection(document_id)
            back_pd = pd.DataFrame(
                self._data.loc[in_index, RAW_TEXT_COL]
                .reindex(document_id)
                .fillna('Not Found')
            )
            return back_pd
        else:
            data_indices = self._data.index.compute()
            in_index = [
                doc_id for doc_id in document_id
                if doc_id in data_indices
            ]
            back_pd = pd.DataFrame(
                self._data.loc[in_index, RAW_TEXT_COL].compute()
                .reindex(document_id)
                .fillna('Not Found')
            )
            return back_pd

    def write_vw(self, file_path):
        """ """
        with open(file_path, 'w', encoding='utf-8') as f:
            for index, data in self._data.iterrows():
                vw_string = data[VW_TEXT_COL]
                f.write(vw_string + '\n')

    def _check_collection(self, data_path):
        """
        Checks if folder with collection:
        1) Exists
        2) Same as the one this dataset holds

        Parameters
        ----------
        data_path : str
            folder for the checking
        Returns
        -------
        same_collection : bool
        """
        path_to_collection = os.path.join(data_path, 'vw.txt')
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            return False, path_to_collection

        if self._data_hash is None:
            temp_file_path = os.path.join(data_path, 'temp_vw.txt')
            self.write_vw(temp_file_path)
            self._data_hash = blake2bchecksum(temp_file_path)
            os.remove(temp_file_path)

        if os.path.isfile(path_to_collection):
            same_collection = blake2bchecksum(path_to_collection) == self._data_hash
        else:
            same_collection = False
        return same_collection, path_to_collection

    def get_batch_vectorizer(self, batch_vectorizer_path=None):
        """
        Get batch vectorizer.

        Parameters
        ----------
        batch_vectorizer_path : str
             (Default value = None)

        Returns
        -------
        batch_vectorizer :

        """
        if batch_vectorizer_path is None:
            batch_vectorizer_path = self._batch_vectorizer_path

        same_collection, path_to_collection = self._check_collection(
            batch_vectorizer_path
        )

        if same_collection:
            batches_exist = len(glob(os.path.join(batch_vectorizer_path, '*.batch'))) > 0
            if not batches_exist:
                self.write_vw(path_to_collection)
                batch_vectorizer = artm.BatchVectorizer(
                    data_path=path_to_collection,
                    data_format='vowpal_wabbit',
                    target_folder=batch_vectorizer_path,
                    batch_size=self.batch_size
                )
            else:
                batch_vectorizer = artm.BatchVectorizer(
                    data_path=batch_vectorizer_path,
                    data_format='batches'
                )
        else:
            warnings.warn(W_DIFF_BATCHES_1 + W_DIFF_BATCHES_2.format(batch_vectorizer_path))
            try:
                shutil.rmtree(batch_vectorizer_path)
            except FileNotFoundError:
                pass
            os.mkdir(batch_vectorizer_path)
            self.write_vw(path_to_collection)

            batch_vectorizer = artm.BatchVectorizer(
                data_path=path_to_collection,
                data_format='vowpal_wabbit',
                target_folder=batch_vectorizer_path,
                batch_size=self.batch_size
            )

        return batch_vectorizer

    def get_dictionary(self, batch_vectorizer_path=None):
        """
        Get dictionary.

        Parameters
        ----------
        batch_vectorizer_path : str
             (Default value = None)

        Returns
        -------
        dictionary :

        """
        if self._cached_dict is not None:
            return self._cached_dict

        if batch_vectorizer_path is None:
            batch_vectorizer_path = self._batch_vectorizer_path

        dictionary = artm.Dictionary()
        dict_path = os.path.join(batch_vectorizer_path, 'dict.dict')

        same_collection, path_to_collection = self._check_collection(
            batch_vectorizer_path
        )

        if same_collection:
            if not os.path.isfile(dict_path):
                dictionary.gather(data_path=batch_vectorizer_path)
                dictionary.save(dictionary_path=dict_path)
            dictionary.load(dictionary_path=dict_path)
            self._cached_dict = dictionary
            return dictionary
        else:
            _ = self.get_batch_vectorizer(batch_vectorizer_path)
            dictionary.gather(data_path=batch_vectorizer_path)
            dictionary.save(dictionary_path=dict_path)
            dictionary.load(dictionary_path=dict_path)
            self._cached_dict = dictionary
            return dictionary

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
        modalities_list = [
            get_modality_names(vw_string[VW_TEXT_COL])[1]
            for _, vw_string in self._data.iterrows()
        ]
        all_modalities = set([
            modality
            for modalities in modalities_list
            for modality in modalities
        ])
        return all_modalities

    def get_possible_modalities(self):
        """
        Returns extracted modalities.

        Returns
        -------
        set
            all modalities in Dataset

        """
        return self._modalities
