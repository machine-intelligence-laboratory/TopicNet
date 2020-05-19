import gzip
import os
import pandas as pd
import shutil
import ssl
import sys
import urllib

from glob import glob
from tqdm import tqdm
from urllib.request import (
    Request,
    urlopen,
)


from ..cooking_machine.dataset import Dataset


_SERVER_URL = 'https://93.175.29.159:8085'
_ARCHIVE_EXTENSION = '.gz'
_DEFAULT_DATASET_FILE_EXTENSION = '.csv'


def get_info() -> str:
    """
    Gets info about all datasets.

    Returns
    -------
    str with MarkDown syntax

    Examples
    --------
    As the return value is MarkDown text,
    in Jupyter Notebook one may do the following
    to format the output information nicely

    >>> from IPython.display import Markdown
    ...
    >>> Markdown(get_info())

    """
    req = Request(_SERVER_URL + '/info')
    context = ssl._create_unverified_context()

    with urlopen(req, context=context) as response:
        return response.read().decode('utf-8')


def load_dataset(dataset_name: str, **kwargs) -> Dataset:
    """
    Load dataset by dataset_name.
    Run ``get_info()`` to get dataset information

    Parameters
    ----------
    dataset_name: str
        dataset name for download

    Another Parameters
    ------------------
    kwargs
        optional properties of
        :class:`~topicnet.cooking_machine.Dataset`

    """
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)

    try:
        saved_dataset = _init_dataset_if_downloaded(dataset_path, **kwargs)
    except FileNotFoundError:
        pass
    else:
        print(
            f'Dataset already downloaded!'
            f' Save path is: "{saved_dataset._data_path}"'
        )

        return saved_dataset

    req = Request(_SERVER_URL + '/download')

    context = ssl._create_unverified_context()
    values = {'dataset-name': dataset_name}
    data = urllib.parse.urlencode(values).encode("utf-8")

    print(f'Downloading the "{dataset_name}" dataset...')

    try:
        with urlopen(req, data=data, context=context) as answer:
            total_size = int(answer.headers.get('content-length', 0))
            block_size = 1024
            save_path = dataset_path + answer.getheader('file-extension')

            t = tqdm(total=total_size, unit='iB', unit_scale=True, file=sys.stdout)

            with open(save_path + _ARCHIVE_EXTENSION, 'wb') as f:
                while True:
                    chunk = answer.read(block_size)

                    if not chunk:
                        break

                    t.update(len(chunk))
                    f.write(chunk)

            t.close()

            if total_size != 0 and t.n != total_size:
                raise RuntimeError(
                    "Failed to download dataset!"
                    " Some data was lost during network transfer"
                )

            with gzip.open(save_path + _ARCHIVE_EXTENSION, 'rb') as file_in, open(save_path, 'wb') as file_out:  # noqa E501
                # more memory-efficient than plain file_in.read()
                shutil.copyfileobj(file_in, file_out)

            print(f'Dataset downloaded! Save path is: "{save_path}"')

            return Dataset(save_path, **kwargs)

    except Exception as exception:
        if os.path.isfile(save_path):
            os.remove(save_path)

        raise exception

    finally:
        if os.path.isfile(save_path + _ARCHIVE_EXTENSION):
            os.remove(save_path + _ARCHIVE_EXTENSION)


def _init_dataset_if_downloaded(dataset_path: str, **kwargs) -> Dataset:
    saved_dataset_path_candidates = [
        p for p in glob(dataset_path + '*')
        if os.path.isfile(p) and not p.endswith(_ARCHIVE_EXTENSION)
    ]
    dataset = None

    if len(saved_dataset_path_candidates) > 0:
        saved_dataset_path = saved_dataset_path_candidates[0]

        try:
            dataset = Dataset(saved_dataset_path, **kwargs)
        except pd.errors.EmptyDataError:
            os.remove(saved_dataset_path)

    if dataset is None:
        raise FileNotFoundError()

    return dataset
