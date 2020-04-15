import urllib
from urllib.request import Request, urlopen
import ssl

import os
from glob import glob
from tqdm import tqdm
import gzip

from ..cooking_machine.dataset import Dataset


URL = 'https://93.175.29.159:8085'


def get_info() -> str:
    """
    Gets info about all datasets.

    Returns
    -------
    str with MarkDown syntax

    """
    req = Request(URL + '/info')
    context = ssl._create_unverified_context()
    with urlopen(req, context=context) as response:
        return response.read().decode('utf-8')


def load_dataset(dataset_name: str, internals_folder_path: str = None) -> Dataset:
    """
    Load dataset by dataset_name. Run get_info() to get dataset information

    Parameters
    ----------
    dataset_name: str
        dataset name for download

    internals_folder_path: str (optional)
        path to the directory with dataset internals
        for topicnet.cooking_machine.dataset.Dataset initialize

    Returns
    -------
    topicnet.cooking_machine.dataset.Dataset

    """
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_name)

    if glob(dataset_path + '*'):
        return Dataset(glob(dataset_path + '*')[0], internals_folder_path=internals_folder_path)

    req = Request(URL + '/download')

    context = ssl._create_unverified_context()
    values = {'dataset-name': dataset_name}
    data = urllib.parse.urlencode(values).encode("utf-8")

    with urlopen(req, data=data, context=context) as answer:
        total_size = int(answer.headers.get('content-length', 0))
        block_size = 1024
        save_path = dataset_path + answer.getheader('file-extension')

        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(save_path + '.gz', 'wb') as f:
            while True:
                chunk = answer.read(block_size)
                if not chunk:
                    break
                t.update(len(chunk))
                f.write(chunk)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("Failed to download file")
            return None
        else:
            with gzip.open(save_path + '.gz', 'rb') as gz:
                with open(save_path, 'wb') as f:
                    f.write(gz.read())
            os.remove(save_path + '.gz')
            return Dataset(save_path, internals_folder_path=internals_folder_path)
