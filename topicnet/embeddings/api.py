import os
import shutil
import ssl
import sys

from tqdm import tqdm
from urllib.request import (
    Request,
    urlopen,
)

from ..cooking_machine.models import TopicModel


_SERVER_URL = 'https://127.0.0.1:9001/'

_ARCHIVE_EXTENSION = '.tar.gz'


def load_model(model_name: str) -> TopicModel:
    """
    Load model by model_name.
    Run ``get_info()`` to get model information

    Parameters
    ----------
    model_name: str
        name of model to download

    """
    # model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    model_path = os.path.join('.', model_name)

    print(f'Checking if model "{model_name}" was already downloaded before')

    if os.path.isdir(model_path):
        return TopicModel.load(model_path)

    req = Request(_SERVER_URL + '/download/' + model_name)

    context = ssl._create_unverified_context()

    print(f'Downloading the "{model_name}" model...')

    save_path = None

    try:
        # with urlopen(req, data=data, context=context) as answer:
        with urlopen(req, context=context) as answer:
            total_size = int(answer.headers.get('content-length', 0))
            block_size = 1024
            save_path = model_path  # + answer.getheader('file-extension')

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
                    "Failed to download the model!"
                    " Some data was lost during network transfer"
                )

            shutil.unpack_archive(save_path + _ARCHIVE_EXTENSION, save_path)
            return TopicModel.load(save_path)

    except Exception as exception:
        if save_path is not None and os.path.isfile(save_path):
            os.remove(save_path)

        raise exception

    finally:
        if save_path is not None and os.path.isfile(save_path + _ARCHIVE_EXTENSION):
            os.remove(save_path + _ARCHIVE_EXTENSION)
