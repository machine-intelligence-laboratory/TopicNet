from flask import Flask, send_from_directory, abort

import os
import shutil


DATA_PATH = '/data_mil/public_models/'
app = Flask(__name__)


@app.route('/download/<model_name>', methods=['GET', 'POST'])
def download_file(model_name):
    source_dir = os.path.join(DATA_PATH, model_name)
    base_file_name = "_" + model_name
    source_archive = os.path.join(DATA_PATH, base_file_name + '.tar.gz')

    if not os.path.isfile(source_archive):
        if not os.path.isdir(source_dir):
            abort(404)
        shutil.make_archive(
            os.path.join(DATA_PATH, base_file_name),
            'gztar',
            source_dir
        )

    response = send_from_directory(
        directory=DATA_PATH,
        path=base_file_name + '.tar.gz',
        cache_timeout=0
    )
    response.headers['file-extension'] = os.path.splitext(source_archive)[-1]
    return response


if __name__ == '__main__':
    app.run(port='9001')
