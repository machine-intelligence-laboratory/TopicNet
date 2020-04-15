from distutils.core import setup


setup(
    name = 'topicnet',
    packages = [
        'topicnet',
        'topicnet.cooking_machine',
        'topicnet.cooking_machine.cubes',
        'topicnet.cooking_machine.models',
        'topicnet.cooking_machine.recipes',
        'topicnet.dataset_manager',
        'topicnet.viewers'
    ],
    version = '0.7.0',
    license='MIT',
    description = 'TopicNet is a module for topic modelling using ARTM algorithm',
    author = 'Machine Intelligence Laboratory',
    author_email = 'alex.goncharov@phystech.edu',
    url = 'https://github.com/machine-intelligence-laboratory/TopicNet',
    download_url = 'https://github.com/machine-intelligence-laboratory/TopicNet/archive/v0.7.0.tar.gz', 
    keywords = [
        'ARTM',
        'topic modeling',
        'regularization',
        'multimodal learning',
        'document vector representation'
    ],
    install_requires=[
        'bigartm',
        'colorlover',
        'dask[dataframe]',
        'dill',
        'ipython',
        'jinja2',
        'numexpr',
        'numpy',
        'pandas',
        'plotly',
        'pytest',
        'scikit-learn',
        'scipy',
        'six',
        'strictyaml',
        'toolz',
        'tqdm',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
