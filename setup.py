from distutils.core import setup


_README_CONTENT = open('README.md').read()

setup(
    name='topicnet',
    packages=[
        'topicnet',
        'topicnet.cooking_machine',
        'topicnet.cooking_machine.cubes',
        'topicnet.cooking_machine.models',
        'topicnet.cooking_machine.recipes',
        'topicnet.dataset_manager',
        'topicnet.viewers',
    ],
    version='0.9.0',
    license='MIT',
    description='TopicNet is a module for topic modelling using ARTM algorithm',
    long_description=_README_CONTENT,
    long_description_content_type='text/markdown',
    author='Machine Intelligence Laboratory',
    author_email='alex.goncharov@phystech.edu',
    url='https://github.com/machine-intelligence-laboratory/TopicNet',
    download_url='https://github.com/machine-intelligence-laboratory/TopicNet/archive/v0.9.0.tar.gz',
    keywords=[
        'ARTM',
        'topic modeling',
        'regularization',
        'multimodal learning',
        'document vector representation',
    ],
    install_requires=[
        'bigartm>=0.9.2',
        'colorlover==0.3.0',
        'dask[dataframe]==2023.5.0',
        'dill==0.3.8',
        'ipython==8.12.3',
        'jinja2==3.1.4',
        'numba==0.58.1',
        'numexpr==2.8.6',
        'numpy==1.24.4',
        'pandas==2.0.3',
        'plotly==5.20.0',
        'protobuf==3.20.3',  # BigARTM dependency
        'pytest==8.1.1',
        'scikit-learn==1.3.2',
        'scipy==1.10.1',
        'six==1.16.0',
        'strictyaml==1.7.3',
        'toolz==0.12.1',
        'tqdm==4.66.3',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
