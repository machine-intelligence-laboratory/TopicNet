from distutils.core import setup
setup(
  name = 'topicnet',         # How you named your package folder (MyLib)
  packages = ['topicnet', 'topicnet.cooking_machine', 'topicnet.cooking_machine.models', 'topicnet.cooking_machine.cubes', 'topicnet.viewers'],   # Chose the same as "name"
  version = '0.4.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'TopicNet is a module for topic modelling using ARTM algorithm',   # Give a short description about your library
  author = 'Machine Intelligence Laboratory',                   # Type in your name
  author_email = 'alex.goncharov@phystech.edu',      # Type in your E-Mail
  url = 'https://github.com/machine-intelligence-laboratory/TopicNet',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/machine-intelligence-laboratory/TopicNet/archive/v0.4.0.tar.gz', 
  keywords = ['ARTM', 'topic', 'modelling', 'visualization'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'six',
          'scipy',
          'numexpr',
          'pytest',
          'pandas',
          'tqdm',
          'dask',
          'scikit_learn',
          'typing',
          'ipython',
          'strictyaml'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
