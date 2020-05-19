# TopicNet

The library was created to assist in the task of building topic models.
It aims to automate away many routine tasks related to topic model training, allowing a user to focus on the task at hand.
Also, it provides additional tools to construct advanced topic models.
The library consists of the following modules:

* `cooking_machine` — provides tools to design a topic model construction pipeline, or experiment with regularizers fitting
* `viewers` — provides information about the topic model in an accessible format
* `demos` — demo .ipynb notebooks
* `dataset_manager` — gives opportunity to download datasets for experiments
* `tests` — provides a user with means to test library functionality (contains some examples of intended library usage)


## Project description

In TopicNet framework, advanced topic models are build using Experiment class.
An experiment consists of stages (that we call "cubes") which perform actions over the "models" which are objects of the Experiment. The experiment instance of Experiment class contains all the information about the experiment process and automatically updates its log when a cube is applied to the last level models.
It is worth noting that the experiment is linear, meaning it does not support multiple different cubes at the same stage of the experiment. If that need arises one is recommended to create a new experiment with a new cube on the last level.
The experiment instance of Experiment class contains all the information about the *experiment process* and automatically updates its log when the cube is applied to the last level models.
Summarizing: the key entity Experiment is a sequence of cubes that produce models on each stage of the *experiment process*
