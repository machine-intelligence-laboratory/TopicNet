# Cooking Machine

#### Cube
A unit of model training pipeline. This unit encapsulates an action over one or many model hyperparameters. This action and hyperparameter space are stored as cube properties and then saved in Experiment. 

**Input:** model or list of models, regularizer or list of them, hyperparameter search sapce(grid), iterations number or a function defining it, custom metrics.  
**Output:** models.  
**Body:** performs actions over `artm` model. Can modify, create new models and alter their Experiment.

#### Model
A class containing Topic Model and its description:

* stores topic model description;
* outputs the description in human-readable form;
* the model can only load and copy itself, the artm-model is an attribute and in order to change it is should be extracted, modified and put back;
* stores experiment id;
* stores parent model id;
* stores model topic names;
* stores regularizers list with their parameters;
* stores modality weights;
* stores save path for data, model and model information;
* stores training metric values.

#### Experiment
Class providing experiment infrastructure:

* keeps the description of all actions on the models;
* provides human-readable log of experiment;
* keeps the model training sequence in memory;
* automaticly runs integrity check;
* able to copy itself.