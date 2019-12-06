# Cubes and their Strategies

Cube types:

* `BaseCube` — a parent class for all the Cubes
* `RegularizersModifierCube` — cube that adds or alter model regularizers
* `CubeCreator` — cube that allows to change model fundamental hyperparameters (topic number)
* `RegularizationControllerCube` - cube that ties together a complicated usage of `RegularizersModifierCube`. This cube allows for change of regularization coefficients across the model training. This allows to obtain soemwhat unique results by combining contradictionary restrictions on the model.
---

Strategy types:

* `BaseStrategy` — a parent class for all the Strategies
* `PerplexityStrategy` — performs search in given hyperparameter space until certain score exceeds a boundary 
* `GreedyStrategy` — strategy that performes search in hyperparameter space consequently changing dimensions to perform a 1D search for a minimum
---


## Cube internal structure

**The main cube attributes**:

* `parameters` — paramteres is an iterable object containing all
the specific information about current cube. 
The class architecture implies that parameters should contain an iterable field
describing the hyperparameters search space


**Cube methods worth noticing**:

* `__call__` — performes the cube actions to the model using provided dataset.
Always recieves instance of TopicModel class and instance of Dataset class.
This method does the internal workings of training models with new hyperparameters.
It is responsible for logging the events (which parameters where changed)
happening during the model training.

* `apply` — method of the cube that prepares model for further training.
This method should be specified by the user as it contains an "essence" of what is happening at this stage of the training. It could be new type of model reinitialization, change of the regualarization coefficient, adding a new level of hierarchy etc. This function defines what the cube does in the training pipeline.

* `get_jsonable_from_parameters` — is a cube-specific function that transforms it parameters to dict-like form which later is written in JSON format log of the experiment.

---

## What do you need to create a new cube?

Following this 3 easy steps you will be able to write down your own cube:

1. Inherit your Cube from BaseCube.

2. Child class should define following methods `__init__`, `apply`, `get_jsonable_from_parameters`.
It is strongly descouraged to change `__call__` method.

3. `get_jsonable_from_parameters()[i]` corresponds to the same cube step as `parameters[i]`.
