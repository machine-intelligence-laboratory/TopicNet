# Models and scores

Availiable models:

* `BaseModel` — Parent class for model creation
* `TopicModel` — a wrapper class for bigartm topic model
* `DummyTopicModel` — a fake model that contains training information but not actual artm model. Needed to save memory space during the training.
---

Availiable scores:

* `BaseScore` — a parent class for all the Strategies
* `ExampleScore` — Example of minimal working example of custom score
* `IntratextCoherenceScore` — score that calculates coherence as a measure of interpretability of the model using raw documents from dataset. Calculation-heavy score. Recommended to be used **after** model training 
* `BleiLaffertyScore` — An experimental light-weight score to estimate interpretability of the topics
---

## Internal model structure


**main model attributes**:

* `model_id` — a model string id, unique for its Experiment.

* `scores` — dict of lists,
each list corresponds to the score value or list of values at certain training stage.

* `custom_scores` — variable providing custom scores for the model

**main model methods**:

* `_fit` — function performing model training. Takes the dataset and number of iterations.

    **Important Notice!**
    We assume that the model training happens through Cube interface and this method, while 
    important should never be used by users if they are hope to have their actions logged


* `get_phi` — function that returns p(token|topic/cluster) probability distributions
that returns pandas.DataFrame with tokens as index and topics/clusters as columns

    **Important Notice!**
    Strictly speaking the function returns degree to which token belongs to the
    topic/cluster and shouldn't be a probability distribution. But scince its main use-case
    intended for topic models some of the functions using this method might work incorrectly
    in non-distribution case

    
* `get_theta` — function that returns p(topic/cluster|document) probability distributions
that returns pandas.DataFrame with topics/clusters as index and document ids as columns.

    **Important Notice!**
    Strictly speaking the function returns degree to which document belongs to the
    topic/cluster and shouldn't be a probability distribution. But scince its main use-case
    intended for topic models some of the functions using this method might work incorrectly
    in non-distribution case


* `save` — saves model to the path directory.

* `load` — loads model from the path directory 

* `clone` — creates copy of a model.

* `get_jsonable_from_parameters` — turns model parameters to jsonable format for logging purposes

---

## What do you need to create your own model?

Following this steps you should be able to code a model integrated with the library methods:

1. New model class is inherrited from BaseModel

2. A child class should contain methods `__init__`, `_fit`, `get_phi`, `get_theta`,
`save`, `load`, `clone`, `get_jsonable_from_parameters`.
