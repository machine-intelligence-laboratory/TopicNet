# TopicNet Recipes

This module contains mechanisms to generate code for training topic models on your data. It was created in orded to simplify knowledge transition about model training from the field researchers to the end users and possibly for easier exchange of a code between the research groups. As a backbone it uses snippets of YAML configs that require filling in information about the collection and hyperparameters of the required topic model.
Currently it is recommended to import `BaselineRecipe`, `SearchRecipe`, `MultimodalSearchRecipe` classes for the experiment environment generation. However, for the compatibility with previous examples found in `topicnet/demos/*-Recipe.ipynb` notebooks we also have `ARTM-baseline` and `exploratory_search` configs in YAML format.

----

* `BaselineRecipe` - Class for generating a pipeline training a topic models with decorrelation regularization, maximizing custom BleiLafferty score from TopicNet library `topicnet.cooking_machine.models.scores.BleiLaffertyScore`.
* `SearchRecipe` - a Class recreating training scenario from `exploratory_search` YAML config. Provides good startegy for training topic models for collection search properties. A link to the publication can be found in the comments section of the recipe.
* `MultimodalSearchRecipe` - a Class that modifies previos strategy for the case of multimodal data allowing to recreate previous scenario for each modality separately.
* `intratext_coherence_maximization.yml` - a strin in YAML format (like the old recipes) allowing to build topic model with decorrelation, Phi and Theta matrices Sparsing and Smoothing with background topics maximizing the intratext coherence score `topicnet.cooking_machine.models.scores.IntratextCoherenceScore`.
* `topic_number_search.yml` - a recipe recreating published strategy to find optimal topic number for given dataset. References to the publication can be found in the config dosctring.