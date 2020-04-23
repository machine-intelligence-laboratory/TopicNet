# Demo
This section provides demonstrations of how to use this library in NLP tasks.

1. [RTL-Wiki-Preprocessing](RTL-Wiki-Preprocessing.ipynb) --  notebook describing how to get a Wikipedia dataset and write data in VW format.

2. [RTL-Wiki-Building-Topic-Mode](RTL-Wiki-Building-Topic-Model.ipynb) --  notebook with first steps to build topic model by consequently tuning its hyperparameters

3. [Visualizing-Your-Model-Documents](Visualizing-Your-Model-Documents.ipynb) -- notebook providing a fresh outlook on unstructured document collection with the help of a topic model

4. [20NG-Preprocessing](20NG-Preprocessing.ipynb) -- preparing data from a well-know 20 Newsgroups dataset

5. [20NG-GenSim-vs-TopicNet](20NG-GenSim-vs-TopicNet.ipynb) -- a comparison between two topic models build by Gensim and TopicNet library. In the notebook we compare model topics by calculating their [UMass coherence measure](https://palmetto.demos.dice-research.org/) and using Jaccard measure to compare topic top-tokens diversity

6. [PostNauka-Building-Topic-Model](PostNauka-Building-Topic-Model.ipynb)-- an analog of the RTL-Wiki notebook performed on the corpus of Russian pop-science articles given by postnauka.ru

7. [PostNauka-Recipe](PostNauka-Recipe) -- a demonstration of rapid-prototyping methods provided by the library

8. [Coherence-Maximization-Recipe](Coherence-Maximization-Recipe.ipynb) -- a recipe for hyperparameter search in regard to custom Coherence metric

9. [Topic-Prior-Regularizer-Tutorial](Topic-Prior-Regularizer-Tutorial.ipynb) -- a demonstration of the approach to learning topics from the unbalanced corpus

10. [Making-Decorrelation-and-Topic-Selection-Friends](Making-Decorrelation-and-Topic-Selection-Friends.ipynb) -- reproduction of a very complicated experiment on automatically learning optimal number of topics from the collection. Hurdle is -- both needed regularizers when working together nullify token-topic matrix.

----
P.S. All the guides are supposed to contain **working** examples of the library code.
If you happen to find code that is no longer works, please write about it in the library issues.
We will try to resolve it as soon as possible and plan to include fixes in the nearest releases.
