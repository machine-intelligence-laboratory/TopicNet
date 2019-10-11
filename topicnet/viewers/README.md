## Viewers 

[Русская версия](README-rus.md)

Module ```viewers``` provides information from a topic model allowing to estimate the model quality. Its advantage is in unified call ifrastucture to the topic model making the routine and tedious task of extracting the information easy.

Currently module contains following viewers:


* ```base_viewer``` - module responsible for base infrastructure


* ```initial_doc_to_topic_viewer``` - first edition of  ```TopDocumentsViewer``` - Deprecated


* ```spectr_viewer``` - first edition of ```TopicSpectrumViewer``` using t-SNE clustering  - Deprecated


* ```spectrum``` - module contains heuristics for solving TSP to arrange topics minimizing total distance of the spectrun, 
```TopicSpectrumViewer```


* ```tokens_viewer``` - first edition of  ```TopTokensViewer``` - Deprecated


* ```top_documents_viewer``` - module with functions that work with dataset document collections and class ```TopDocumentsViewer``` wrapping this functionality.


* ```top_similar_documents_viewer``` - module containing class for finding simmilar document for a givenone. This viewer helps to estimate homogeneity of clusters given by the model


* ```top_tokens_viewer``` - module with class for displaying the most relevant tokens in each topic of the model.


* ```topic_mapping``` - module allowing to compare topics between two different models trained on the same collection.