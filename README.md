# TopicNet 
[Русская версия](README-rus.md)

---
### What is TopicNet?
```topicnet```  library was created to assist in the task of building topic models. It aims at automating model training routine freeing more time for artistic process of constructing a target functional for the task at hand.
### How does it work?
The work starts with defining ```TopicModel``` from an ARTM model at hand or with help from ```model_constructor``` module. This model is then assigned a root position for the ```Experiment``` that will provide infrastructure for the model building process. Further, the user can define a set of training stages by the functionality provided by the ```cooking_machine.cubes``` modules and observe results of their actions via ```viewers``` module.
### Who will use this repo?
This repo is intended to be used by people that want to explore BigARTM functionality without writing an essential overhead for model training pipelines and information retrieval. It might be helpful for the experienced users to help with rapid solution prototyping

---
## How to install TopicNet
**Core library functionality is based on BigARTM library** which requires manual installation.  
To avoid that you can use a [docker images](https://hub.docker.com/r/xtonev/bigartm/tags) with preinstalled topicnet libraries in them. 
Alternatively, you can follow [BigARTM installation manual](https://bigartm.readthedocs.io/en/stable/installation/index.html)
After setting up the environment you can fork this repository or use ```pip install topicnet``` to install the library.  

---
## How to use TopicNet
Let's say you have a handful of raw text mined from some source and you want to perform some topic modelling on them. Where should you start? 
### Data Preparation
As always before training data needs to be preprocessed. TopicNet does not perform data preprocessing itself. Instead, it demands data being prepared by the user and loaded via [Dataset (no link yet)]() class.
Here is a basic example of how one can achieve that:
```
import nltk
import artm
import string

import pandas as pd
from glob import glob

WIKI_DATA_PATH = '/Wiki_raw_set/raw_plaintexts/'
files = glob(WIKI_DATA_PATH+'*.txt')
```
Loading all texts from files and leaving only alphabetical characters and spaces:
```
right_symbols = string.ascii_letters + ' '
data = []
for path in files:
    entry = {}
    entry['id'] = path.split('/')[-1].split('.')[0]
    with open(path,'r') as f:
        text = ''.join([char for char in f.read() if char in right_symbols])
        entry['raw_text'] = ''.join(text.split('\n'))
    data.append(entry)
wiki_texts = pd.DataFrame(data)
```
#### Perform tokenization:
```
tokenized_text = []
for text in wiki_texts['raw_text'].values:
    tokenized_text.append(' '.join(nltk.word_tokenize(text)))
wiki_texts['tokenized'] = tokenized_text
```
#### Perform lemmatization:
```
from nltk.stem import WordNetLemmatizer
lemmatized_text = []
wnl = WordNetLemmatizer()
for text in wiki_texts['raw_text'].values:
    lemmatized = [wnl.lemmatize(word) for word in text.split()]
    lemmatized_text.append(lemmatized)
wiki_texts['lemmatized'] = lemmatized_text
```
#### Get bigrams:
```
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_documents(wiki_texts['lemmatized'])
finder.apply_freq_filter(5)
set_dict = set(finder.nbest(bigram_measures.pmi,32100)[100:])
documents = wiki_texts['lemmatized']
bigrams = []
for doc in documents:
    entry = ['_'.join([word_first, word_second])
             for word_first, word_second in zip(doc[:-1],doc[1:])
             if (word_first, word_second) in set_dict]
    bigrams.append(entry)
wiki_texts['bigram'] = bigrams
```

#### Write them all to Vowpal Wabbit format and save result to disk:
```
vw_text = []
for index, data in wiki_texts.iterrows():
    vw_string = ''    
    doc_id = data.id
    lemmatized = '@lemmatized ' + ' '.join(data.lemmatized)
    bigram = '@bigram ' + ' '.join(data.bigram)
    vw_string = ' |'.join([doc_id, lemmatized, bigram])
    vw_text.append(vw_string)
wiki_texts['vw_text'] = vw_text

wiki_texts[['id','raw_text', 'vw_text']].to_csv('/Wiki_raw_set/wiki_data.csv')
```
### Training topic model
Here we can finally get on the main part: making your own, best of them all, manually crafted Topic Model
#### Get your data
We need to load our data prepared previously with Dataset:
```
data = Dataset('/Wiki_raw_set/wiki_data.csv')
```
#### Make initial model
In case you want to start from a fresh model we suggest you use this code:
```
from topicnet.cooking_machine.model_constructor import init_simple_default_model

model_artm = init_simple_default_model(
    dataset=demo_data,
    modalities_to_use={'@lemmatized': 1.0, '@bigram':0.5},
    main_modality='@lemmatized',
    n_specific_topics=14,
    n_background_topics=1,
)
```
Note that here we have model with two modalities: `'@lemmatized'` and `'@bigram'`.  
Further, if needed, one can define a custom score to be calculated during the model training.
```
from topicnet.cooking_machine.models.base_score import BaseScore

class ThatCustomScore(BaseScore):
    def __init__(self):
        super().__init__()

    def call(self, model,
             eps=1e-5,
             n_specific_topics=14):
        phi = model.get_phi().values[:,:n_specific_topics]
        specific_sparsity = np.sum(phi < eps) / np.sum(phi < 1)
        return specific_sparsity
```
Now, `TopicModel` with custom score can be defined:
```
from topicnet.cooking_machine.models.topic_model import TopicModel

custom_score_dict = {'SpecificSparsity': ThatCustomScore()}
tm = TopicModel(model_artm, model_id='Groot', custom_scores=custom_score_dict)
```
#### Define experiment
```
from topicnet.cooking_machine.experiment import Experiment
experiment = Experiment(experiment_id="simple_experiment", save_path="experiments", topic_model=tm)
```
#### Toy with the cubes
Defining a next stage of the model training:
```
from topicnet.cooking_machine.cubes import RegularizersModifierCube

my_first_cube = RegularizersModifierCube(
    num_iter=5,
    tracked_score_function=retrieve_score_for_strategy('PerplexityScore@lemmatized'),
    regularizer_parameters={
        'regularizer': artm.DecorrelatorPhiRegularizer(name='decorrelation_phi', tau=1),
        'tau_grid': [0,1,2,3,4,5],
    },
    reg_search='grid',
    verbose=True
)
my_first_cube(tm, demo_data)
```
Selecting a model with best perplexity score:
```
perplexity_select = 'PerplexityScore@lemmatized -> min COLLECT 1'
best_model = experiment.select(perplexity_select)
```
#### View the results
Browsing the model is easy: create a viewer and call its `view()` method:
```
thresh = 1e-5
top_tok = TopTokensViewer(best_model, num_top_tokens=10, method='phi')
top_tok_html =  top_tok.to_html(top_tok.view(),thresh=thresh)
for line in first_model_html:
    display_html(line, raw=True)
```
