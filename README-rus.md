# TopicNet 
[English version](README.md)

---
### Что такое TopicNet?
Библиотека ```topicnet``` помогает строить тематические модели посредством автоматизации рутинных процессов моделирования.

### Как работать с библиотекой?
Сначала, вы инициализируете объект ```TopicModel``` с помощью имеющейся ARTM модели или конструируете первую модель
при помощи модуля ```model_constructor```.
Полученной моделью нужно проинициализировать экземпляр класса ```Experiment``` ведущий учёт этапов тренировки
и моделей полученных в ходе этих этапов.
Все возможные на данный момент типы этапов тренировки содержатся в ```cooking_machine.cubes``` а просмотреть полученные модели
можно при помощи модуля ```viewers``` имеющего широкий функционал способов выведения информации о модели.

### Кому может быть полезна данная библиотека?
Данный проект будет интересен двум категориям пользователей.
* Во-первых, он будет полезен для тех, кто хочет воспользоваться функционалом предоставляемым библиотекой BigARTM, но не готов 
писать дополнительный код для построения пайплайнов тренировки и вывода результатов моделирования.
* Во-вторых этот код будет полезен опытным конструкторам тематических моделей, так как позволяет быстро построить
прототип желаемого решения.

---
## Как установить TopicNet

Можно форкнуть данный репозиторий или же установить его с помощью команды: ```pip install topicnet```.
**Тем не меннее**,  большая часть функционала TopicNet завязана на библиотеку BigARTM, которая требует установки вручную.
Более подробное описание установки BigARTM можно найти здесь: [BigARTM installation manual](https://bigartm.readthedocs.io/en/stable/installation/index.html)

---
## Краткая инструкция по работе с TopicNet
Предположим у вас есть куча сырых текстов из какого-то источника и вы хотите навернуть тематическую модельку сверху этого всего.
С чего начать?

### Подготовка данных
Как и в любой другой ML задаче данные должны быть сначала подготовленны. TopicNet оставляет обработку данных на предусмотрение пользователя. Тем не менее для работы с предобработанными данными используется класс [Dataset (нужна документация)]()
Далее средует пример такой предобработки:
```
import nltk
import artm
import string

import pandas as pd
from glob import glob

WIKI_DATA_PATH = '/Wiki_raw_set/raw_plaintexts/'
files = glob(WIKI_DATA_PATH+'*.txt')
```
Загружаем все текстовые данные оставляя только буквы:
```
right_symbols = string.ascii_letters + ' '
data = []
for path in files:
    entry = {}
    entry['id'] = path.split('/')[-1].split('.')[0]
    with open(path,'r') as f:
        text = ''.join([char for char in f.read() if char in right_symbols])
        entry['raw_text'] = ''.join(text.split('\n')).lower()
    data.append(entry)
wiki_texts = pd.DataFrame(data)
```
#### Проведем токенизацию:
```
tokenized_text = []
for text in wiki_texts['raw_text'].values:
    tokenized_text.append(' '.join(nltk.word_tokenize(text)))
wiki_texts['tokenized'] = tokenized_text
```
#### Лемматизация:
```
from nltk.stem import WordNetLemmatizer
lemmatized_text = []
wnl = WordNetLemmatizer()
for text in wiki_texts['raw_text'].values:
    lemmatized = [wnl.lemmatize(word) for word in text.split()]
    lemmatized_text.append(lemmatized)
wiki_texts['lemmatized'] = lemmatized_text
```
#### Найдём биграмы:
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

#### Переведём все в формат Vowpal Wabbit и сохраним результаты на диск:
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
### Тренировка тематической модели
Теперь можно приступить к самому интеерсному: создание своей собственной, самой лучшей насвете, крафтовой Тематической Модели.
#### Загрузим данные
```
data = Dataset('/Wiki_raw_set/wiki_data.csv')
```
#### Создадим начальную модель
В случае если вы хотите создать свежую артм модель воспользуйтесь следующим кодом:
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
Следует отметить, что мы получаем модель с двумя модальностями: `'@lemmatized'` и `'@bigram'`.
Далее, при необходимости, можно определить свой скор, который будет считаться при тренировки модели:
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
Теперь, `TopicModel` с кастомным скором может быть определена следующим образом:
```
from topicnet.cooking_machine.models.topic_model import TopicModel

custom_score_dict = {'SpecificSparsity': ThatCustomScore()}
tm = TopicModel(model_artm, model_id='Groot', custom_scores=custom_score_dict)
```
#### Определить эксперимент
```
from topicnet.cooking_machine.experiment import Experiment
experiment = Experiment(experiment_id="simple_experiment", save_path="experiments", topic_model=tm)
```
#### Взять кубики
Определим этап тренировки модели и применим его к имеющейся модели:
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
Выберем лучшую модель для следующего этапа:
```
perplexity_select = 'PerplexityScore@lemmatized -> min COLLECT 1'
best_model = experiment.select(perplexity_select)
```
#### Просмотр результатов моделирования
Вывести данные о модели легко: создайте необходимый вам вьювер и используйте его view метод.
```
thresh = 1e-5
top_tok = TopTokensViewer(best_model, num_top_tokens=10, method='phi')
top_tok_html =  top_tok.to_html(top_tok.view(),thresh=thresh)
for line in first_model_html:
    display_html(line, raw=True)
```
