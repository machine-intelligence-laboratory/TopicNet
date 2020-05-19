from .dataset import Dataset
import artm

import os
import re
import sys
import shutil
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm


class DatasetCooc(Dataset):
    """
    Class prepare dataset in vw format for WNTM model
    """
    def __init__(
        self,
        data_path: str,  # имя такое же, как у параметра обычного Датасета
        internals_folder_path: str = None,
        cooc_window: int = 10,
        min_tf: int = 5,
        min_df: int = 5,
        threshold: int = 2,
        **kwargs
    ):
        """
        Parameters
        ----------
        data_path : str
            path to a file with input data for training models
            in vowpal wabbit format;
        internals_folder_path : str
            path to the directory with dataset internals, which includes:

            * vowpal wabbit file
            * dictionary file
            * batches directory

            The parameter is optional:
            the folder will be created by the dataset if not specified.
            This is a part of Dataset internal functioning.
            When working with any text collection `data_path` for the first time,
            there is no such folder: it will be created by
            topicnet.cooking_machines.Dataset class.
        cooc_window : int
            number of tokens around specific token,
            which are used in calculation of
            cooccurrences
        min_tf : int
            minimal value of cooccurrences of a
            pair of tokens that are saved in
            dictionary of cooccurrences
            Optional parameter, default min_tf =5
            More info http://docs.bigartm.org/en/stable/tutorials/python_userguide/coherence.html
        min_df: int
            minimal value of documents in which a
            specific pair of tokens occurred
            together closely
            Optional parameter, default min_df =5
            More info http://docs.bigartm.org/en/stable/tutorials/python_userguide/coherence.html
        threshold : int
            The frequency threshold above which
            the received pairs are selected to form
            the dataset
        """

        self._ordinary_dataset = Dataset(
            data_path,  # just in case
            internals_folder_path=internals_folder_path,
            **kwargs
        )
        _ = self._ordinary_dataset.get_dictionary()
        _ = self._ordinary_dataset.get_batch_vectorizer()

        # Теперь создана internals папка, батчи и словарь обычного датасета, всё такое

        self.dataset_dir = os.path.join(
            self._ordinary_dataset._internals_folder_path,
            'coocs_dataset',  # как-то так: тут уже всё про совстречаемости
        )

        if not os.path.isdir(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        self.dataset_name = os.path.basename(data_path)
        self.dataset_path = data_path
        self.cooc_window = cooc_window
        self.min_tf = min_tf
        self.min_df = min_df

        self._get_vocab()
        self._get_cooc_scores(cooc_window, min_tf, min_df)
        self._get_vw_cooc(threshold)

        super().__init__(self.wntm_dataset_path)

    def _get_vocab(self):
        batch_vectorizer_path = os.path.join(self.dataset_dir, 'batches')
        artm.BatchVectorizer(data_path=self.dataset_path,
                             data_format='vowpal_wabbit',
                             target_folder=batch_vectorizer_path)

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=batch_vectorizer_path)
        dictionary_path = batch_vectorizer_path + '/dictionary.txt'
        dictionary.save_text(dictionary_path=dictionary_path)

        self.vocab_path = os.path.join(self.dataset_dir, 'vocab.txt')

        with open(dictionary_path, 'r') as dictionary_file:
            with open(self.vocab_path, 'w') as vocab_file:
                """
                The first two lines of dictionary_file do not contain data
                """
                dictionary_file.readline()
                dictionary_file.readline()
                for line in dictionary_file:
                    elems = re.split(', ', line)
                    vocab_file.write(' '.join(elems[:2]) + '\n')

    def _get_cooc_scores(self, cooc_window, min_tf, min_df):
        try:
            bigartm_tool_path = subprocess.check_output(["which", "bigartm"]).strip()
        except FileNotFoundError:
            sys.exit(
                """
                For use dataset_cooc.py please build bigartm tool

                https://bigartm.readthedocs.io/en/stable/installation/linux.html#step-3-build-and-install-bigartm-library

                """
            )

        cooc_tf_path = os.path.join(self.dataset_dir, 'cooc_tf_')
        cooc_df_path = os.path.join(self.dataset_dir, 'cooc_df_')
        ppmi_tf_path = os.path.join(self.dataset_dir, 'ppmi_tf_')
        ppmi_df_path = os.path.join(self.dataset_dir, 'ppmi_df_')

        subprocess.check_output([bigartm_tool_path, '-c', self.dataset_path, '-v',
                                 self.vocab_path, '--cooc-window', str(cooc_window),
                                 '--cooc-min-tf', str(min_tf), '--write-cooc-tf',
                                 cooc_tf_path, '--cooc-min-df', str(min_df),
                                 '--write-cooc-df', cooc_df_path, '--write-ppmi-tf',
                                 ppmi_tf_path, '--write-ppmi-df', ppmi_df_path])

    def _transform_coocs_file(
        self,
        source_file_path: str,
        target_file_path: str
    ):
        """
        source_file is assumed to be either ppmi_tf_ or ppmi_df_
        """

        vocab = open(self.vocab_path, 'r').readlines()
        vocab = [line.strip().split()[0] for line in vocab]

        cooc_values = dict()
        word_word_value_triples = set()

        lines = open(source_file_path, 'r').readlines()
        pbar = tqdm(total=len(lines))

        for i, l in enumerate(lines):
            pbar.update(10)
            l_i = l.strip()
            words = l_i.split()
            words = words[1:]  # exclude modality
            anchor_word = words[0]

            other_word_values = words[1:]

            for word_and_value in other_word_values:
                other_word, value = word_and_value.split(':')
                value = float(value)

                cooc_values[(anchor_word, other_word)] = value
                if (other_word, anchor_word) not in cooc_values:
                    cooc_values[(other_word, anchor_word)] = value

                word_word_value_triples.add(
                    tuple([
                        tuple(sorted([
                            vocab.index(anchor_word),
                            vocab.index(other_word)
                        ])),
                        value
                    ])
                )
        pbar.close()
        new_text = ''

        for (w1, w2), v in word_word_value_triples:
            new_text += f'{w1} {w2} {v}\n'

        with open(target_file_path, 'w') as f:
            f.write(''.join(new_text))

        return cooc_values

    def _get_vw_cooc(self, threshold):
        with open(self.vocab_path, 'r') as f:
            data = f.readlines()

        cooc_values = self._transform_coocs_file(
            os.path.join(self.dataset_dir, 'ppmi_tf_'),
            os.path.join(self.dataset_dir, 'new_ppmi_tf_')
        )

        vw_lines = {}

        for line in data:
            token, modality = line.strip().split()
            vw_lines[token] = '{} |{}'.format(token, modality)

        for coocs_pair, frequency in cooc_values.items():
            (token_doc, token_word) = coocs_pair
            if frequency >= threshold:
                vw_lines[token_doc] = vw_lines[token_doc] + ' ' + '{}:{}'.format(
                    token_word, frequency
                )

        self.wntm_dataset_path = os.path.join(self.dataset_dir, f'new_{self.dataset_name}')

        with open(self.wntm_dataset_path, 'w') as f:
            f.write('\n'.join(list(vw_lines.values())))

    def transform_theta(self, model):
        """
        Transform theta matrix
        """
        with open(self.dataset_path, 'r') as f:
            data = f.readlines()

        doc_token = {}
        for doc in data:
            doc = doc.split()
            doc_token[doc[0]] = [token.split(':')[0] for token in doc[2:]]

        token_doc = {}
        for doc in doc_token:
            for token in doc_token[doc]:
                if token not in token_doc:
                    token_doc[token] = [doc]
                else:
                    token_doc[token] += [doc]

        doc_inds = {doc: ind for ind, doc in enumerate(doc_token.keys())}
        nwd = {token: [0]*len(doc_inds) for token in token_doc}
        for token in token_doc:
            for doc in token_doc[token]:
                nwd[token][doc_inds[doc]] += 1

        theta = model.get_theta(dataset=self)
        cols = theta.columns
        inds = theta.index.values

        nwd_matrix = np.array([nwd[token] for token in cols])
        new_theta = np.dot(theta.values, nwd_matrix)
        return pd.DataFrame(data=new_theta, columns=doc_inds.keys(), index=inds)

    def clear_all_cooc_files(self):
        """
        Clear cooc_dir folder
        """
        shutil.rmtree(os.path.join(self.dataset_dir, 'batches'))
        os.remove(self.vocab_path)

        os.remove(os.path.join(self.dataset_dir, 'cooc_tf_'))
        os.remove(os.path.join(self.dataset_dir, 'cooc_df_'))
        os.remove(os.path.join(self.dataset_dir, 'ppmi_tf_'))
        os.remove(os.path.join(self.dataset_dir, 'ppmi_df_'))

        os.remove(os.path.join(self.dataset_dir, 'new_ppmi_tf_'))

        os.remove(self.WNTM_dataset_path)

        shutil.rmtree(self.dataset_dir)
