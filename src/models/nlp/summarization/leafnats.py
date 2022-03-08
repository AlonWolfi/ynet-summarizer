import argparse
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from config import NATS_DIR, PROJECT_DIR, TMP_DIR
from nltk.cluster.util import cosine_distance
from this import d
from tqdm.notebook import tqdm

from .base_model import BaseModel
from .LeafNATS.eval_scripts.eval_pyrouge import run_pyrouge
from .LeafNATS.utils.utils import str2bool
from .pointer_generator_network.model import modelPointerGenerator


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LeafNATSModel(BaseModel):

    def __init__(self, stop_words, **params):
        # 'task':'validate',
        if not os.path.exists(TMP_DIR / 'data'):
            os.mkdir(TMP_DIR)
            os.mkdir(TMP_DIR / 'data')
        default_args = {
            'data_dir': TMP_DIR / 'data',
            'file_corpus': 'train.txt',
            'file_val': 'val.txt',
            'n_epoch': 35,
            'batch_size': 16,
            'checkpoint': 100,
            'val_num_batch': 30,
            'nbestmodel': 10,
            'continue_training': True,
            'train_base_model': False,
            'use_move_avg': False,
            'use_optimal_model': True,
            'model_optimal_key': '0,0',
            'is_lower': False,
            'device': torch.device("cpu"),
            'file_vocab': 'vocab',
            'max_vocab_size': 50000,
            'word_minfreq': 5,
            'emb_dim': 128,
            'src_hidden_dim': 256,
            'trg_hidden_dim': 256,
            'src_seq_lens': 400,
            'trg_seq_lens': 100,
            'rnn_network': 'lstm',
            'attn_method': 'luong_concat',
            'repetition': 'vanilla',
            'pointer_net': True,
            'oov_explicit': True,
            'attn_decoder': True,
            'share_emb_weight': True,
            'learning_rate': 0.0001,
            'grad_clip': 2.0,
            'file_test': 'test.txt',
            'file_output': 'summaries.txt',
            'beam_size': 5,
            'test_batch_size': 1,
            'copy_words': True,
            # for app
            'app_model_dir': '../../pg_model/',
            'app_data_dir': '../../',
        }

        self.args = default_args
        self.args.update(params)

    @classmethod
    def pandas_to_txt(cls, articles_tokenized, y, filename):
        lines = []
        for article, summary in zip(articles_tokenized, y):
            article_joined = ' '.join(
                [f"<s> {' '.join(sentence)} </s>" for sentence in article])
            summary_joined = f"<s> {' '.join(summary)}</s>"
            lines.append(
                f'''{article_joined}<sec>{summary_joined}<sec>{summary_joined}'''
            )
        with open(str(TMP_DIR / 'data' / (filename + '.txt')), 'w+', encoding="utf-8") as f:
            for line in lines:
                f.write(line + '\n')
        return filename

    @classmethod
    def create_corpus(cls, articles_tokenized, y):
        vocab = defaultdict(lambda: 0)
        for article, summary in zip(articles_tokenized, y):
            for sentence in article:
                for token in sentence:
                    vocab[token] += 1
            for token in summary:
                vocab[token] += 1
        with open(str(TMP_DIR / 'data' / 'vocab'), 'w+', encoding="utf-8") as f:
            for token, cnt in vocab.items():
                f.write(token + ' ' + str(cnt) + '\n')
    
    def fit(self, articles_tokenized, articles_raw=None, y=None, *args, **kwargs):
        self.pandas_to_txt(articles_tokenized, y, 'train')
        self.create_corpus(articles_tokenized, y)

        
        self.args['task'] = 'train'
        self.model = modelPointerGenerator(Namespace(**self.args))
        self.model.train()

    def _choose_last_model(self):
        last_model = [f for f in os.listdir(NATS_DIR) if f.startswith('decoder2proj')][-1]
        _, last_epoch, last_batch = last_model.split('_')
        last_batch = last_batch.split('.')[0]

        with open(os.path.join(NATS_DIR, 'model_validate.txt'), 'w+') as f:
            f.write(
                f'''..\\nats_results\\{last_model} {last_epoch} {last_batch} 1.00 4.00''' + '\n'
            )


    def predict(self, articles_tokenized, articles_raw=None, y=None, *args, **kwargs):
        y = pd.Series([['טיוטה']] * len(articles_tokenized),
                      index=articles_tokenized.index)
        # self.pandas_to_txt(articles_tokenized, y, 'val')
        self.pandas_to_txt(articles_tokenized, y, 'test')
        self._choose_last_model()
        
        self.args['task'] = 'test'
        self.model = modelPointerGenerator(Namespace(**self.args))
        self.model.test()

        summaries = []
        with open(str(PROJECT_DIR / 'nats_results' / 'summaries.txt'), 'r+', encoding="utf-8") as f:
            for line in f.readlines():

                summary = line.split('<sec>')[0]
                summaries.append({
                    'summary_tokens': summary.split(' '),
                    'summary': summary
                })
        return pd.DataFrame(
            data=summaries,
            index=articles_tokenized.index
        )
