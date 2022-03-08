import networkx as nx
import numpy as np
import pandas as pd
from nltk.cluster.util import cosine_distance
from tqdm.notebook import tqdm

from .base_model import BaseModel


class FirstKModel(BaseModel):

    def __init__(self, stop_words, k=1):
        '''
            k: number of sentences to choose
        '''
        self.stop_words = stop_words
        self.k = k

    @classmethod
    def _join_k_lists(cls, list_of_lists):
        return [ele for l in list_of_lists for ele in l]

    def fit(self, articles_tokenized, articles_raw, y=None, *args, **kwargs):
        pass

    def predict(self, articles_tokenized, articles_raw, y=None, *args, **kwargs):
        if type(articles_raw) == str:
            articles_raw = [articles_raw]
            articles_tokenized = [articles_tokenized]
        if type(articles_raw) == list:
            articles_raw = pd.Series(articles_raw)
            articles_tokenized = pd.Series(articles_tokenized)
        summaries = []
        for article_tokenized, article_raw in tqdm(list(zip(articles_tokenized, articles_raw))):
            summaries.append({
                'summary_tokens': self._join_k_lists(article_tokenized[:self.k]),
                'summary': '.'.join(article_raw.split('.')[:self.k])
            })
        return pd.DataFrame(
            data=summaries,
            index=articles_raw.index
        )
