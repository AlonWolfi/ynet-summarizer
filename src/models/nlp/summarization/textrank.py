from tkinter.messagebox import NO
import networkx as nx
import numpy as np
import pandas as pd
from nltk.cluster.util import cosine_distance
from tqdm.notebook import tqdm
import gensim

from .base_model import BaseModel


class TextRankModel(BaseModel):

    def __init__(self, stop_words, model=None, weight=None):
        self.stop_words = stop_words
        self.model = model # model to use with textrank
        self.weight = weight # weight function to use with model

    
    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if self.model is None:
            self._sentence_similarity_base(sent1,sent2,stopwords)
        elif self.model == 'word2vec':
            self._sentence_similarity_with_word_embedding(sent1,sent2,stopwords)
        else:
            raise NotImplementedError()


    def _sentence_similarity_base(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # set default weight function
        if self.weight is None:
            self.weight = lambda x: x

        #load word2vec model
        model = gensim.models.Word2Vec.load('model/word2vec.model', mmap='r')
        wv = model.wv

        # build the vector for the first sentence
        for w1 in sent1:
            if w1 in stopwords:
                continue
            for w2 in sent2:
                if w2 in stopwords:
                    continue
                vector1[all_words.index(w1)] += self.weight(wv.similarity(w1,w2))

        # build the vector for the second sentence
        for w2 in sent2:
            if w2 in stopwords:
                continue
            for w1 in sent1:
                if w1 in stopwords:
                    continue
                vector2[all_words.index(w2)] += self.weight(wv.similarity(w2,w1))

        return 1 - cosine_distance(vector1, vector2)

    def _sentence_similarity_with_word_embedding(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:  # ignore if both are same sentences
                    continue
                similarity_matrix[idx1][idx2] = self._sentence_similarity(
                    sentences[idx1], sentences[idx2], stop_words)
        return similarity_matrix

    def _choose_best_sentence(self, sentences_list, article, stop_words):
        # stop_words = STOP_WORDS
        summarize_text = []
        # Step 1 - Read text and tokenize
        sentences = sentences_list
        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = self._build_similarity_matrix(
            sentences, stop_words)
        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(
            sentence_similarity_martix)
        scores = nx.pagerank_numpy(sentence_similarity_graph)
        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        ranks = np.argsort(list(scores.values()))[::-1]
        # Step 5 - Offcourse, output the summarize text
        return ranks[0]

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
            chosen_sentence_idx = self._choose_best_sentence(
                article_tokenized, article_raw, self.stop_words)
            summaries.append({
                'summary_tokens': article_tokenized[chosen_sentence_idx],
                'summary': article_raw.split('.')[chosen_sentence_idx]
            })
        return pd.DataFrame(
            data=summaries,
            index=articles_raw.index
        )
