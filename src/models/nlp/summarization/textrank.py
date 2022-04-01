from tkinter.messagebox import NO
import networkx as nx
import numpy as np
import pandas as pd
from nltk.cluster.util import cosine_distance
from tqdm.notebook import tqdm
import gensim
from config import PROJECT_DIR
from os import path
from .base_model import BaseModel


class TextRankModel(BaseModel):

    def __init__(self, stop_words, type=None, weight=None, N=1):
        self.stop_words = stop_words
        self.type = type # model to use with textrank

        self.weight = weight # weight function to use with model
        # set default weight function
        if self.weight is None:
            self.weight = lambda x: x

        self.N = N # number of sentences
        self.model = None
        if type == 'word2vec':
            #load word2vec model
            self.model = gensim.models.Word2Vec.load(path.join(PROJECT_DIR, 'word2vec/word2vec.model'), mmap='r')
        elif type == 'doc2vec':
            #load word2vec model
            self.model = gensim.models.Doc2Vec.load(path.join(PROJECT_DIR, 'doc2vec/doc2vec.model'), mmap='r')
    
    
    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if self.type is None:
            return self._sentence_similarity_base(sent1,sent2,stopwords)
        elif self.type == 'word2vec':
            return self._sentence_similarity_with_word_embedding(sent1,sent2,stopwords)
        elif self.type == 'doc2vec':
            return self._sentence_similarity_with_sentence_embedding(sent1,sent2,stopwords)
        else:
            raise NotImplementedError()


    def _sentence_similarity_with_word_embedding(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)


        # get word embeddings
        wv = self.model.wv

        # build the vector for the first sentence
        for w1 in sent1:
            if w1 in stopwords:
                continue
            vector1[all_words.index(w1)] += 1

            # add partial score for similar words
            if  wv.has_index_for(w1):
                for w2 in sent2:
                    if w2 in stopwords:
                        continue
                    if wv.has_index_for(w2):
                        vector1[all_words.index(w2)] = max(self.weight(wv.similarity(w1,w2)), vector1[all_words.index(w2)])

        # build the vector for the second sentence
        for w2 in sent2:
            if w2 in stopwords:
                continue
            vector2[all_words.index(w2)] += 1

            # add partial score for similar words
            if wv.has_index_for(w2):
                for w1 in sent1:
                    if w1 in stopwords:
                        continue
                    if wv.has_index_for(w1):
                        vector2[all_words.index(w1)] = max(self.weight(wv.similarity(w2,w1)), vector2[all_words.index(w1)])

        # for w in sent1:
        #     if wv.has_index_for(w):
        #         vector1.append(wv[w])
            
        # for w in sent2:
        #     if wv.has_index_for(w):
        #         vector2.append(wv[w])

        # vector1 = np.average(vector1, axis=0)
        # vector2 = np.average(vector1, axis=0)

        return 1 - cosine_distance(vector1, vector2)

    def _sentence_similarity_with_sentence_embedding(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        return self.weight(self.model.similarity_unseen_docs(sent1,sent2, epochs=100))

    def _sentence_similarity_base(self, sent1, sent2, stopwords=None):
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
        #print(similarity_matrix)
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
        return ranks[:self.N]

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
            raw_sentences = article_raw.split('.')

            # build summary
            summary_tokens = []
            summary = ""
            for idx in chosen_sentence_idx:
                summary_tokens += article_tokenized[idx] + ['.']
                summary += raw_sentences[idx]

            summaries.append({
                    'summary_tokens': summary_tokens,
                    'summary': summary
                })
        return pd.DataFrame(
            data=summaries,
            index=articles_raw.index
        )
