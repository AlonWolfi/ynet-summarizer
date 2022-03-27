# %%
##### base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# display
from IPython.display import display


# warnings
import warnings
warnings.filterwarnings('ignore')

# fix random seed
from numpy.random import seed as set_random_seed
set_random_seed(42)


# debug
from icecream import ic
debug = ic


# %%
from data_extraction.articles import get_articles
articles_processed = get_articles(processed=True)
articles_processed.sample().iloc[0]
for col in articles_processed.columns:
    articles_processed = articles_processed[articles_processed[col].str.len() > 0]
from spacy.lang.he.stop_words import STOP_WORDS
STOP_WORDS.update(',')
from feature_extraction.article import tokenize_article
articles = articles_processed.apply(tokenize_article, axis = 1, args = [[STOP_WORDS]])
raw_articles = get_articles()

data = pd.merge(
    articles,
    raw_articles,
    right_index = True,
    left_index = True,
    how = 'inner',
    suffixes= ('', '_raw')
)
#TODO: remove '?'

# %%
from utils.metric_utils import jaccard_score, bleu, rouge

def check_score(r, score = jaccard_score, *score_args, **score_kwargs):
    summary = r['summary_tokens']
    sub_title = r['sub_title']
    return score(sub_title[0], summary, *score_args, **score_kwargs)
# scores = data.apply(check_score, axis = 1, args = (bleu, 1))
# data['score'] = scores

# %%
test = data.sample(50)

# %%
from models.nlp.summarization.textrank import TextRankModel
from models.nlp.summarization.firstk import  FirstKModel
from scipy.special import expit


# %%
def compare_models(models: dict, metrics: dict, train, test, fit = False, plot = False):
    summaries_pred = {}
    scores_table = {}
    for model_name, model in models.items():
        if fit:
            model.fit(
                train['content'],
                train['content_raw'],
                y = train['sub_title'].apply(lambda l: l[0]),
            )
        summaries_pred[model_name] = model.predict(
            test['content'],
            test['content_raw']
        )
        scores_table[model_name] = {}
    for metric_name, metric_args in metrics.items():
        for model_name, model in models.items():
            data = test.copy(deep=True)
            data = pd.merge(
                data,
                summaries_pred[model_name],
                right_index = True,
                left_index = True,
                how = 'inner',
                suffixes= ('', '')
            )
            scores = data.apply(check_score, axis = 1, args = metric_args)
            score_mean = np.round(scores.mean(), 2)
            if plot:
                plt.hist(scores, label = f'{model_name}: {score_mean}', alpha = 1.5 * (1/len(models.keys())))
            scores_table[model_name][metric_name] = score_mean
        if plot:
            plt.title(f'Hisogram for {metric_name} score')
            plt.legend()
            plt.show()
    return pd.DataFrame(scores_table)

# %%
t = TextRankModel(None)
s = t.predict(test['content'][0],test['content_raw'][0])
print(s['summary'][0])
t = TextRankModel(None,'word2vec')
s = t.predict(test['content'][0],test['content_raw'][0])
print(s['summary'][0])
t = TextRankModel(None,'word2vec')
s = t.predict(test['content'][0],test['content_raw'][0])
print(s['summary'][0],expit)


# %%
scores_table = compare_models(
    train = data,
    test = test,
    models = {
        'first-1': FirstKModel(STOP_WORDS, k=1),
        'first-2': FirstKModel(STOP_WORDS, k=2),
        'base': TextRankModel(STOP_WORDS),
        'word2vec': TextRankModel(STOP_WORDS,'word2vec'),
        'word2vec+sigmoid': TextRankModel(STOP_WORDS,'word2vec', expit),

        
    },
    metrics = {
        'jaccard': (jaccard_score,),
        'bleu-1': (bleu, 1),
        'bleu-2': (bleu, 2),
        'bleu-3': (bleu, 3),
        'rouge-1': (rouge, 1),
        'rouge-2': (rouge, 2),
        'rouge-3': (rouge, 3),
        'rouge-L': (rouge, 'L'),
    },
    fit = False,
    plot = True
)


