import os

import pandas as pd

from config import DATA_DIR
from nlp.yap_api import YapApi
from utils.utils import hash, read_data, save_data

yap = YapApi()
def process_article(article):
    content = ''
    tokenized_text, main_title, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(article['main_title'])
    tokenized_text, sub_title, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(article['sub_title'])
    tokenized_text, content, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(article['content'])
    return pd.Series({
        'main_title': main_title,
        'sub_title' : sub_title,
        'content' : content
    })



def save_process_article(article):  
    article_filename = hash(article.name) + '.pickle'
    article_path = DATA_DIR / 'articles_processed' / article_filename

    if os.path.exists(article_path):
        article_processed = read_data(article_path)
    else:
        article_processed = process_article(article)
        if article_processed is None:
            return None
        save_data(article_processed, article_path)
    return article_processed

def tokenize_sentence(s, stop_words = []):
    s = s.strip()
    tokens = s.split(' ')
    tokens = [c for c in tokens if c not in stop_words]
    return tokens
    
def tokenize_article(art, stop_words = []):
    art_tokenized = pd.Series(index=art.index, name=art.name)
    art_tokenized['main_title'] = tokenize_sentence(art['main_title'], stop_words)
    for col in ['sub_title', 'content']:
        art_tokenized[col] = art[col].strip().split('.')
        art_tokenized[col] = [tokenize_sentence(s, stop_words) for s in art_tokenized[col]]
    return art_tokenized
