import os

import pandas as pd

from config import DATA_DIR
from data_extraction.articles_urls_scraper import get_article_urls
from utils.utils import read_data, hash


def get_articles(processed=False):
    if processed:
        articles_dir_path = DATA_DIR / 'articles_processed'
    else:
        articles_dir_path = DATA_DIR / 'articles'
    article_urls = get_article_urls()
    articles_paths = []
    for url in article_urls:
        article_filename = hash(url) + '.pickle'
        article_path = articles_dir_path / article_filename

        if os.path.exists(article_path):
            article = read_data(article_path)
            articles_paths.append(pd.Series(
                article,
                name=url
            ))
    return pd.DataFrame(
        articles_paths
    )
