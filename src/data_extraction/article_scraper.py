import json

import bs4
import pandas as pd
import requests

import os
from config import DATA_DIR
from utils.utils import read_data, save_data, hash

def scrape_article(article_url: str):
    try:
        print(article_url)
        if article_url.startswith('https'):
            article_url = article_url.replace('https', 'http')
        html = bs4.BeautifulSoup(requests.post(article_url, timeout=(100.05, 1000)).content)
        ld_json = html.find_all(attrs={'type': 'application/ld+json'})[0]
        ld_json = json.loads([l for l in ld_json.children][0])
        main_title = ld_json['headline']
        sub_title = ld_json['description']
        content = ld_json['articleBody']

        # main_title = html.find_all(attrs={'class' : 'mainTitle'})[0].text.strip('"')
        # sub_title = html.find_all(attrs={'class' : 'subTitle'})[0].text.strip('"')
        # content = html.find_all(attrs={'class' : 'public-DraftEditor-content'})[0].text.strip('"')
        return pd.Series({
            'main_title': main_title,
            'sub_title': sub_title,
            'content': content
        })
    except:
        print(f'{article_url} failed')
        return None

def get_article(article_url: str):
    article_filename = hash(article_url) + '.pickle'
    article_path = DATA_DIR / 'articles' / article_filename

    if os.path.exists(article_path):
        article_data = read_data(article_path)
    else:
        article_data = scrape_article(article_url)
        if article_data is None:
            return None
        save_data(article_data, article_path)
    return article_data
