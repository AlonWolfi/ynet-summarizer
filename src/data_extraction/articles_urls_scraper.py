import bs4
import pandas as pd
import requests

valid_topic_ids = ['891-317', '109-185']


def _scrape_articles_urls_from_topic(topic_id: str, year: int, month: int):
    if year < 2000:
        year = 2000 + year if year < 30 else 1900 + year
    if month < 10:
        month = '0' + str(month)
    url = f'https://www.ynet.co.il/home/0,7340,L-4269-{topic_id}-{year}{month}-1,00.html'
    html = bs4.BeautifulSoup(requests.get(url).content)
    html = html.find_all(attrs={'class': 'ghciArticleIndex1'})[0].find_all('table')[1]

    articles_htmls = html.find_all(attrs={'class': 'smallheader'})
    articles_urls = ['https://www.ynet.co.il' + article.get('href')
                     for article in articles_htmls]
    return articles_urls


def _scrape_articles():
    articles_urls = []
    for topic_id in valid_topic_ids:
        for year in range(2000, 2021 + 1):
            for month in range(1, 12 + 1):
                try:
                    urls = _scrape_articles_urls_from_topic(topic_id, year, month)
                    assert type(urls) == list
                    articles_urls += urls
                except:
                    print(topic_id, year, month, 'failed')
                    pass
    return list(set(articles_urls))


import os
from config import DATA_DIR
from utils.utils import read_data, save_data


def get_article_urls():
    article_urls_path = DATA_DIR / 'articles_urls.txt'

    if os.path.exists(article_urls_path):
        article_urls = read_data(article_urls_path).split('\n')
    else:
        article_urls = _scrape_articles()
        article_urls_txt = '\n'.join(article_urls)
        save_data(article_urls_txt, article_urls_path)
    return pd.Series(
        article_urls
    )
