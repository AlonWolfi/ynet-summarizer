import pandas as pd


def process_keyword(keyword):
    features = pd.DataFrame(index=keyword.index)
    features = features.join(pd.get_dummies(keyword, prefix='OH'))
    features['isna'] = keyword.isna().astype('int')
    features.columns = ['keyword_' + c for c in features.columns]
    return features
