import pandas as pd


def raw_vs_processed(feature, process_function):
    return pd.DataFrame({
        'raw': feature,
        'process': process_function(feature)
    })
