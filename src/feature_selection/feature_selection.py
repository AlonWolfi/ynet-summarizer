from scipy.stats import pearsonr


# from scipy.stats import pearsonr, spearmanr, kstest
# from sklearn.feature_selection import SelectPercentile, chi2
# sp = SelectPercentile(score_func=chi2, percentile=20)
# X_train = sp.fit_transform(X_train, y_train)
# X_test = sp.transform(X_test)

def select_features(X, y):
    print(f'started with {len(X.columns)} features')
    final_cols = []
    for c in X.columns:
        col = X[c]
        if col.nunique() < 5.:
            continue
        if abs(pearsonr(y, col)[0]) < 0.005:
            continue

        final_cols.append(c)
    print(f'finished with {len(final_cols)} features')
    return final_cols
