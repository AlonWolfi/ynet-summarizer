import numpy as np
from icecream import ic as debug
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def get_y(self, y_pred, y_true):
        return y_pred


class SoftMetric(Metric):
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, y_pred, y_true):
        return self.metric(y_true, y_pred)


class HardMetric(Metric):
    def __init__(self, metric):
        self.metric = metric

    @staticmethod
    def _get_y_by_thresh(y_proba, th=0.5):
        return (y_proba > th).astype(int)

    def get_thresh(self, y_test, y_pred, n_thers=500):
        thresholds = [n / n_thers for n in list(range(1, n_thers, 1))]
        scores = [self.metric(y_test, self._get_y_by_thresh(
            y_pred, thresh)) for thresh in thresholds]
        return thresholds[np.argmax(scores)]

    def __call__(self, y_true, y_pred):
        return self.metric(y_true, self.get_y(y_true, y_pred))

    def get_y(self, y_true, y_pred):
        if len(y_pred.shape) == 1:
            th = self.get_thresh(y_true, y_pred)
            return self._get_y_by_thresh(y_pred, th)
        if len(y_pred.shape) == 2:
            y = []
            for c in range(y_pred.shape[1]):
                y_proba_c = y_pred[:, c]
                y_test_c = y_true[:, c]
                th = self.get_thresh(y_test_c, y_proba_c)
                y.append(self._get_y_by_thresh(y_proba_c, th))

            return np.array(y).T


def jaccard_score(tokens_true, tokens_pred):
    intersection = set(tokens_true).intersection(set(tokens_pred))
    debug(intersection)
    union = set(tokens_true).union(set(tokens_pred))
    debug(len(union))
    return len(intersection) / len(union)


def rouge(tokens_true, tokens_pred, n: int = None):
    # if n is None than returns rouge-L
    score_type = f'rouge{n}' if n else 'rougeL'
    scorer = rouge_scorer.RougeScorer([score_type], use_stemmer=True)
    s = scorer.score(tokens_true, tokens_pred)
    return s[score_type].fmeasure


def bleu(tokens_true, tokens_pred, k=3):
    # k: [1,2,3]
    weights = [1/k] * k + (4-k) * [0]
    return sentence_bleu([tokens_true], tokens_pred, weights=weights)
