import nltk
nltk.download('punkt')
from nltk import word_tokenize

START_TOKEN = '###'
END_TOKEN = '$$$'

def tokenize_text(txt, max_len = 100):
    tok = word_tokenize(txt)
    tok = [START_TOKEN] + tok[:(max_len - 2)] + [END_TOKEN]
    return tok