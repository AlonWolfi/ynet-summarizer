import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from config import DATA_DIR
from extras.nlp.tokenizer import tokenize_text, START_TOKEN, END_TOKEN
from utils import read_data, read_chat
from .dataset import Dataset


def predict(model, index_to_word, word_to_index, text, sequence_length=5, next_words=100):
    model.eval()
    words = text.split(' ')
    start = 0
    if len(words) < sequence_length:
        start = sequence_length - len(words)
        words = [START_TOKEN] * (sequence_length - len(words)) + words
    words = words[-sequence_length:]

    for i in range(0, next_words):
        x = torch.tensor([[word_to_index[w] for w in words[-sequence_length:]]])
        y_pred = model(x)  # , randomize = True)

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(index_to_word[word_index])
        if words[-1] == END_TOKEN:
            return words[start:-1]

    return words


@st.cache
def get_w2i():
    text = read_chat()['text']
    args = {
        'max_epochs': 5,
        'batch_size': 256,
        'sequence_length': 5,
        'max_len': 100,

    }
    tokenized = text.apply(tokenize_text)
    dataset = Dataset(tokenized, **args)
    return dataset.index_to_word, dataset.word_to_index


def complete_sentence(sentence):
    index_to_word, word_to_index = get_w2i()
    model = read_data(DATA_DIR / 'models' / 'massage_model.pickle')
    answers = []
    for i in range(10):
        answers.append(' '.join(predict(model, index_to_word, word_to_index, text=sentence)))
    return answers
