import torch
import numpy as  np
from collections import Counter

from src.modeling.tokenizer import  START_TOKEN, END_TOKEN

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenized_text,
            sequence_length=10,
            max_len=100,
            **kwargs
    ):
        self.sequence_length = sequence_length
        self.max_len = max_len

        self.tokenized_text = tokenized_text
        self.lengths = np.minimum(self.tokenized_text.apply(len), max_len)
        #         self.text = self.detokenize(tokenized_text)
        self.uniq_words = self.get_uniq_words(self.tokenized_text)
        print(f'Started dataset with {len(self.uniq_words)} words')

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.X = self.get_X()

    #     @staticmethod
    #     def detokenize(tokenized_text):
    #         return tokenized_text.apply(lambda l : ' '.join(l))

    @staticmethod
    def get_uniq_words(tokenized):
        all_words = [e for l in tokenized for e in l]
        word_counts = Counter(all_words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def get_X(self):
        X = []
        for message in self.tokenized_text:
            # for padding in beggining
            m = [START_TOKEN] * self.sequence_length
            m += message
            m += [END_TOKEN] * max(self.max_len - len(message), 0)
            m = [self.word_to_index[word] for word in m]
            X.append(m)
        return np.array(X, dtype='int32')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, massage_idx):
        massage = self.X[massage_idx]
        massage_len = self.lengths[massage_idx]
        return massage, massage_len
        # for padding
#         word_start = self.sequence_length + np.random.randint(-self.sequence_length, massage_len - self.sequence_length)

#         seq_length = self.sequence_length
#         word_end = word_start + seq_length
#         start = 0
#         if word_start < 0:
#             start = 10 - word_start

#         return (
#             torch.tensor(massage[word_start:(word_start + self.sequence_length)], dtype = torch.long),
#             torch.tensor(massage[word_start + self.sequence_length], dtype = torch.long),
#         )
