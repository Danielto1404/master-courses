import typing as tp
from collections import Counter


class SkipGramDataset:
    def __init__(self, data: tp.List[str], window_size: int):
        self.data = data
        self.window_size = window_size
        self.num_words = None
        self.words = []
        self.word2index = {}
        self.index2word = {}

    def generate(self):
        word_counts = Counter(self.data)
        self.num_words = len(word_counts.keys())
        self.words = sorted(list(word_counts.keys()), reverse=False)
        self.word2index = dict((word, i) for i, word in enumerate(self.words))
        self.index2word = dict((i, word) for i, word in enumerate(self.words))

        training_data = []
        for i, word in enumerate(self.data):

            w_target = self.onehot(word)
            w_context = []

            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i and 0 <= j < len(self.data):
                    v = self.onehot(self.data[j])
                    w_context.append(v)

            training_data.append((w_target, w_context))

        return training_data

    def onehot(self, word):
        word_vec = [0] * self.num_words
        word_index = self.word2index[word]
        word_vec[word_index] = 1
        return word_vec
