import numpy as np


class Word2Vec:
    def __init__(
            self,
            num_words: int,
            embedding_dim: int = 20,
            lr: float = 0.02,
            epochs: int = 40
    ):
        self.num_words = num_words
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.epochs = epochs

        self.w1 = np.random.uniform(-0.1, 0.1, (num_words, self.embedding_dim))
        self.w2 = np.random.uniform(-0.1, 0.1, (self.embedding_dim, num_words))

    def forward(self, x):
        x = np.array(x)
        h = self.w1.T @ x
        u = self.w2.T @ h
        y_c = self.softmax(u)
        return y_c, h, u

    def backward(self, e: np.ndarray, h: np.ndarray, x: np.ndarray):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, self.w2 @ e.T)

        self.w1 = self.w1 - self.lr * dl_dw1
        self.w2 = self.w2 - self.lr * dl_dw2

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        ex = np.exp(x - np.max(x))
        return ex / ex.sum(axis=0)

    def train(self, training_data):
        for i in range(self.epochs):
            for w_target, w_context in training_data:
                y_pred, h, u = self.forward(w_target)
                ei = np.sum([np.subtract(y_pred, word) for word in w_context], axis=0)
                self.backward(ei, h, w_target)

    def vector(self, index: int):
        return self.w1[index]
