import string
import re

from w2v.dataset import SkipGramDataset
from w2v.model import Word2Vec


def clean(x: str) -> str:
    translation = str.maketrans(string.punctuation, " " * len(string.punctuation))
    x = x.translate(translation)
    x = re.sub(r"\s+", " ", x.lower())
    return x


def train(data: str):
    data = clean(data).split(" ")

    dataset = SkipGramDataset(data, window_size=4)
    training_data = dataset.generate()

    w2v = Word2Vec(num_words=dataset.num_words)
    w2v.train(training_data)

    answer = {}
    for w in data:
        i = dataset.word2index[w]
        answer[w] = w2v.vector(i)

    return answer
