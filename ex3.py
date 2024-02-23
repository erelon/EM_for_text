# Erel Shtossel 316297696
import warnings

import numpy as np

from random import seed
from collections import Counter

seed(42)
np.random.seed(42)


def clean_less_that_3(data):
    word_count = Counter()
    all_worlds = [i.split() for i in data]
    for line in all_worlds:
        for w in line:
            word_count[w] += 1

    for line in all_worlds:
        remove = []
        for w in line:
            if word_count[w] < 3:
                remove.append(w)
        for r in remove:
            line.remove(r)
    return [" ".join(x) for x in all_worlds]


def article_to_vectors(data):
    word_count = Counter()
    all_worlds = [i.split() for i in data]
    for line in all_worlds:
        for w in line:
            word_count[w] += 1

    return_data = []
    order = {k: v for k, v in zip(word_count.keys(), range(len(word_count)))}
    for article in data:
        representation = np.zeros(len(word_count))
        for w in article.split():
            representation[order[w]] += 1
        return_data.append(representation)
    return return_data, len(word_count)


class EM(object):
    def __init__(self, data, cluster_num, vocab_size, epsilon=1e-5, gamma=0.1):
        self.data_size = len(data)
        self.epsilon = epsilon
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.w = np.zeros([cluster_num, len(data)])
        for e, article in enumerate(data):
            self.w[e % cluster_num][e] = 1.0

        self.p = np.zeros([cluster_num, vocab_size])
        self.alpha = np.zeros([cluster_num])
        self.data = np.array(data)
        self.n_t_k = self.data
        self.n_t = self.data.sum(axis=1)

        self.maximization()

    def expectation(self):
        # self.w[0,0]=
        denominator = (np.log(self.p) @ self.n_t_k.T) + np.log(self.alpha).reshape(-1, 1)
        x = 4
        pass

    def maximization(self):
        self.alpha = self.w.sum(axis=1) / self.data_size
        self.alpha = self.alpha.clip(min=self.epsilon)
        self.alpha = self.alpha / self.alpha.sum()

        self.p = ((self.w @ self.data + self.gamma).T / (
            (self.w * self.n_t).sum(axis=1)) + self.vocab_size * self.gamma).T

    def train(self):
        for i in range(10):
            self.expectation()
            self.maximization()


if __name__ == '__main__':
    with open("develop.txt", "r") as f:
        raw_train_data = f.readlines()

    train_text_data = raw_train_data[2::4]
    train_topics = [i.strip(">\n").split()[2:] for i in raw_train_data[::4]]

    train_text_data = clean_less_that_3(train_text_data)

    train_data, vocab_size = article_to_vectors(train_text_data)

    em = EM(train_data, 9, vocab_size)

    em.train()
