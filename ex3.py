# Erel Shtossel 316297696
import warnings
import matplotlib.pyplot as plt
import numpy as np
import math

from random import seed
from collections import Counter
import torch
from torch.nn import functional as f

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
            if word_count[w] <= 3:
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
    def __init__(self, data, cluster_num, vocab_size, epsilon=1e-5, gamma=0.1, k=10):
        self.data_size = len(data)
        self.epsilon = epsilon
        self.gamma = gamma
        self.k = k
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
        z_i = (np.log(self.p) @ self.n_t_k.T) + np.log(self.alpha).reshape(-1, 1)
        m = z_i.max(axis=0)
        z_i_m = z_i - m
        k_indices = z_i_m < -self.k
        # z_i_m_no_under_k = np.copy(z_i_m)
        # z_i_m_no_under_k[k_indices] = 0.

        w = np.exp(z_i_m) / np.exp(z_i_m).sum(axis=0)
        w[k_indices] = 0
        self.w = w

    def maximization(self):
        self.alpha = self.w.sum(axis=1) / self.data_size
        self.alpha = self.alpha.clip(min=self.epsilon)
        self.alpha = self.alpha / self.alpha.sum()

        self.p = (self.w @ self.data + self.gamma) / ((self.w @ self.n_t) + self.vocab_size * self.gamma).reshape(-1, 1)

    def likelihood(self):
        z_i = (np.log(self.p) @ self.n_t_k.T) + np.log(self.alpha).reshape(-1, 1)
        m = z_i.max(axis=0)
        z_i_m = z_i - m
        k_indices = z_i_m < -self.k

        z = np.exp(z_i_m)
        z[k_indices] = 0

        return sum(np.log(z.sum(axis=0)) + m)
    
    def plot_likelihood(self, log_likelihoods):
        # Plot log likelihoods
        plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods)
        plt.xlabel("Iteration Number")
        plt.ylabel("Log Likelihood")
        plt.title("Log Likelihood vs Iteration Number")
        plt.grid(True)
        plt.show()

    def train(self):
        log_likelihoods = []
        perplexities = []
        last_log_likelihood = 0.0
        for i in range(100):
            self.expectation()
            self.maximization()

            log_likelihood = self.likelihood()
            log_likelihoods.append(log_likelihood)
            if log_likelihood == last_log_likelihood:
                break
            print(log_likelihood)
            last_log_likelihood = log_likelihood
            perplexity =  math.exp(-1 / self.data_size * log_likelihood)
            perplexities.append(perplexity)
        self.plot_likelihood(log_likelihoods)
        


if __name__ == '__main__':
    with open("develop.txt", "r") as f:
        raw_train_data = f.readlines()

    train_text_data = raw_train_data[2::4]
    train_topics = [i.strip(">\n").split()[2:] for i in raw_train_data[::4]]

    train_text_data = clean_less_that_3(train_text_data)

    train_data, vocab_size = article_to_vectors(train_text_data)

    em = EM(train_data, 9, vocab_size, k=100,gamma=0.01)

    em.train()

    x = 3
