import argparse
import re

import numpy as np
import queue
import pdb
import os
from tqdm import trange, tqdm
import sys
from copy import deepcopy

def _download_dataset(file_path):
    import sys
    assert sys.version_info.major == 3, "Use Python3"

    import urllib.request
    url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/korea.txt"

    import os
    dir_path = os.path.dirname(file_path)
    if not os.path.isfile(file_path):
        os.makedirs(dir_path, exist_ok=True)
        urllib.request.urlretrieve(url, file_path)
        print("Download: {}".format(file_path))
    else:
        print("Already exist: {}".format(file_path))


def preprocess(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z]+', ' ', sentence)
    return sentence


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Vocabulary:

    """Do not modify this class"""

    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2count = {}
        self.num_words = 0
        self.total_words = 0

    def init_dict(self, sentences, min_count, max_count):
        for line in sentences:
            self.add_sentence(line)

        self.trim(min_count, max_count)

        for (k, c) in self.word2count.items():
            self.total_words += c

    def add_sentence(self, sentence):
        sentence = preprocess(sentence)
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.index2count[self.num_words] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1
            self.index2count[self.word2index[word]] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count, max_count):
        if self.trimmed:
            return
        self.trimmed = True
        q = queue.PriorityQueue()
        keep_words = 0
        for (k, v) in self.word2count.items():
            if v >= min_count:
                keep_words += 1
                q.put((v, k))

        print('Words to Keep: {} / {} = {:.2f}%'.format(
            keep_words, len(self.word2index), 100 * keep_words / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2count = {}
        self.num_words = 0

        while not q.empty():
            freq, word = q.get()
            if freq < max_count:
                for _ in range(freq):
                    self.add_word(word)


class SkipGram:

    def __init__(self, vocab, embedding_dimension):
        self.sentences = []
        self.vocab = vocab
        self.embed_dim = embedding_dimension
        self.W = None
        self.W_prime = None
        self.table = []

    def init_unigram_table(self):
        table_size = 1e8
        pow_frequency = np.array(list(self.vocab.index2count.values())) ** 0.75
        word_pow_sum = np.sum(pow_frequency)
        ratio_array = pow_frequency / word_pow_sum
        word_count_list = np.around(ratio_array * table_size)
        for word_index, word_freq in enumerate(tqdm(word_count_list)):
            self.table.append(np.asarray([word_index] * int(word_freq)))
        self.table = np.concatenate(self.table)

    def save_embedding(self, file_name):

        embedding = self.W

        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(self.vocab.index2word), self.embed_dim))
        for (w_id, word) in self.vocab.index2word.items():
            e = embedding[w_id]
            fout.write('%s %s\n' % (word, " ".join(map(lambda x: str(x), e))))

        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            embedding = TSNE(n_components=2).fit_transform(embedding)
            for (w_id, word) in self.vocab.index2word.items():
                x, y = embedding[w_id]
                plt.scatter(x, y)
                plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
            plt.show()
        except ModuleNotFoundError:
            print("Please install matplotlib to see visualization.")

    def forward(self, index1, index2, W, W_prime):
        """
        Implement forward!
        :param index1: (int) : index of W matrix
        :param index2: (int) : index of W_prime matrix
        :param W: [vocab_dim, embedding_dim] (np.ndarray)  : W matrix
        :param W_prime: [vocab_dim, embedding_dim] (np.ndarray) : W_prime matrix
        :return: (float)
        """
        return np.dot(W[index1],W_prime[index2])

    def backward(self, forwards, label):
        """
        Calculate gradients from a Negative Sampling objective function!
        I give you some codes which handles saturation problems.
        :param forwards: (float) : the result from the forward
        :param label: (int) : whether the word is positive or negative
        :return: (float) : gradient
        """
        bound = 6
        if forwards > bound:
            gradient = (label - 1)
        elif forwards < - bound:
            gradient = (label - 0)
        else:
            gradient = (label - sigmoid(forwards))
        return gradient

    def optimize(self, learning_rate, gradients, W, W_prime):
        """
        Implement stochastic gradient descent algorithm!
        You should update the value of W and W_prime matrix!
        :param learning_rate: (float) : learning rate
        :param gradients: (list) : list of (gradient (float), center index (int), context index (int)) triples
        :param W: [vocab_dim, embedding_dim] (np.ndarray) : W matrix
        :param W_prime: [vocab_dim, embedding_dim] (np.ndarray) : W_prime matrix
        There's no return
        """
        W_prime_original = deepcopy(W_prime)

        # Update W_prime with W and grad.
        for grad, center_i, context_i in gradients:
            W_prime[context_i] = W_prime[context_i] + learning_rate*grad*np.array(W[center_i])

        # Update W with W_prime_original and grad.
        for grad, center_i, context_i in gradients:
            W[center_i] = W[center_i] + learning_rate*grad*np.array(W_prime_original[context_i])
        
    def CalculateProb(self, t, f):
        return t/f + (t/f)**(1/2)

    def subsampling(self, sample_bound, sentence):
        """w_cnt
        Implement subsampling!
        Input sentence is a naive string!
        You should do some works to split a sentence into a array of words
        :param sample_bound: (float): scale something check README.md!
        :param sentence: (string)
        :return ret, w_cnt
        - ret: [word_num] (list) : a subsampled sentence
        - w_cnt: (int) :number of words in a subsampled sentence
        """
        
        #count = self.vocab.index2count
        #sentence = preprocess(sentence)
        ret = []
        w_cnt = 0
        sentence = sentence.split(" ")
        for word in sentence:
            if word in self.vocab.word2index:
                prob = self.CalculateProb(sample_bound, self.vocab.word2count[word]/self.vocab.total_words)
                random = np.random.random()
                if prob >= random:
                    ret.append(self.vocab.word2index[word])
                    w_cnt += 1
        return ret, w_cnt
        
        '''
        sampled_word_indices = []
        for word in sentence.strip().split(' '):
            if self.vocab.word2count.get(word) is None:
                continue
            else:
                raise NotImplementedError
        return sampled_word_indices
        '''

    def train(self, input_file_name, output_file_name,
              total_epoch, learning_rate, min_count, max_count, window_size, num_negative, sample_bound,
              debug):
        np.random.seed(6)

        if debug:
            pdb.set_trace()

        print('Starting training using file ', input_file_name)
        input_file = open(input_file_name, 'r', encoding="utf-8")

        # Read sentences from a input file
        self.sentences = input_file.readlines()

        # Initialize a vocabulary with a training corpus
        self.vocab.init_dict(self.sentences, min_count, max_count)

        # Also construct the unigram language model
        print("\nInit Unigram Table")
        self.init_unigram_table()

        # Initialize weights
        low = -0.5 / self.embed_dim
        high = 0.5 / self.embed_dim
        self.W = np.random.uniform(low, high, (self.vocab.num_words, self.embed_dim))
        self.W_prime = np.zeros((self.vocab.num_words, self.embed_dim))

        w_count = 0

        print("\nRunning...")
        for _ in trange(total_epoch):
            for sentence in self.sentences:
                sentence = preprocess(sentence)
                line, w_cnt = self.subsampling(sample_bound, sentence)
                w_count += w_cnt

                line_pos = 0

                for word_idx in line:

                    soft_slide = np.random.randint(window_size, size=1).item()
                    start_idx = max(line_pos - window_size + soft_slide, 0)
                    end_idx = line_pos + window_size + 1 - soft_slide

                    for center_idx in line[start_idx:end_idx]:

                        if self.vocab.index2count.get(center_idx) is None:
                            continue
                        if center_idx == word_idx:
                            continue
                        center_idx = int(center_idx)

                        gradients = []
                        for neg_sample in range(num_negative + 1):
                            if neg_sample == 0:
                                context_idx = word_idx
                                label = 1
                            else:
                                rand = np.random.randint(int(len(self.table)), size=1).item()
                                context_idx = int(self.table[rand])
                                if context_idx == 0:
                                    context_idx = np.random.randint(self.vocab.num_words, size=1).item()
                                if context_idx == word_idx:
                                    continue
                                label = 0
                            ff = self.forward(center_idx, context_idx, self.W, self.W_prime)
                            gradients.append((self.backward(ff, label), center_idx, context_idx))

                        self.optimize(learning_rate, gradients, self.W, self.W_prime)

                    line_pos += 1

        print("\nSave embedding at {}".format(output_file_name))
        self.save_embedding(output_file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parser for SkipGram Negative Sampling')
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--input-file-name", type=str, default="../data/korea.txt")
    parser.add_argument("--output-file-name", type=str, default="./embedding_results.txt")
    parser.add_argument("--total-epoch", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--min-count", type=int, default=7, help="Take words that appear more than min_count")
    parser.add_argument("--max-count", type=int, default=100000, help="Take words that appear less than max_count")
    parser.add_argument("--window-size", type=int, default=10, help="Size of window")
    parser.add_argument("--num-negative", type=int, default=15, help="Number of negative samples")
    parser.add_argument("--sample-bound", type=float, default=1e-5, help="Sampling bound for subsampling")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args(args=[])

    _download_dataset(args.input_file_name)
    os.makedirs(os.path.dirname(args.output_file_name), exist_ok=True)

    voc = Vocabulary()
    skip = SkipGram(voc, args.embedding_dim)
    skip.train(
        input_file_name=args.input_file_name,
        output_file_name=args.output_file_name,
        total_epoch=args.total_epoch,
        learning_rate=args.learning_rate,
        min_count=args.min_count,
        max_count=args.max_count,
        window_size=args.window_size,
        num_negative=args.num_negative,
        sample_bound=args.sample_bound,
        debug=args.debug,
    )
