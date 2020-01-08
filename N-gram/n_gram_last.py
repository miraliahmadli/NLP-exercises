import nltk
from itertools import product
import math
from pathlib import Path
import os
import sys

SOS = "<s>"
EOS = "</s>"
UNK = "<UNK>"

"""
# N-gram

See docstrings for the details.

## Instruction

1. Implement add_sentence_tokens
2. Implement replace_unknown
3. Implement _convert_oov
4. Implement perplexity
5. Implement _best_candidate
"""


def _download_dataset():
    assert sys.version_info.major == 3, "Use Python3"

    import ssl
    import urllib.request
    for file_type in ["train", "test"]:
        file_name = "ngram_{}.txt".format(file_type)
        url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/{}".format(file_name)
        dir_path = "../data"
        file_path = os.path.join(dir_path, file_name)
        if not os.path.isfile(file_path):
            os.makedirs(dir_path, exist_ok=True)
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ctx) as u, open(file_path, 'wb') as f:
                f.write(u.read())
            print("Download: {}".format(file_path))
        else:
            print("Already exist: {}".format(file_path))


def load_data(data_dir):
    """ Load train and test corpora from a directory.
        Directory must contain two files: train.txt and test.txt.
        Newlines will be stripped out.

        :param data_dir: (Path) pathlib.Path of the directory to use.
        :return: The train and test sets, as lists of sentences.
    """
    train_path = data_dir.joinpath('ngram_train.txt').absolute().as_posix()
    test_path = data_dir.joinpath('ngram_test.txt').absolute().as_posix()

    with open(train_path, 'r') as f:
        train = [l.strip() for l in f.readlines()]
    with open(test_path, 'r') as f:
        test = [l.strip() for l in f.readlines()]
    return train, test


def add_sentence_tokens(sentences, n):
    """ Wrap each sentence in SOS and EOS tokens.
        For n >= 2, n-1 SOS tokens are added, otherwise only one is added.

        :param sentences" (list of str) the sentences to wrap
        :param n: (int) order of the n-gram model which will use these sentences.
        :return: list of sentences with SOS and EOS tokens wrapped around them.

        [Example]
        sentences = ['don't put off until tomorrow', 'what you can do today']
        n = 3
        Then return should be
        ['<s> <s> don't put off until tomorrow </s> </s>', '<s> <s> what you can do today </s> </s>']
    """
    start  = (SOS + " ")*max(n-1,1)
    end = (" " + EOS)*max(n-1,1)
    ans = [(start + sentence + end) for sentence in sentences]
    #print(ans)
    return ans


def replace_unknown(tokens, cut_off):
    """ Replace tokens which appear (<= cutoff) times in the corpus with <UNK>.

        :param tokens: (list of str) the tokens comprising the corpus.
        :param cut_off: (int) the bound of cutting the token.
        :return: The same list of tokens with each singleton replaced by <UNK>.

        [Hint]
        use nltk.FreqDist when you build a vocab.

        [Example]
        tokens = ['<s>', 'I', 'love', 'cake', 'I', 'like', 'cake', '</s>']
        Then, here are the counts for each words in tokens:
        'I': 2, 'love': 1, 'cake': 2, 'like': 1
        and cutoff = 1
        Then output should be
        ['<s>', 'I', UNK, 'cake', 'I', UNK, 'cake', '</s>']
    """
    from nltk.probability import FreqDist
    fdist = FreqDist(tokens)
    #for word in tokens:
    #    fdist[word.lower()] += 1
    tokens = [UNK if word!=SOS and word!=EOS and fdist.get(word)<=cut_off else word for word in tokens]
    return tokens


def preprocess(sentences, n, cut_off):
    """ Add SOS/EOS/UNK tokens to given sentences and tokenize.

        :param sentences: (list of str) the sentences to preprocess
        :param n: (int) order of the n-gram model which will use these sentences.
        :param cut_off: (int) the bound of cutting the token.
        :return: The preprocessed sentences, tokenized by words.
    """
    sentences = add_sentence_tokens(sentences, n)
    tokens = ' '.join(sentences).split(' ')
    tokens = replace_unknown(tokens, cut_off)
    return tokens


class LanguageModel(object):

    def __init__(self, train, n, laplace=1):
        """ An n-gram language model trained on a given corpus.

            For a given n and given training corpus, constructs an n-gram language model for the corpus by:
            1. preprocessing the corpus (adding SOS/EOS/UNK tokens)
            2. calculating (smoothed) probabilities for each n-gram
            Also contains methods for calculating the perplexity of the model
            against another corpus, and for generating sentences.

            :param train: (list of str) list of sentences comprising the training corpus.
            :param n: (int) the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
            :param laplace: (int or float) lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).
        """
        self.n = n
        self.laplace = laplace
        self.tokens = preprocess(train, n, 1)
        self.vocab = nltk.FreqDist(self.tokens)
        self.model = self._create_model()

        """
        self.masks = list(reversed(list(product((0, 1), repeat=2))))
        >>> [(1, 1), (1, 0), (0, 1), (0, 0)] 

        self.masks = list(reversed(list(product((0, 1), repeat=3))))
        >>> [(1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0), (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
        """
        self.masks = list(reversed(list(product((0, 1), repeat=n))))

    def check_ngram_freqdist(self):
        n_grams = list(nltk.ngrams(self.tokens, self.n))
        n_vocab = nltk.FreqDist(n_grams)
        freqdist = [(k, v) for k, v in n_vocab.items()]

        return n_grams, freqdist

    def _smooth(self):
        """ Apply Laplace smoothing to n-gram frequency distribution.

            Here, n_grams refers to the n-grams of the tokens in the training corpus,
            while m_grams refers to the first (n-1) tokens of each n-gram.

            :return: Mapping of each n-gram (tuple of str) to its Laplace-smoothed probability (float).
        """
        vocab_size = len(self.vocab)

        n_grams = nltk.ngrams(self.tokens, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens, self.n - 1)
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count + self.laplace) / (m_count + self.laplace * vocab_size)

        return {n_gram: smoothed_count(n_gram, count) for n_gram, count in n_vocab.items()}

    def _create_model(self):
        """ Create a probability distribution for the vocabulary of the training corpus.

            If building a unigram model, the probabilities are simple relative frequencies
            of each token with the entire corpus.
            Otherwise, the probabilities are Laplace-smoothed relative frequencies.

            :return: A dict mapping each n-gram (tuple of str) to its probability (float).
        """
        if self.n == 1:
            num_tokens = len(self.tokens)
            return {(unigram,): count / num_tokens for unigram, count in self.vocab.items()}
        else:
            return self._smooth()

    def _convert_oov(self, ngram):
        """ Used in test case to minimize loss of information and to handle out of vocabulary problems.

            Convert, if necessary, a given n-gram to one which is known by the model.
            Starting with the unmodified ngram, check each possible permutation of the n-gram
            with each index of the n-gram containing either the original token or <UNK>. Stop
            when the model contains an entry for that permutation.
            This is achieved by creating a 'bitmask' for the n-gram tuple, and swapping out
            each flagged token for <UNK>. Thus, in the worst case, this function checks 2^n
            possible n-grams before returning.

            :return: The n-gram with <UNK> tokens in certain positions such that the model
                     contains an entry for it.

            [Hint]
            1. Use mask function I gave to you.
            2. Use self.masks, self.model

            [Example]
            Let's consider a bigram case.

            unmodified ngram = ('the', 'company')
            self.masks = [(1, 1), (1, 0), (0, 1), (0, 0)] 

            Check each possible permutation of the n-gram with each index of the n-gram
            containing either the original token or <UNK>
            [('the', 'company'), ('the', '<UNK>'), ('<UNK>', 'company'), ('<UNK>', '<UNK>')]            
            
            If self.model contains ('the', 'company'), then return ('the', 'company')
            If self.model does not contain ('the', 'company'), check whether ('the', '<UNK>') is
            contained in self.model. If true return ('the', '<UNK>'). If not check whether
            ('<UNK>', 'company') is contained in self.model. If true return ('<UNK>', 'company).
            If not check whether ('<UNK>', '<UNK>') is contained in self.model.
            If true return ('<UNK>', '<UNK>').
        """
        def mask(ngram, bitmask):
            return tuple((token if flag == 1 else "<UNK>" for token, flag in zip(ngram, bitmask)))
        
        for maska in self.masks:
            perm = mask(ngram, maska)
            if self.model.get(perm):
                return perm
        

    def perplexity(self, test):
        """ Calculate the perplexity of the model against a given test corpus.

            :param test: (list of str) sentences comprising the training corpus.

            :return: The perplexity of the model as a float.

            [Hint]
            1. You have to preprocess the test_data. After that test_tokens will be generated.
            2. Create a new n-gram model with a nltk.ngrams function.
            3. Calculate the length of test_tokens
            4. Use _convert_oov function to convert test_ngrams
            5. Use self.model to calculate probabilities
            6. Use math.exp, math.log, sum(), map()
            7. perplexity = exp(-1/N * sum( log prob ) )
        """
        test_tokens = preprocess(test, self.n, 1)
        test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)
        test_ngrams = [self._convert_oov(test_ngram) for test_ngram in test_ngrams]
        perplexity = math.exp(-1/N * sum(map(lambda elem: math.log(self.model.get(elem)),test_ngrams)))
        return perplexity
        #raise NotImplementedError

    def _best_candidate(self, prev, i, without=None):
        """ Choose the most likely next token given the previous (n-1) tokens.
            If selecting the first word of the sentence (after the SOS tokens),
            the i'th best candidate will be selected, to create variety.
            If no candidates are found, the EOS token is returned with probability 1.

            :param prev (tuple of str): the previous n-1 tokens of the sentence.
            :param i (int): which candidate to select if not the most probable one.
            :param without (list of str): tokens to exclude from the candidates list.

            :return: A tuple with the next most probable token and its corresponding probability.

            [Hint]
            For a given prev X1 ~ Xn-1 
            1. Add <UNK> to without list.
	        2. Find the ngrams (candidates) where the first n-1 tokens are same as prev.
               ((Xn, prob) for ngram, prob in self.model.items() ~~~~)
            3. filter out some (Xn, prob) when Xn is in without
               -Use filter()
            4. sort candidates
               -Use sorted()
            5. return one which has the highest probability
               - You have to consider the case when len(candidates) == 0
               - You also have to consider the case when prev == () or prev[-1] == "<s>"
        """
        without = without or []
        without.append(UNK)
        def fun(candidate):
            #print(candidate)
            return not(candidate[0] in without)
        candidates = []
        for ngram, prob in self.model.items():
            if prev == ngram[:-1]:
                candidates.append((ngram[-1],prob))
        #candidates = ((ngram[-1],prob) for ngram, prob in self.model.items())
        candidates = filter(fun, candidates)
        candidates = sorted(candidates, key=lambda x: x[1])
        candidates.reverse()
        #print("Candidates: ",candidates)
        if len(candidates)==0:
            #print(EOS)
            return EOS,1
        elif prev == () or prev[-1] == SOS:
            #print(candidates[i])
            return candidates[i]
        else:
            #print(candidates[-1])
            return candidates[0]
        #raise NotImplementedError

    def generate_sentences(self, num, min_len=12, max_len=24):
        """ Generate num random sentences using the language model.
            Sentences always begin with the SOS token and end with the EOS token.
            While unigram model sentences will only exclude the UNK token, n>1 models
            will also exclude all other words already in the sentence.

            :param num: (int) the number of sentences to generate.
            :param min_len: (int) minimum allowed sentence length.
            :param max_len: (int) maximum allowed sentence length.

            :yield: A tuple with the generated sentence and the combined probability
                    (in log-space) of all of its n-grams.
        """
        for i in range(num):
            sent, prob = ["<s>"] * max(1, self.n - 1), 1
            while sent[-1] != "</s>":
                prev = () if self.n == 1 else tuple(sent[-(self.n - 1):])
                blacklist = sent + (["</s>"] if len(sent) < min_len else [])
                next_token, next_prob = self._best_candidate(prev, i, without=blacklist)
                sent.append(next_token)
                prob *= next_prob

                if len(sent) >= max_len:
                    sent.append("</s>")

            yield ' '.join(sent), -1 / math.log(prob)


def run_language_model(train, test, N, Laplace, GenNum):
    print("Loading {}-gram model...".format(N))
    lm = LanguageModel(train, N, laplace=Laplace)
    print("Vocabulary size: {}".format(len(lm.vocab)))
    bigram, freqdist = lm.check_ngram_freqdist()

    print("10 examples of bigram")
    print(bigram[:10])

    print("10 examples of FreqDist")
    print(freqdist[:10])

    print("Generating sentences...")
    for sentence, prob in lm.generate_sentences(GenNum):
        print("{} ({:.5f})".format(sentence, prob))

    perplexity = lm.perplexity(test)
    print("Model perplexity: {:.3f}".format(perplexity))
    print("")


if __name__ == '__main__':
    # Load and prepare train/test data
    _download_dataset()
    train_data, test_data = load_data(Path('../data/'))

    run_language_model(train_data, test_data, 1, 0.01, 10)

    run_language_model(train_data, test_data, 2, 0.01, 10)

    run_language_model(train_data, test_data, 3, 0.01, 10)

    run_language_model(train_data, test_data, 4, 0.01, 10)

    run_language_model(train_data, test_data, 5, 0.01, 10)
