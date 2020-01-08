import csv
import random
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tqdm import trange
from collections import Counter
from collections import defaultdict
try:
    import sentencepiece as spm
except ModuleNotFoundError:
    print("Please install sentencepiece")
    print("$ pip3 install sentencepiece (terminal) or !pip install sentencepiece (Colab)")
    exit()

"""
# Korean NLP

In this task, we use a Korean dataset named Naver Sentiment Movie Corpus (NSMC).
We are going to solve a binary sentiment classification problem again.
However, if we run an RNN model that we built last time, you can observe a serious overfitting problem.
To handle this overfitting problem, I suggest you to try sub-word models.
Here, we are going to use a Byte Pair Encoder and a Wordpiece model.
You have to implement some parts of a Byte Pair Encoder and a bidirectional RNN model.
See how the performance is changed according to the encoding methods.
"""


def _download_dataset(size=10000):
    assert sys.version_info.major == 3, "Use Python3"

    import ssl
    import urllib.request
    url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/koreview_{}k.csv".format(size // 1000)

    dir_path = "../data"
    file_path = os.path.join(dir_path, "koreview_{}k.csv".format(size // 1000))
    if not os.path.isfile(file_path):
        os.makedirs(dir_path, exist_ok=True)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as u, open(file_path, 'wb') as f:
            f.write(u.read())
        print("Download: {}".format(file_path))
    else:
        print("Already exist: {}".format(file_path))


def _get_review_data(path, num_samples, train_test_ratio=0.8):
    """Do not modify the code in this function."""
    _download_dataset()
    print("Load Data at {}".format(path))
    reviews, sentiments = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            reviews.append(line["review"])
            sentiments.append(int(line["sentiment"]))

    # Data shuffle
    random.seed(42)
    zipped = list(zip(reviews, sentiments))
    random.shuffle(zipped)
    reviews, sentiments = zip(*(zipped[:num_samples]))
    reviews, sentiments = np.asarray(reviews), np.asarray(sentiments)

    # Train/test split
    num_data, num_train = len(sentiments), int(len(sentiments) * train_test_ratio)
    return (reviews[:num_train], sentiments[:num_train]), (reviews[num_train:], sentiments[num_train:])


def _batch_generator(xs, ys, batch_size, shuffle=True):
    """Generate tuple (len: 2) of ndarray the shape of which is (batch_size, ?)"""
    idx = 0
    if shuffle:
        permut = np.random.permutation(ys.shape[0])
        xs, ys = xs[permut], ys[permut]
    while idx < ys.shape[0]:
        next_idx = idx + batch_size
        yield xs[idx:next_idx], ys[idx:next_idx]
        idx = next_idx


class BytePairEncoder:

    def __init__(self, n_iter_for_bpe=10, verbose=True):
        self.n_iters = n_iter_for_bpe if n_iter_for_bpe > 0 else 10
        self.units = {}
        self.sub_vocab = {}
        self.max_length = 0
        self.verbose = verbose

    def train(self, sents):
        if self.verbose:
            print('begin vocabulary scanning', end='', flush=True)

        vocabs = self._sent_to_vocabs(sents)
        if self.verbose:
            print('\rterminated vocabulary scanning', flush=True)

        self.units = self._build_subword_units(vocabs)
        self.sub_vocab = self._unit_to_vocab(self.units)

    def vocab(self):
        return self.sub_vocab

    def _unit_to_vocab(self, units):
        assert (units != {})
        vocab = {}
        index = 0
        _units = ((unit, freq) for unit, freq in units.items())
        _units = sorted(_units, key=lambda unit: unit[1], reverse=True)
        for unit, freq in _units:
            if unit not in vocab:
                vocab[unit] = index
                index += 1
        return vocab

    def _sent_to_vocabs(self, sents):
        """Make a vocab with sentences

        : param sents: (array_like of str).
            e.g., ["이거 평점이 왜 낮음?", "이거 마지막 반전"]
        : return: (dict)
            e.g. {"_ 이 거": 2, "_ 평 점 이": 1, "_ 왜": 1, "_ 낮 음 ?":1, "_ 마 지 막":1, "_ 반 전":1}
        [Hint] I recommend you to use Counter.
        """
        vocabs = Counter((token.replace('_', '') for sent in sents for token in sent.split() if token))
        return {'_ ' + ' '.join(w): c for w, c in vocabs.items() if w}

    def _build_subword_units(self, vocabs):
        """
        : param vocabs: (dict)
            e.g. {"_ 이 거": 2, "_ 평 점 이": 1, "_ 왜": 1, "_ 낮 음 ?":1, "_ 마 지 막":1, "_ 반 전":1}
        """

        def get_stats(vocabs):
            """
            : param vocabs: (dict)
                e.g. {"_ 이 거": 2, "_ 평 점 이": 1, "_ 왜": 1, "_ 낮 음 ?":1, "_ 마 지 막":1, "_ 반 전":1}
            : return: defaultdict(<class 'int'>, (dict))
                e.g. {('_', '이'): 2, ('이', 거'): 2, ('_', '평': 1, ('평', '점'): 1, ('점', '이'): 1,
                      ('_', '왜'): 1, ('_', '낮'): 1, ('낮', '음'): 1, ('음', '?'): 1, ('_', '마'): 1,
                      ('마', '지'): 1, ('지', '막'): 1, ('_', '반'): 1, ('반', '전'): 1}
            """
            pairs = defaultdict(int)
            for elem in vocabs.keys():
                val = vocabs.get(elem)
                elem = elem.split()
                for i in range(len(elem)-1):
                    pairs[(elem[i], elem[i+1])] = val
            #raise NotImplementedError
            return pairs

        def merge_vocab(pair, v_in):
            """
            : param pair: (tuple of str)
                e.g. ('이', '거')
            : param v_in: (dict)
                e.g. {"_ 이 거": 2, "_ 평 점 이": 1, "_ 왜": 1, "_ 낮 음 ?":1, "_ 마 지 막":1, "_ 반 전":1}
            : return: (dict)
                e.g. {"_ 이거": 2, "_ 평 점 이": 1, "_ 왜": 1, "_ 낮 음 ?":1, "_ 마 지 막":1, "_ 반 전":1}
            [Caution] When you merge a pair, you have to check all items in vocabulary.
            """
            v_out = {}
            #pair = pair[0]+" "+pair[1]
            pair = list(pair)
            for elem in v_in.keys():
                new_elem = ""
                val = v_in.get(elem)
                elem = elem.split()
                for i in range(len(elem)-1):
                    if elem[i:i+2] == pair:
                        new_elem += elem[i]
                    else:
                        new_elem += elem[i] + " "
                new_elem += elem[-1]
                v_out[new_elem] = val
            #raise NotImplementedError
            return v_out

        if self.verbose:
            print('\tTraining bpe started')

        for _ in trange(self.n_iters + 1):
            """
            [Hint]
            1. Generate pairs by using a get_stats function.
            2. Check if pairs is None, if true then break.
            3. Pick a maximum pair according to frequency.
               (Use pairs.get as a key for a max function)
            4. Merge the best pair in vocabs by using a merge_vocab function.
               and update vocabs.
            """

            pairs = get_stats(vocabs)
            if pairs is None:
                break
            
            max_freq = max(pairs , key = pairs.get)

            vocabs = merge_vocab(max_freq, vocabs)
            #raise NotImplementedError

        if self.verbose:
            print('\tTraining bpe was done')

        units = {}
        for word, freq in vocabs.items():
            for unit in word.split():
                units[unit] = units.get(unit, 0) + freq
        self.max_length = max((len(w) for w in units))
        return units

    def tokenize(self, s):
        return ' '.join([self._tokenize(w) for w in s.split()])

    def _tokenize(self, w):

        def initialize(w):
            w = '_' + w
            subwords = []
            n = len(w)
            for b in range(n):
                for e in range(b + 1, min(n, b + self.max_length) + 1):
                    subword = w[b:e]
                    if subword not in self.units:
                        continue
                    subwords.append((subword, b, e, e - b))
            return subwords

        def longest_match(subwords):
            matched = []
            subwords = sorted(subwords, key=lambda x: (-x[3], x[1]))
            while subwords:
                s, b, e, l = subwords.pop(0)  # str, begin, end, length
                matched.append((s, b, e, l))
                removals = []
                for i, (_, b_, e_, _) in enumerate(subwords):
                    if (b_ < e and b < e_) or (b_ < e and e_ > b):
                        removals.append(i)
                for i in reversed(removals):
                    del subwords[i]
            return sorted(matched, key=lambda x: x[1])

        subwords = initialize(w)
        subwords = longest_match(subwords)
        subwords = ' '.join([s for s, _, _, _ in subwords])
        return subwords


def train_wordpiece(sentences, voc_size_for_wordpiece):
    input_file = 'spm_input.txt'

    with open(input_file, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write('{}\n'.format(sent))

    templates = '--input={} --model_prefix={} --vocab_size={}'

    prefix = 'NSMC'
    cmd = templates.format(input_file, prefix, voc_size_for_wordpiece)

    spm.SentencePieceTrainer.Train(cmd)


def encode_text(sentences, vectorizer=None, max_len=None, msg_prefix="\n", verbose=True, encode_ver=0, bpe=None):
    """Encode array_like of strings to ndarray of integers.
    :param sentences: (array_like of str).
        e.g., ["I like apples", "Me too"]
    :param vectorizer: (CountVectorizer, optional)
    :param max_len: (int) maximum length of encoded sentences.
    :param msg_prefix:
    :param verbose:
    :return: Tuple[CountVectorizer, int, ndarray]
        e.g., (CountVectorizer,
                3,
                array([[1, 2, 3], [4, 5, 0]]))
    """
    if verbose:
        print("{} Encode texts to integers".format(msg_prefix))

    sp, vocab = None, None
    if encode_ver == 0:
        # Not recommend to modify below vectorizer/vocab lines.
        if vectorizer is None:
            vectorizer = CountVectorizer(stop_words=None)
            vectorizer.fit(sentences)
        # dictionary of (token, encoding) pair.
        #    e.g., {"I": 0, "like": 1, "apples": 2, "Me": 3, "too": 4}
        vocab = vectorizer.vocabulary_

        # Convert str to int.
        # - Use preprocess_and_tokenize the type of which is 'Callable[str, List[str]]'
        # - Do not use '0'. We will use '0' in zero padding.
        # e.g., sentences: ["I like apples", "Me too"] and
        #       vocab: {"I": 0, "like": 1, "apples": 2, "Me": 3, "too": 4}
        #       Then, encoded_sentences: [[0 + 1, 1 + 1, 2 + 1], [3 + 1, 4 + 1]] -> [[1, 2, 3], [4, 5]]
        preprocess_and_tokenize = vectorizer.build_analyzer()

    elif encode_ver == 1:
        vocab = bpe.vocab()

    elif encode_ver == 2:
        prefix = 'NSMC'
        sp = spm.SentencePieceProcessor()
        sp.Load('{}.model'.format(prefix))
        with open('{}.vocab'.format(prefix), encoding='utf-8') as f:
            vocab = [doc.strip() for doc in f]

    encoded_sentences = []
    for s in sentences:
        if encode_ver == 0:
            tokens = preprocess_and_tokenize(s)
            encoded_sentences.append(np.asarray([vocab[t] + 1 for t in tokens if t in vocab]))
        elif encode_ver == 1:
            tokens = bpe.tokenize(s)
            encoded_sentences.append(np.asarray([vocab[t] + 1 for t in tokens if t in vocab]))
        elif encode_ver == 2:
            encoded_sentences.append([index + 1 for index in sp.EncodeAsIds(s)])

    assert len(encoded_sentences) == len(sentences)
    assert all([0 not in es for es in encoded_sentences])
    # Get max_len (maximum length).
    # If max_len is given, use it.
    # e.g., [[1, 2, 3], [4, 5]] (from ["I like apples", "Me too"])
    #       -> 3
    max_len = max_len or max(len(es) for es in encoded_sentences)

    # Add zero padding to make length of all sentences the same.
    # e.g., [[1, 2, 3], [4, 5]]
    #       -> [[1, 2, 3], [4, 5, 0]]
    pad_encoded_sentences = np.zeros((len(sentences), max_len), dtype=np.int32)
    for idx, es in enumerate(encoded_sentences):
        length = len(es) if len(es) <= max_len else max_len
        pad_encoded_sentences[idx, :length] = es[:length]

    return vocab, vectorizer, max_len, pad_encoded_sentences


def _encode_one_hot_label(labels, num_classes):
    """Encode integer labels to one-hot vectors.
    :param labels: (array_like of integers)
        e.g., [0, 1, 0]
    :param num_classes: (int)
    :return: ndarray the shape of which is [len(labels), num_classes]
        e.g., [[1, 0],
               [0, 1],
               [1, 0]]
    """
    return np.eye(num_classes)[labels]


def build_model(learning_rate,
                max_len,
                num_classes,
                num_vocab,
                num_embed,
                num_hidden,
                num_lstm_cells,
                l2_lambda):
    """Build a RNN with TensorFlow"""

    # Placeholders.
    X = tf.placeholder(tf.int32, [None, max_len])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Convert integer vectors [num_batches, max_len] to embed vectors [num_batches, max_len, num_embed].
    # Use tf.nn.embedding_lookup (See https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/nn/embedding_lookup)
    embedding = tf.Variable(tf.random_uniform((num_vocab, num_embed), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, X)

    # Build forward and backward RNN cells.
    # Use tf.nn.rnn_cell.(MultiRNNCell, DropoutWrapper, GRUCell, LSTMCell, RNNCell, ...).
    #   - You do not have to use all these classes.
    #   - My personal recommendation is a combination of MultiRnnCell/DropoutWrapper/GRUCell.
    #   - See https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/nn/rnn_cell
    fw_multi_cell: tf.nn.rnn_cell.MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell([
       tf.nn.rnn_cell.DropoutWrapper(
            #tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True),
            tf.nn.rnn_cell.GRUCell(num_hidden),
            output_keep_prob=keep_prob,
       ) for _ in range(num_lstm_cells)
    ])

    bw_multi_cell: tf.nn.rnn_cell.MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell([
       tf.nn.rnn_cell.DropoutWrapper(
            #tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True),
            tf.nn.rnn_cell.GRUCell(num_hidden),
            output_keep_prob=keep_prob,
       ) for _ in range(num_lstm_cells)
    ])

    # Use a tf.nn.bidirectional_dynamic_rnn method.
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell, bw_multi_cell, embed, dtype='float32')

    # Concatenate outputs by using a tf.concat function.
    outputs = tf.concat(outputs, axis=2)
    #raise NotImplementedError

    # Use only last output, [num_batches, max_len, num_hidden] -> [num_batches, num_hidden]
    outputs = tf.transpose(outputs, [1, 0, 2])[-1]

    # Build a fully-connected layer.
    # [num_batches, num_embed] -> [num_batches, num_classes]
    weight = tf.Variable(tf.random_normal([num_hidden * 2, num_classes]))
    bias = tf.Variable(tf.random_normal([num_classes]))
    logits = tf.matmul(outputs, weight) + bias

    # Predictions, loss function, optimizer
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    cost += l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return X, Y, keep_prob, optimizer, cost, predictions


def train_model(session: tf.Session,
                X: tf.Tensor, Y: tf.Tensor, keep_prob: tf.Tensor,
                optimizer: tf.Operation, cost: tf.Tensor, predictions: tf.Tensor,
                train_xs: np.ndarray, train_ys: np.ndarray, val_xs: np.ndarray, val_ys: np.ndarray,
                batch_size: int, total_epoch: int, keep_prob_value: float,
                verbose=True):
    """Train the given RNN with train_xs, train_ys, Validate it with val_xs, val_ys."""

    for epoch in trange(total_epoch):

        train_loss = 0.

        # Train models.
        # Use _batch_generator(., ., .).
        total_pred_list, total_ys_list = [], []
        for batch_xs, batch_ys in _batch_generator(train_xs, train_ys, batch_size):
            _, loss = session.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: keep_prob_value})

            total_pred_list.append(session.run(predictions, feed_dict={X: batch_xs, keep_prob: 1.0}))
            total_ys_list.append(np.argmax(batch_ys, 1))
            train_loss += loss

        train_acc = accuracy_score(np.concatenate(total_ys_list),
                                   np.concatenate(total_pred_list))

        val_loss, val_predictions = session.run([cost, predictions], feed_dict={X: val_xs, Y: val_ys, keep_prob: 1.0})
        val_acc = accuracy_score(np.argmax(val_ys, 1), val_predictions)

        if verbose:
            print("\n[Epoch {}]".format(epoch))
            print("\t - Training Loss: {}, Training Acc: {}".format(round(train_loss, 4), round(train_acc, 4)))
            print("\t - Validation Loss: {}, Validation Acc: {}".format(round(val_loss, 4), round(val_acc, 4)))


def _test_model(session, X, keep_prob, predictions, test_xs, test_ys):
    test_predictions = session.run(predictions, feed_dict={X: test_xs, keep_prob: 1.0})
    test_acc = accuracy_score(np.argmax(test_ys, 1), test_predictions)
    return test_acc


def run(test_xs=None, test_ys=None, num_samples=10000, verbose=True,
        encode_ver=0, voc_size_for_wordpiece=2000, n_iter_for_bpe=1000):
    """You do not have to consider test_xs and test_ys, since they will be used for grading only."""

    # Data
    (train_xs, train_ys), (val_xs, val_ys) = _get_review_data(path="../data/koreview_10k.csv", num_samples=num_samples)
    if verbose:
        print("\n[Example of xs]: [\"{}...\", \"{}...\", ...]\n[Example of ys]: [{}, {}, {}, {}, {}, ...]".format(
            train_xs[0][:70], train_xs[1][:70], *train_ys[:5]))
        print("\n[Num Train]: {}\n[Num Test]: {}".format(len(train_ys), len(val_ys)))
        print("\n[# of 1 in Training]: {}\n[# of 0 in Training]: {}".format(sum(train_ys == 1), sum(train_ys == 0)))

    bpe = None
    if encode_ver == 1:
        bpe = BytePairEncoder(n_iter_for_bpe=n_iter_for_bpe)
        bpe.train(train_xs)

    elif encode_ver == 2:
        train_wordpiece(train_xs, voc_size_for_wordpiece=voc_size_for_wordpiece)

    # Encode text (train set) to integer ndarray.
    vocab, vectorizer, max_length, train_xs = encode_text(train_xs, msg_prefix="\n[Train]", verbose=verbose,
                                                          encode_ver=encode_ver, bpe=bpe)
    train_ys = _encode_one_hot_label(train_ys, num_classes=2)
    assert train_xs.shape[1] == max_length

    # Encode text (validation set) to integer ndarray.
    _, _, _, val_xs = encode_text(val_xs, vectorizer=vectorizer, max_len=max_length,
                                  msg_prefix="\n[Val]", verbose=verbose, encode_ver=encode_ver, bpe=bpe)
    val_ys = _encode_one_hot_label(val_ys, num_classes=2)

    # Build a RNN model.
    # You can change hyper-parameters (kwargs) except max_len, num_classes, num_vocab.
    X_placeholder, Y_placeholder, keep_prob_placeholder, adam_opt, ce_loss, preds = build_model(
        learning_rate=0.01,
        max_len=max_length,  # Do not change.
        num_classes=2,  # Do not change.
        num_vocab=len(vocab) + 1,  # Do not change.
        num_embed=128,
        num_hidden=64,
        num_lstm_cells=2,
        l2_lambda=0.005,
    )

    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train
    # You can change hyper-parameters (kwargs).
    train_model(sess,
                X_placeholder, Y_placeholder, keep_prob_placeholder, adam_opt, ce_loss, preds,
                train_xs, train_ys, val_xs, val_ys,
                batch_size=512, total_epoch=25, keep_prob_value=0.75)

    if test_xs is not None:
        _, _, _, test_xs = encode_text(test_xs, vectorizer=vectorizer, max_len=max_length,
                                       msg_prefix="\n[Test]", verbose=verbose)
        test_ys = _encode_one_hot_label(test_ys, num_classes=2)
        test_acc = _test_model(sess, X_placeholder, keep_prob_placeholder, preds, test_xs, test_ys)
        return {"variables": tf.trainable_variables(), "test_accuracy": test_acc}
    else:
        return {"variables": tf.trainable_variables()}


if __name__ == '__main__':

    #change to 0, 1, 2
    ENCODE_VER = 1

    tf.reset_default_graph()

    # Spacing by ' ' (Blank)
    if ENCODE_VER == 0:
        run(encode_ver=0)

    # Byte Pair Encoding: https://en.wikipedia.org/wiki/Byte_pair_encoding
    elif ENCODE_VER == 1:
        run(encode_ver=1, n_iter_for_bpe=1000)

    # WordPiece: https://github.com/google/sentencepiece
    elif ENCODE_VER == 2:
        run(encode_ver=2, voc_size_for_wordpiece=2000)

    else:
        raise ValueError
