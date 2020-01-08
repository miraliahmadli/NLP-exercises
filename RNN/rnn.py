import csv
import random
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tqdm import trange

"""
# RNN

In this task, you will implement a RNN model for binary sentiment classification.

Implement three methods:
- encode_text(sentences, vectorizer, max_len, msg_prefix, verbose) -> vectorizer, max_len, pad_encoded_sentences
- build_model(learning_rate, max_len, num_classes, num_vocab, num_embed, num_hidden, num_lstm_cells, l2_lambda)
    -> X, Y, keep_prob, optimizer, cost, predictions
- train_model(session, X, Y, keep_prob, optimizer, cost, predictions,
              train_xs, train_ys, val_xs, val_ys, batch_size, total_epoch, keep_prob_value, verbose) -> None

## Instruction

* See skeleton codes below for more details.
* Do not remove assert lines and do not modify methods that start with an underscore.

## Important Notes
* TF 2.0 has been released recently, but our task is designed with TF 1.4
* Our code is compatible to 1.5 which is the default version of current Google Colab.
* We highly recommend using Google Colab with GPU for students who have not a GPU in your local or remote computer.
    - Runtime > Change runtime type > Hardware accelerator: GPU
* For one epoch of training, Colab+GPU takes 35s, Colab+CPU takes 165s, and local laptop (TA's) takes 140s.
* TA's code got 82-84 validation accuracy in a total of 545-555s (15 epochs) at Colab+GPU.

"""

tf.set_random_seed(2019)
np.random.seed(2019)
random.seed(2019)


def _download_dataset(size=10000):
    assert sys.version_info.major == 3, "Use Python3"

    import ssl
    import urllib.request
    url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/review_{}k.csv".format(size // 1000)

    dir_path = "../data"
    file_path = os.path.join(dir_path, "review_{}k.csv".format(size // 1000))
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


def encode_text(sentences, vectorizer=None, max_len=None, msg_prefix="\n", verbose=True):
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

    # Not recommend to modify below vectorizer/vocab lines.
    if vectorizer is None:
        vectorizer = CountVectorizer(stop_words="english")
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
    encoded_sentences = []
    for s in sentences:
        tokens = preprocess_and_tokenize(s)
        # Hint: encoded_sentences.append(/* BLANK */)
        encoded_sentences.append([vocab.get(token,0)+1 for token in tokens])
        #raise NotImplementedError

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
        # Hint: pad_encoded_sentences[idx, :length] = /* BLANK */
        pad_encoded_sentences[idx, :length] = encoded_sentences[idx]
        #raise NotImplementedError

    return vectorizer, max_len, pad_encoded_sentences


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

    # Build RNN cells.
    # Use tf.nn.rnn_cell.(MultiRNNCell, DropoutWrapper, GRUCell, LSTMCell, RNNCell, ...).
    #   - You do not have to use all these classes.
    #   - My personal recommendation is a combination of MultiRnnCell/DropoutWrapper/GRUCell.
    #   - See https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/nn/rnn_cell

    # Hint:
    # cell = tf.nn.rnn_cell.MultiRNNCell([
    #    tf.nn.rnn_cell.DropoutWrapper(
    #
    #       /* BLANK */
    #
    #    ) for _ in range(num_lstm_cells)
    # ])
    cell = tf.nn.rnn_cell.MultiRNNCell([
       tf.nn.rnn_cell.DropoutWrapper(
            #tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True),
            tf.nn.rnn_cell.GRUCell(num_hidden),
            output_keep_prob=keep_prob,
       ) for _ in range(num_lstm_cells)
    ])

    #cell.summary()
    #raise NotImplementedError

    outputs, states = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)

    # Use only last output, [num_batches, max_len, num_hidden] -> [num_batches, num_hidden]
    outputs = tf.transpose(outputs, [1, 0, 2])[-1]

    # Build a fully-connected layer.
    # [num_batches, num_embed] -> [num_batches, num_classes]
    weight = tf.Variable(tf.random_normal([num_hidden, num_classes]))
    bias = tf.Variable(tf.random_normal([num_classes]))
    logits = tf.matmul(outputs, weight) + bias# weight*outputs + bias  # Use weight and bias
    #raise NotImplementedError

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
        for batch_xs, batch_ys in _batch_generator(train_xs, train_ys, batch_size):
            # Hint: _, loss = session.run([/* BLANK */], feed_dict={/* BLANK */})
            _, loss = session.run([optimizer, cost], feed_dict={X : batch_xs, Y : batch_ys, keep_prob : keep_prob_value})
            #raise NotImplementedError

        val_loss, val_predictions = session.run([cost, predictions], feed_dict={X: val_xs, Y: val_ys, keep_prob: 1.0})
        val_acc = accuracy_score(np.argmax(val_ys, 1), val_predictions)

        if verbose:
            print("\n[Epoch {}]".format(epoch))
            print("\t - Training Loss: {}".format(round(train_loss, 4)))
            print("\t - Validation Loss: {}, Validation Acc: {}".format(round(val_loss, 4), round(val_acc, 4)))


def _test_model(session, X, keep_prob, predictions, test_xs, test_ys):
    test_predictions = session.run(predictions, feed_dict={X: test_xs, keep_prob: 1.0})
    test_acc = accuracy_score(np.argmax(test_ys, 1), test_predictions)
    return test_acc


def run(test_xs=None, test_ys=None, num_samples=10000, verbose=True):
    """You do not have to consider test_xs and test_ys, since they will be used for grading only."""

    # Data
    (train_xs, train_ys), (val_xs, val_ys) = _get_review_data(path="../data/review_10k.csv", num_samples=num_samples)
    if verbose:
        print("\n[Example of xs]: [\"{}...\", \"{}...\", ...]\n[Example of ys]: [{}, {}, ...]".format(
            train_xs[0][:70], train_xs[1][:70], train_ys[0], train_ys[1]))
        print("\n[Num Train]: {}\n[Num Test]: {}".format(len(train_ys), len(val_ys)))

    # Encode text (train set) to integer ndarray.
    vectorizer, max_length, train_xs = encode_text(train_xs, msg_prefix="\n[Train]", verbose=verbose)
    train_ys = _encode_one_hot_label(train_ys, num_classes=2)
    assert train_xs.shape[1] == max_length

    # Encode text (validation set) to integer ndarray.
    _, _, val_xs = encode_text(val_xs, vectorizer=vectorizer, max_len=max_length,
                               msg_prefix="\n[Val]", verbose=verbose)
    val_ys = _encode_one_hot_label(val_ys, num_classes=2)

    # Build a RNN model.
    # You can change hyper-parameters (kwargs) except max_len, num_classes, num_vocab.
    X_placeholder, Y_placeholder, keep_prob_placeholder, adam_opt, ce_loss, preds = build_model(
        learning_rate=0.01,
        max_len=max_length,  # Do not change.
        num_classes=2,  # Do not change.
        num_vocab=len(vectorizer.vocabulary_) + 1,  # Do not change.
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
                batch_size=512, total_epoch=15, keep_prob_value=0.75)

    if test_xs is not None:
        _, _, test_xs = encode_text(test_xs, vectorizer=vectorizer, max_len=max_length,
                                    msg_prefix="\n[Test]", verbose=verbose)
        test_ys = _encode_one_hot_label(test_ys, num_classes=2)
        test_acc = _test_model(sess, X_placeholder, keep_prob_placeholder, preds, test_xs, test_ys)
        return {"variables": tf.trainable_variables(), "test_accuracy": test_acc}
    else:
        return {"variables": tf.trainable_variables()}


if __name__ == '__main__':
    tf.reset_default_graph()  # Reset graph in the runtime.
    run()
