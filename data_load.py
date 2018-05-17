from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex


def load_vocab(corpus):
    vocab = [line.split()[0] for line in codecs.open('preprocessed/{0}.vocab.txt'.format(corpus), 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(src_sents, trg_sents):
    src2idx, idx2src = load_vocab('src')
    trg2idx, idx2trg = load_vocab('trg')

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for src_sent, trg_sent in zip(src_sents, trg_sents):
        # Convert word to index, replace OOV with <UNK>
        src_idx = [src2idx.get(word, 1) for word in (src_sent + " </S>").split()]
        trg_idx = [trg2idx.get(word, 1) for word in (trg_sent + " </S>").split()]
        if max(len(src_idx), len(trg_idx)) <= hp.maxlen:
            x_list.append(np.array(src_idx))
            y_list.append(np.array(trg_idx))
            Sources.append(src_sent)
            Targets.append(trg_sent)

    # Padding
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    #X = []
    #Y = []
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
        #X.append(np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0)))
        #Y.append(np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0)))

    return X, Y, Sources, Targets
    
def load_train_data():
    src_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    trg_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]

    X, Y, Sources, Targets = create_data(src_sents, trg_sents)
    return X, Y

def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    src_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    trg_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
        
    X, Y, Sources, Targets = create_data(src_sents, trg_sents)
    return X, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, Y = load_train_data()

    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # Queues
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
        num_threads=8,
        batch_size=hp.batch_size, 
        capacity=hp.batch_size*64,   
        min_after_dequeue=hp.batch_size*32, 
        allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()