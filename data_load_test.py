
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf
import numpy as np

from data_load import load_vocab, create_data


class VocabUtilsTest(tf.test.TestCase):
    def test_load_vocab(self):
        test_vocab = ['<PAD>', '<UNK>', '<S>','</S>', 'the', 'to', 'of', 'and', 'a']

        test2idx = {word: idx for idx, word in enumerate(test_vocab)}
        idx2test = {idx: word for idx, word in enumerate(test_vocab)}

        word2idx, idx2word = load_vocab('src')

        for key, value in test2idx.items():
            self.assertEqual(value, word2idx[key])

        for key, value in idx2test.items():
            self.assertEqual(value, idx2test[key])

    def test_create_data(self):
        test_src_sents = ["Thank you so much , Chris .", 
                        "I flew on Air Force Two for eight years ."]
        test_trg_sents = ["Terima kasih banyak , Chris .",
                        "Saya terbang menggunakan ' Air Force Two ' selama delapan tahun ."]
        expected = [np.array([ 180,   13,   40,  109,    1, 1111,    1,    3,    0,    0]),
        np.array([ 170,  128,   36,    1, 1096,    1,    3,    0,    0,    0]),
        'Thank you so much , Chris .',
        'Terima kasih banyak , Chris .']

        X, Y, Source, Target = create_data(test_src_sents, test_trg_sents)
        self.assertAllEqual(expected[0], X[0])
        self.assertAllEqual(expected[1], Y[0])
        self.assertEqual(expected[2], Source[0])
        self.assertEqual(expected[3], Target[0])

if __name__ == "__main__":
  tf.test.main()