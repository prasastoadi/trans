# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'corpora/train.tags.en-id.en'
    target_train = 'corpora/train.tags.en-id.id'
    source_test = 'corpora/IWSLT17.TED.tst2017plus.en-id.en.xml'
    target_test = 'corpora/IWSLT17.TED.tst2017plus.en-id.id.xml'
    
    source_train_small = 'corpora/train.tags.en-id.small.en'
    target_train_small = 'corpora/train.tags.en-id.small.id'

    source_train_medium = 'corpora/train.tags.en-id.medium.en'
    target_train_medium = 'corpora/train.tags.en-id.medium.id'

    source_train = source_train
    target_train = target_train
    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 50 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    
    
    
