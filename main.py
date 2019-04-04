# -*- coding:utf-8 -*-
"""
Created on 2019/4/4 10:12 AM.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""

import tensorflow as tf
from prepro import create_vocabulary, create_yelp_ids
from solver import Solver
from model import BiLSTM

flags = tf.app.flags
flags.DEFINE_string('mode', 'test', "train or test")
flags.DEFINE_integer('vocab_size', 5000, 'size of vocab list')
flags.DEFINE_integer('batch_size', 128, 'size of batch')
flags.DEFINE_integer('num_embedding_units', 100, 'size of embeddings layer')
flags.DEFINE_integer('num_hidden_units', 128, 'size of hidden layer')
flags.DEFINE_integer('maxlen', 18, 'max length')

flags.DEFINE_integer('train_step', 2000, 'step of pretraining')
flags.DEFINE_string('model_save_dir', './save/', 'path of saving model')
flags.DEFINE_string('log_dir', './logs/', 'path of logs')

FLAGS = flags.FLAGS

if __name__ == '__main__':

    word2idx, idx2word, vocab_path = create_vocabulary(FLAGS.vocab_size)
    create_yelp_ids(word2idx)

    if not tf.gfile.Exists(FLAGS.model_save_dir):
        tf.gfile.MakeDirs(FLAGS.model_save_dir)

    model = BiLSTM(vocab_size=FLAGS.vocab_size,
                   batch_size=FLAGS.batch_size,
                   embedding_size=FLAGS.num_embedding_units,
                   num_hidden_size=FLAGS.num_hidden_units,
                   maxlen=FLAGS.maxlen)
    solver = Solver(model=model,
                    training_iter=FLAGS.train_step,
                    word2idx=word2idx,
                    idx2word=idx2word,
                    log_dir=FLAGS.log_dir,
                    model_save_dir=FLAGS.model_save_dir)

    if FLAGS.mode == 'train':
        solver.train()
    elif FLAGS.mode == 'test':
        solver.test()