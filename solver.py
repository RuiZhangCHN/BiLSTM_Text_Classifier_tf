# -*- coding:utf-8 -*-
"""
Created on 2019/4/4 10:18 AM.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import tensorflow as tf
import numpy as np
import os

class Solver(object):

    def __init__(self, model, training_iter, word2idx, idx2word, log_dir, model_save_dir):

        self.model = model
        self.training_iter = training_iter

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.log_dir = log_dir
        self.model_save_dir = model_save_dir

        self.word2idx = word2idx
        self.idx2word = idx2word

    def load_data(self, split='train'):
        texts_data, labels_data = [], []

        if split == 'train':
            with open('./ids/yelp/sentiment.train.1.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(1)
            with open('./ids/yelp/sentiment.train.0.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(0)
        elif split == 'dev':
            with open('./ids/yelp/sentiment.dev.1.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(1)
            with open('./ids/yelp/sentiment.dev.0.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(0)
        elif split == 'test':
            with open('./ids/yelp/sentiment.test.1.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(1)
            with open('./ids/yelp/sentiment.test.0.ids') as fin:
                for line in fin.readlines():
                    text = [int(v) for v in line.strip().split()]
                    texts_data.append(text)
                    labels_data.append(0)

        texts_data = np.array(texts_data)
        labels_data = np.array(labels_data)

        shuffle_idx = np.random.permutation(range(len(texts_data)))
        texts_data = texts_data[shuffle_idx]
        labels_data = labels_data[shuffle_idx]

        return texts_data, labels_data

    def prepare_text_batch(self, batch, pad_to_max=True):
        maxlen = self.model.maxlen if pad_to_max else max([len(b) for b in batch])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences=batch,
                                                               maxlen=maxlen,
                                                               padding='post',
                                                               value=0)
        return padded

    def train(self):

        # load data
        train_texts, train_labels = self.load_data('train')
        dev_texts, dev_labels = self.load_data('dev')

        model = self.model

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            train_loss = 0.
            train_acc = 0.

            best_dev_acc = - np.inf

            for step in range(self.training_iter+1):
                i = step % int(len(train_texts) // model.batch_size)
                batch_texts = train_texts[i*model.batch_size:(i+1)*model.batch_size]
                batch_labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]

                batch_inp_text = self.prepare_text_batch(batch_texts)

                feed_dict = {model.texts: batch_inp_text,
                             model.labels: batch_labels,
                             model.keep_prob: 0.75}

                _loss, _acc, _ = sess.run([model.loss, model.accuracy, model.train_op], feed_dict)

                train_loss += _loss
                train_acc += _acc

                if (step+1) % 10 == 0:
                    train_loss /= 10.0
                    train_acc /= 10.0

                    print('Training[%d/%d]:  train_loss:[%.4f] train_acc:[%.4f]'
                          % (step+1, self.training_iter, train_loss, train_acc))

                    train_loss = 0.
                    train_acc = 0.

                if (step+1) % 400 == 0:
                    dev_loss = 0.
                    dev_acc = 0.

                    num_of_batch = len(dev_texts) // model.batch_size

                    for j in range(num_of_batch):
                        dev_batch_texts = dev_texts[j*model.batch_size:(j+1)*model.batch_size]
                        dev_batch_labels = dev_labels[j*model.batch_size:(j+1)*model.batch_size]
                        dev_batch_inp = self.prepare_text_batch(dev_batch_texts)

                        _loss_dev, _acc_dev = sess.run([model.loss, model.accuracy],
                                                 feed_dict={model.texts: dev_batch_inp,
                                                            model.labels: dev_batch_labels,
                                                            model.keep_prob: 1.0})
                        dev_loss += _loss_dev
                        dev_acc += _acc_dev

                    dev_loss /= num_of_batch
                    dev_acc /= num_of_batch

                    print('Developing[%d/%d]: dev_loss:[%.4f] dev_acc:[%.4f]'
                          % (step+1, self.training_iter, dev_loss, dev_acc))

                    if dev_acc > best_dev_acc:
                        print('Saving model ...')
                        saver.save(sess, os.path.join(self.model_save_dir, 'best-model'))
                        best_dev_acc = dev_acc

    def test(self):

        # load test dataset
        texts, labels = self.load_data('test')

        model = self.model

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()

            restorer = tf.train.Saver()
            restorer.restore(sess, os.path.join(self.model_save_dir, 'best-model'))

            test_acc = 0.
            num_of_batch = len(texts) // model.batch_size

            for j in range(num_of_batch):
                text_batch = texts[j*model.batch_size:(j+1)*model.batch_size]
                labels_batch = labels[j*model.batch_size:(j+1)*model.batch_size]
                inp_batch = self.prepare_text_batch(text_batch)

                _acc = sess.run(model.accuracy,
                                feed_dict={model.texts: inp_batch,
                                           model.labels: labels_batch,
                                           model.keep_prob: 1.0})

                test_acc += _acc

            test_acc /= num_of_batch

            print('Test Accuracy:', test_acc)

    def run(self, inputs):
        raise NotImplementedError('NOT implemented yet.')