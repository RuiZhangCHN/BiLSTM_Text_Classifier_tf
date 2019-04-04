# -*- coding:utf-8 -*-
"""
Created on 2019/4/1 8:54 PM.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import tensorflow as tf
import os
import json

def create_vocabulary(vocab_size, ids_dir='./ids/'):

    if not tf.gfile.Exists(ids_dir):
        tf.gfile.MakeDirs(ids_dir)

    vocab_ids_path = os.path.join(ids_dir, ('vocabs%d.ids' % vocab_size))

    if not tf.gfile.Exists(vocab_ids_path):
        print('Vocab file %s not found. Creating new vocabs.ids file' % vocab_ids_path)

        word2freq = {}

        with open('./data/yelp/sentiment.train.0') as fin:
            for line in fin.readlines():
                for word in line.strip().split():
                    word2freq[word] = word2freq.get(word, 0) + 1

        with open('./data/yelp/sentiment.train.1') as fin:
            for line in fin.readlines():
                for word in line.strip().split():
                    word2freq[word] = word2freq.get(word, 0) + 1

        sorted_dict = sorted(word2freq.items(), key=lambda item: item[1], reverse=True)
        sorted_dict = sorted_dict[:vocab_size - 4]

        word2idx = {'_PAD_': 0, '_UNK_': 1, '_BOS_': 2, '_EOS_': 3,}
        idx2word = ['_PAD_', '_UNK_', '_BOS_', '_EOS_']
        for w, _ in sorted_dict:
            if w not in ['_PAD_', '_UNK_', '_BOS_', '_EOS_']:
                word2idx[w] = len(word2idx)
                idx2word.append(w)

        print('Save vocabularies into file %s ...' % vocab_ids_path)
        json.dump({'word2idx': word2idx, 'idx2word': idx2word}, open(vocab_ids_path, 'w'), ensure_ascii=False)

        return word2idx, idx2word, vocab_ids_path

    else:
        print('Loading Vocabularies from %s ...' % vocab_ids_path)
        d_vocab = json.load(open(vocab_ids_path))
        word2idx = d_vocab['word2idx']
        idx2word = d_vocab['idx2word']

        return word2idx, idx2word, vocab_ids_path


def create_yelp_ids(word2idx):

    if not tf.gfile.Exists('./ids/yelp/'):
        tf.gfile.MakeDirs('./ids/yelp/')

    if not tf.gfile.Exists('./ids/yelp/sentiment.train.0.ids'):
        print('Ids file for yelp/train.0 not found. Creating ...')

        data = []
        with open('./data/yelp/sentiment.train.0') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/yelp/sentiment.train.0.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/yelp/sentiment.train.1.ids'):
        print('Ids file for yelp/train.1 not found. Creating ...')

        data = []
        with open('./data/yelp/sentiment.train.1') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/yelp/sentiment.train.1.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/yelp/sentiment.dev.0.ids'):
        print('Ids file for yelp/dev.0 not found. Creating ...')

        data = []
        with open('./data/yelp/sentiment.dev.0') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/yelp/sentiment.dev.0.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/yelp/sentiment.dev.1.ids'):
        print('Ids file for yelp/dev.1 not found. Creating ...')

        data = []
        with open('./data/yelp/sentiment.dev.1') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/yelp/sentiment.dev.1.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/yelp/sentiment.test.0.ids'):
        print('Ids file for yelp/test.0 not found. Creating ...')

        data = []
        with open('./data/yelp/sentiment.test.0') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/yelp/sentiment.test.0.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')

    if not tf.gfile.Exists('./ids/yelp/sentiment.test.1.ids'):
        print('Ids file for yelp/test.1 not found. Creating ...')

        data = []
        with open('./data/yelp/sentiment.test.1') as fin:
            for line in fin.readlines():
                words = line.strip().split()
                if len(words) < 15 and len(words) > 0:
                    text_ids = []
                    for w in words:
                        text_ids.append(word2idx.get(w, 1))
                    text_ids = [word2idx['_BOS_']] + text_ids + [word2idx['_EOS_']]
                    data.append(text_ids)

        with open('./ids/yelp/sentiment.test.1.ids', 'w') as fout:
            for d in data:
                fout.write(' '.join([str(id) for id in d]))
                fout.write('\n')