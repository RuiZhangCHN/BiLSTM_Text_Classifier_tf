# -*- coding:utf-8 -*-
"""
Created on 2019/4/4 10:18 AM.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import tensorflow as tf

class BiLSTM(object):

    def __init__(self,
                 vocab_size,
                 batch_size,
                 embedding_size,
                 num_hidden_size,
                 maxlen,
                 num_categories=2):

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_hidden_size = num_hidden_size
        self.maxlen = maxlen
        self.num_categories = num_categories

        self._build_model()
        self._build_graph()

    def _build_model(self):

        with tf.device('/cpu:0'):
            with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
                self.embeddings = tf.get_variable('embedding_lookup',
                                                  [self.vocab_size, self.embedding_size],
                                                  dtype=tf.float32)

        self.hidden_proj = tf.layers.Dense(self.num_hidden_size, activation='linear')

        self.fw_encoder_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden_size, name='fw_cell')
        self.bw_encoder_cell = tf.nn.rnn_cell.GRUCell(self.num_hidden_size, name='bw_cell')

        self.discriminator_dense = tf.layers.Dense(self.num_hidden_size, name='discriminator_dense')
        self.discriminator_out = tf.layers.Dense(self.num_categories, name='discriminator_out')

    def _build_graph(self):

        self.texts = tf.placeholder(tf.int32, [None, self.maxlen], name='input_texts')
        self.labels = tf.placeholder(tf.int64, [None], name='input_labels')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.text_lens = tf.cast(tf.reduce_sum(tf.sign(self.texts), 1), tf.int32)

        text_embedding = tf.nn.embedding_lookup(self.embeddings, self.texts)
        proj_emb = self.hidden_proj(text_embedding)

        _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_encoder_cell,
                                                          cell_bw=self.bw_encoder_cell,
                                                          sequence_length=self.text_lens,
                                                          inputs=proj_emb,
                                                          dtype=tf.float32)
        concat_states = tf.concat(states, 1)        # 2*(batch_size x hidden_size) ==> batch_size x (2*hidden_size)
        concat_states = tf.nn.dropout(concat_states, keep_prob=self.keep_prob)

        output = self.discriminator_dense(concat_states)
        output = tf.nn.tanh(output)
        self.output = self.discriminator_out(output)

        self.ypred_for_auc = tf.nn.softmax(self.output)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.output+1e-10,
                labels=self.labels
            )
        )

        ypred = tf.cast(tf.argmax(self.output, 1), tf.int64)
        correct = tf.equal(ypred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        tvars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss, var_list=tvars)