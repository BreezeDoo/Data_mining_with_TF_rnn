#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def linear(inputs, n_cells=1, scope=None):
    with tf.variable_scope(scope or "linear_layer"):
        width = inputs.shape[1].value
        W = tf.get_variable("W_linear", [width, n_cells], initializer=tf.constant_initializer(value = 1, dtype=tf.float32))
        b = tf.get_variable("b_linear", [n_cells], initializer=tf.constant_initializer(value = 0, dtype=tf.float32))
        return tf.matmul(inputs, W) + b


def birnn(inputs, cell_units, rnn_cell=tf.nn.rnn_cell.GRUCell, scope=None):
    """
    Bidirectional RNN
    :param inputs: [batch_size, max_time, ...]
    :param rnn_cell:tf.nn.rnn_cell.LSTMCell or GRUCell
    :param cell_units: number of neurals
    :param scope:
    :return:
    """
    with tf.variable_scope(scope or "encoder"):  # 共享变量
        fw_cell = rnn_cell(cell_units)
        bw_cell = rnn_cell(cell_units)

        ((fw_outputs, bw_outputs), (fw_final_state, bw_final_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs, dtype=tf.float32))

        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        final_state = tf.concat((fw_final_state, bw_final_state), 1)

        return outputs, final_state


def rnn(inputs, rnn_cell=tf.nn.rnn_cell.GRUCell, cell_units=[10, 10], scope=None):
    """
    RNN model. with GRUCells
    """
    with tf.variable_scope(scope or "encoder"):
        # cell = rnn_cell(cell_units)
        rnn_layers = [rnn_cell(size) for size in cell_units]
        multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
        outputs, final_state = tf.nn.dynamic_rnn(multi_rnn_cell, inputs, dtype=tf.float32)
        return outputs, final_state


# 批量取数据
class Batch:
    def __init__(self, data, target):
        self.data = np.array(data)
        self.target = np.array(target)
        self.len = len(self.data)
        self.cursor = 0
        self.idx = list(range(self.len))

    def _reinit(self):
        from random import shuffle
        shuffle(self.idx)
        self.cursor = 0

    def next(self, batch_size):
        if self.cursor + batch_size > self.len:
            self._reinit()
        curidx = self.idx[self.cursor: self.cursor + batch_size]
        curdata = self.data[curidx]
        curtarget = self.target[curidx]
        self.cursor += batch_size
        return curdata, curtarget


class Model:
    def __init__(self):
        pass

    def network(self, steps, nfeatures, rnn_cell_units, out_unit, learning_rate=0.01):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, steps, nfeatures])
        self.y = tf.placeholder(tf.float32, [None, 1])
        rnn_out, rnn_states = rnn(self.x, cell_units=rnn_cell_units)
        self.output = linear(rnn_out[:, -1, :], out_unit)
        self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.output - self.y)))
        self.rounded_output = tf.placeholder(tf.float32, [None, 1])
        self.rounded_loss = tf.reduce_mean(tf.sqrt(tf.square(self.rounded_output - self.y)))
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def train(self, features, targets, configs):
        batch_size = configs['batch_size']
        rnn_cell_units = configs['rnn_cell_units']
        out_units = 1
        learning_rate = configs['learning_rate']
        epochs = configs['epochs']
        steps = features.shape[1]
        n_features = features.shape[2]
        batch = Batch(features, targets)
        steps_perbatch = int(batch.len / batch_size)
        self.network(steps, n_features, rnn_cell_units, out_units, learning_rate)
        loss_list = []
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for ep in range(epochs):
                for astep in range(steps_perbatch):
                    af, at = batch.next(batch_size)
                    _, aloss = sess.run([self.opt, self.loss], feed_dict={self.x: af, self.y: at.reshape([-1, 1])})
                    loss_list.append(aloss)
            saver.save(sess, "./fellow_session/model.ckpt")
        return loss_list

    def test(self, features, targets, configs):
        rnn_cell_units = configs['rnn_cell_units']
        out_units = 1
        steps = features.shape[1]
        n_features = features.shape[2]
        self.network(steps, n_features, rnn_cell_units, out_units)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "fellow_session/model.ckpt")
            out, tloss = sess.run([self.output, self.loss],
                                  feed_dict={self.x: features, self.y: targets.reshape([-1, 1])})
            rounded_output = []
            for itera in out.reshape(out.shape[0]):
                rounded_output.append(round(itera))
            rounded_output = np.array(rounded_output)
            rnd_tloss = sess.run(self.rounded_loss, feed_dict = {self.rounded_output: rounded_output.reshape([-1, 1]), self.y: targets.reshape([-1, 1])})
            # rnd_tloss = sess.run(tf.reduce_mean(tf.sqrt(tf.square(tf.convert_to_tensor(rounded_output - targets.reshape([-1, 1]))))))
        return rnd_tloss, tloss, out


if __name__ == "__main__":
    error_list = []
    rounded_error_list = []
    test_index_list = [1608, 1739, 1890, 2061, 2244, 2448, 2666, 2860, 3039, 3251]
    # test_index_list = [1608, 1739, 1890, 2061, 2244]
    for year in range(2015, 2005, -1):
        test_index = test_index_list[year - 2006]
        features = np.load("features_%d.npy" % year)
        targets = np.load("targets_%d.npy" % year)
        X = features.reshape(features.shape[0], -1)
        trainX = X[:test_index]
        trainT = targets[:test_index]
        testX = X[test_index:]
        testT = targets[test_index:]

        from sklearn.preprocessing import StandardScaler

        trainX = np.array(trainX)
        testX = np.array(testX)
        trainT = np.array(trainT)
        testT = np.array(testT)
        sc = StandardScaler()
        trainX = sc.fit_transform(trainX)
        testX = sc.transform(testX)
        trainX = trainX.reshape((test_index, features.shape[1], features.shape[2]))
        testX = testX.reshape((features.shape[0] - test_index, features.shape[1], features.shape[2]))

        if year >= 2011:
            configs = {'rnn_cell_units': [64, 32], 'learning_rate': 0.01, 'batch_size': 50, 'epochs': 50}
        else :
            configs = {'rnn_cell_units': [64, 32], 'learning_rate': 0.005, 'batch_size': 50, 'epochs': 50}   
        m = Model()
        allloss = m.train(trainX, trainT, configs)

        rnd_tloss, er, py = m.test(testX, testT, configs)
        error_list.append(er)
        rounded_error_list.append(rnd_tloss)
        print("This is year of ", year)
        print("error in average", er)
        print("error in average after rounded", rnd_tloss)

    print(error_list)
    print(rounded_error_list)
    # print("error in average", er)
    # print("error in average after rounded", rnd_tloss)
    # for predicting, real in zip(py[:10], testT[:10]):
    #     print("%.2f %d" % (predicting, real))

    # import matplotlib.pyplot as plt

    # plt.plot(allloss)
    # plt.show()
