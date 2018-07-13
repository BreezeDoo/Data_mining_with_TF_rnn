import numpy as np
import tensorflow as tf


def birnn(inputs, cell_units, rnn_cell=tf.nn.rnn_cell.GRUCell, scope=None):
    """
    Bidirectional RNN
    :param inputs: [batch_size, max_time, ...]
    :param rnn_cell:tf.nn.rnn_cell.LSTMCell or GRUCell
    :param cell_units: number of neurals
    :param scope:
    :return:
    """
    with tf.variable_scope(scope or "encoder"):
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


def linear(inputs, n_cells, scope):
    with tf.variable_scope(scope):
        width = inputs.shape[1].value
        W = tf.get_variable("W_linear", [width, n_cells], initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=1.0,
            seed=None,
            dtype=tf.float32))
        b = tf.get_variable("b_linear", [n_cells], initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=1.0,
            seed=None,
            dtype=tf.float32))
        return tf.matmul(inputs, W) + b


def one_layer(inputs, scope, n_cells=8, activation=tf.nn.relu):
    with tf.variable_scope(scope):
        width = inputs.shape[1].value
        W = tf.get_variable("W_linear", [width, n_cells], initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=1.0,
            seed=None,
            dtype=tf.float32))
        b = tf.get_variable("b_linear", [n_cells], initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=1.0,
            seed=None,
            dtype=tf.float32))
        return activation(tf.matmul(inputs, W) + b)
