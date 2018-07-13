import os

import numpy as np
import tensorflow as tf

from helper_rnn import rnn, linear, Batch, one_layer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data_dir = './'


class Model:
    def __init__(self):
        pass

    def network(self, steps, nfeatures, rnn_cell_units, out_units, units_att=32, learning_rate=0.01):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, steps, nfeatures])
        self.y = tf.placeholder(tf.float32, [None, 1])
        rnn_out, rnn_states = rnn(self.x, rnn_cell=tf.nn.rnn_cell.LSTMCell, cell_units=rnn_cell_units)
        ### attention part
        attentions = []
        self.history_impacts = tf.get_variable('impacts', [steps], initializer=tf.constant_initializer(1.0))
        for astep in range(steps):
            one_attention = one_layer(rnn_out[:, astep, :], 'attention_layer_%d' % astep, units_att)
            attentions.append(one_attention * self.history_impacts[astep])
        flatten_attention = tf.concat(attentions, axis=1)
        outlayer_in = flatten_attention
        ## output part
        k = 0
        for aoutunit in out_units:
            outlayer_o = one_layer(outlayer_in, 'out_layer_%d' % k, aoutunit)
            outlayer_in = outlayer_o
            k += 1
        self.output = linear(outlayer_in, 1, 'out_linear_layer')
        self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.output - self.y)))
        l2impact = tf.reduce_mean(tf.square(self.history_impacts))
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss + l2impact)

    def train(self, features, targets, configs):
        batch_size = configs['batch_size']
        rnn_cell_units = configs['rnn_cell_units']
        out_units = configs['out_units']
        learning_rate = configs['learning_rate']
        epochs = configs['epochs']
        attention_units = configs['attention_units']
        steps = features.shape[1]
        n_features = features.shape[2]
        batch = Batch(features, targets)
        steps_perbatch = int(batch.len / batch_size)
        self.network(steps, n_features, rnn_cell_units, out_units, units_att=attention_units,
                     learning_rate=learning_rate)
        loss_list = []
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for ep in range(epochs):
                for astep in range(steps_perbatch):
                    af, at = batch.next(batch_size)
                    _, aloss, impacts = sess.run([self.opt, self.loss, self.history_impacts],
                                                 feed_dict={self.x: af, self.y: at.reshape([-1, 1])})
                    loss_list.append(aloss)
            saver.save(sess, data_dir + "fellow_session/model.ckpt")
        return loss_list, impacts

    def test(self, features, targets, configs):
        rnn_cell_units = configs['rnn_cell_units']
        attention_units = configs['attention_units']
        out_units = configs['out_units']
        steps = features.shape[1]
        n_features = features.shape[2]
        self.network(steps, n_features, rnn_cell_units, out_units, units_att=attention_units)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, data_dir + "fellow_session/model.ckpt")
            out, tloss = sess.run([self.output, self.loss],
                                  feed_dict={self.x: features, self.y: targets.reshape([-1, 1])})
        return tloss, out


if __name__ == "__main__":
    features = np.load("features.npy")
    targets = np.load("targets.npy")
    trainX = features[:-300]
    trainT = targets[:-300]
    testX = features[-300:]
    testT = targets[-300:]

    configs = {'rnn_cell_units': [40, 40, 40, 40], 'learning_rate': 5e-5, 'batch_size': 64, 'epochs': 200,
                'attention_units': 32, 'out_units': []}
    m = Model()
    allloss, impacts = m.train(trainX, trainT, configs)
    np.save('impacts.npy', impacts)
    er, py = m.test(testX, testT, configs)
    print("error in average", er)
    for predicting, real in zip(py[:10], testT[:10]):
        print("%.2f %d" % (predicting, real))
    import matplotlib.pyplot as plt

    plt.plot(allloss)
    plt.show()