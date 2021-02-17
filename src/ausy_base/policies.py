import tensorflow as tf
import numpy as np


class MLP:
    def __init__(self, sizes, activations=None):
        if activations is None:
            activations = [tf.nn.relu] * (len(sizes) - 2) + [tf.identity]

        self.x = last_out = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, sizes[0]], name="x")
        for l, size in enumerate(sizes[1:]):
            last_out = tf.layers.dense(last_out, size, activation=activations[l], kernel_initializer=tf.glorot_normal_initializer(),
                                       name='layer' + str(l) if l < len(sizes[1:-1]) else "y")
        self.out = last_out


class ConvMLP:
    def __init__(self, in_size, out_size):
        conv_size = 5
        self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, in_size],  name="x")

        # Convolutional Layer 1

        outers = tf.einsum('ai,aj->aij', self.x, self.x)

        input_layer = tf.reshape(outers, [-1, in_size, in_size])
        conv = tf.layers.conv1d(
            inputs=input_layer,
            filters=in_size,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        conv_a = tf.layers.max_pooling1d(conv, 5, strides=3, padding="same", name="layer_conv1")

        conv2 = tf.layers.conv1d(
            inputs=conv_a,
            filters=13,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        conv_n = tf.layers.max_pooling1d(conv2, 3, strides=3, padding="same", name="layer_conv2")
        to_dense = tf.layers.flatten(conv_n)
        layer2 = tf.layers.dense(to_dense, 32, activation=tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer(), name="layer_2")
        layer3 = tf.layers.dense(layer2, 16, activation=tf.nn.relu6, kernel_initializer=tf.glorot_normal_initializer(), name="layer_3")

        self.out = tf.layers.dense(layer3, out_size, activation=tf.nn.tanh, kernel_initializer=tf.glorot_normal_initializer(), name="y")


def getMLPdef(a_dim, s_dim):
    sizes = [s_dim] + [64] * 2 + [a_dim]
    activations = [tf.nn.relu6] * (len(sizes) - 3) + [tf.nn.tanh] + [tf.identity]
    return MLP(sizes, activations)


class MLPGaussianPolicy:
    def __init__(self, session, act_dim, obs_dim, init_sigma=1., min_sigma=1e-1, mean_mult=1.):
        self.sess = session

        with tf.compat.v1.variable_scope("pi"):
            # policy specific..
            self.mlp = getMLPdef(act_dim, obs_dim)

            # action tensor (diagonal Gaussian)
            self.logsigs = tf.Variable(np.log(init_sigma) * tf.ones([1, act_dim]), trainable=True, name='logstd')
            self.min_sigma = tf.Variable(min_sigma, trainable=False, name='min_sig')
            self.sel_sigs = tf.maximum(tf.exp(self.logsigs), self.min_sigma)
            self.gauss_pol = tf.distributions.Normal(mean_mult * self.mlp.out, self.sel_sigs)
            self.act_tensor = tf.identity(self.gauss_pol.sample(), name="policy_output")

            # action proba (diagonal Gaussian)
            self.test_action = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, act_dim], name="test_action")
            self.log_prob = tf.reduce_sum(self.gauss_pol.log_prob(self.test_action), axis=1, keepdims=True)

            # pol entropy (for logging only)
            self.entropy = tf.reduce_sum(tf.math.log(self.sel_sigs) + np.log(2 * np.pi * np.e) / 2)

    def get_action(self, obs):
        if obs.ndim == 1:
            return np.squeeze(self.sess.run(self.act_tensor, {self.mlp.x: np.asmatrix(obs)}), axis=0)
        else:
            return self.sess.run(self.act_tensor, {self.mlp.x: obs})

    def get_log_proba(self, obs, act):
        return self.sess.run(self.log_prob, self.get_log_p_feed_d(obs, act))

    def get_log_p_feed_d(self, obs, act):
        return {self.mlp.x: obs, self.test_action: act}

    def get_entropy(self, obs, act):
        return self.sess.run(self.entropy)
