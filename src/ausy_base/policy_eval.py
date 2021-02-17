import tensorflow as tf
import numpy as np
from ausy_base import data_handling as dat
from baselines import logger


class PolicyEvalBase:
    def __init__(self, session, policy, vfunc, v_lrate=5e-3):
        self.sess = session
        self.v = vfunc
        self.pol = policy

        # loss for v function
        self.target_v = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1], name="target_placeholder")
        tf.compat.v1.summary.scalar("target_v", self.target_v, family="value_function")
        self.loss_v = tf.compat.v1.losses.mean_squared_error(self.v.out, self.target_v)
        self.optimizer_v = tf.compat.v1.train.AdamOptimizer(v_lrate).minimize(self.loss_v)
        tf.compat.v1.summary.scalar("loss_v", self.loss_v, family="value_function")

    def train_v(self, obs, target_v):
        return self.sess.run(self.optimizer_v, {self.v.x: obs, self.target_v: target_v})

    def evaluate_v(self, obs, target_v):
        return self.sess.run(self.loss_v, {self.v.x: obs, self.target_v: target_v})

    def get_v(self, obs):
        return self.sess.run(self.v.out, {self.v.x: obs})


class PolicyEvalV(PolicyEvalBase):
    def update_v(self, obs, act, rwd, done, old_logprob, discount, lam_trace,
                 epochs_per_iter=10, batch_size_v=64, verbose=False):
        # update v-function
        act_log_prob = self.pol.get_log_proba(obs, act)
        retrace_proba_ratio = np.minimum(1., np.exp(act_log_prob - old_logprob))
        for epoch in range(epochs_per_iter):
            # # compute the generalized adv and v_targets
            v_targets, _ = self.get_targets(obs, rwd, done, discount, lam_trace, retrace_proba_ratio)

            # # log Bellman error before training
            if verbose and epoch == 0:
                v_loss_before_training =self.evaluate_v(obs, v_targets)
                logger.record_tabular('v-function_before_training', v_loss_before_training)

            # # gradient descent on loss
            for batch_idx in dat.next_batch_idx(batch_size_v, len(v_targets)):
                self.train_v(obs[batch_idx], v_targets[batch_idx])

        if verbose:
            v_loss_after_training =self.evaluate_v(obs, v_targets)
            logger.record_tabular('v-function_after_training', v_loss_after_training)

    def get_targets(self, obs, rwd, done, discount, lam, prob_ratio=None):
        # computes v_update targets
        v_values = self.get_v(obs)
        gen_adv = np.empty_like(v_values)
        if prob_ratio is None:
            prob_ratio = np.ones([len(v_values)])
        for rev_k, v in enumerate(reversed(v_values)):
            k = len(v_values) - rev_k - 1
            if done[k]:  # this is a new path. always true for rev_k == 0
                gen_adv[k] = prob_ratio[k] * (rwd[k] - v_values[k])
            else:
                gen_adv[k] = prob_ratio[k] * (rwd[k] + discount * v_values[k + 1] - v_values[k] + discount * lam * gen_adv[k + 1])
        return gen_adv + v_values, gen_adv

    def get_adv(self, obs, act, rwd, done, discount, lam, prob_ratio=None):
        _, adv = self.get_targets(obs, rwd, done, discount, lam, prob_ratio)
        return adv
