import tensorflow as tf
import numpy as np
from ausy_base import data_handling as dat
from ausy_base.logs import MyLogger

logger = MyLogger().logger

class PolicyUpdateObjective:
    def __init__(self, session, policy, e_clip=.2, a_lrate=3e-3):
        self.sess, self.pol = session, policy

        # ppo objective + expected div cost penalty
        self.advantage = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1], name='advantage')
        tf.compat.v1.summary.scalar("advantage", self.advantage, family="ppo_objective" )

        self.old_log_probas = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])
        tf.compat.v1.summary.tensor_summary("old_log_probas", self.old_log_probas, family="ppo_objective")
        proba_ratio = tf.exp(self.pol.log_prob - self.old_log_probas)
        self.clip_pr = tf.clip_by_value(proba_ratio, 1 - e_clip, 1 + e_clip)

        self.neg_objective_act = -tf.reduce_mean(tf.minimum(tf.multiply(proba_ratio, self.advantage), tf.multiply(self.clip_pr, self.advantage))) \
                                 + tf.maximum(tf.reduce_mean(tf.abs(proba_ratio - 1)), e_clip)
        self.optimizer_act = tf.compat.v1.train.AdamOptimizer(a_lrate).minimize(self.neg_objective_act)

    def train_pol(self, feed_d):
        return self.sess.run(self.optimizer_act, feed_d)

    def evaluate_pol(self, feed_d):
        return -self.sess.run(self.neg_objective_act, feed_d)

    def get_feed_d(self, obs, act, old_log_p, adv, indices):  # feed dictionary for training/evaluating policy
        feed_d = {self.advantage: adv[indices], self.old_log_probas: old_log_p[indices]}
        feed_d.update(self.pol.get_log_p_feed_d(obs[indices, :], act[indices, :]))
        return feed_d


class P3O:
    def __init__(self, session, policy, p_eval, p_update):
        self.sess, self.pol, self.p_ev, self.p_up = session, policy, p_eval, p_update

    def iteration(self, v_paths, p_paths, discount, lam_trace, epochs, batch_size, exploration_reducer=None, verbose=False):
        # policy eval
        self.p_ev.update_v(v_paths["obs"], v_paths["act"], v_paths["rwd"], v_paths["done"], v_paths["logprob"],
                           discount, lam_trace, epochs, batch_size, verbose=verbose)
        # policy update
        # # get generalized advantages
        gen_adv = self.p_ev.get_adv(p_paths["obs"], p_paths["act"], p_paths["rwd"], p_paths["done"], discount, lam_trace)
        if verbose:
            logger.info('advantages: std {0:.3f} mean {1:.3f} min {2:.3f} max {3:.3f}'.format(np.std(gen_adv), np.mean(gen_adv), np.min(gen_adv), np.max(gen_adv)))
        gen_adv = (gen_adv - np.mean(gen_adv)) / np.std(gen_adv)

        # # log
        if verbose:
            logger.info('entropy before update: {}'.format(self.pol.get_entropy(p_paths["obs"], p_paths["act"])))
            logger.info('policy objective before update: {}'.format(self.p_up.evaluate_pol(self.p_up.get_feed_d(p_paths["obs"], p_paths["act"], p_paths["logprob"], gen_adv, range(len(gen_adv))))))
            logger.info('min sigma: {}'.format(self.sess.run(self.pol.min_sigma)))

        # # reduce lower bound on exploration noise
        if exploration_reducer is not None:
            exploration_reducer()

        # # gradient ascent on policy update objective
        for epoch in range(epochs):
            for batch_idx in dat.next_batch_idx(batch_size, len(gen_adv)):
                self.p_up.train_pol(self.p_up.get_feed_d(p_paths["obs"], p_paths["act"], p_paths["logprob"], gen_adv, batch_idx))

        # # log
        if verbose:
            logger.info('entropy after update: {}'.format(self.pol.get_entropy(p_paths["obs"], p_paths["act"])))
            logger.info('policy objective after update: {}'.format(self.p_up.evaluate_pol(self.p_up.get_feed_d(p_paths["obs"], p_paths["act"], p_paths["logprob"], gen_adv, range(len(gen_adv))))))
            logger.info('min sigma: {}'.format(self.sess.run(self.pol.min_sigma)))
            log_act_probas_new = self.pol.get_log_proba(p_paths["obs"], p_paths["act"])
            abs_diff = np.abs(np.exp(log_act_probas_new - p_paths["logprob"]) - 1)
            logger.info('action ratio: min {0:.3f} mean {1:.3f} max {2:.3f} std {3:.3f}'.format(np.min(abs_diff), np.mean(abs_diff), np.max(abs_diff), np.std(abs_diff)))

        return gen_adv
