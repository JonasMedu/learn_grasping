import tensorflow as tf
import numpy as np
import pandas as pd
from baselines import logger

class FixedIterSaver:
    def __init__(self, session, file_base_name, verbose=True, max_to_keep=1, iter_to_save=[]):
        self.sess, self.file_base_name, self.verbose = session, file_base_name, verbose
        self.saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="policy"), max_to_keep=max_to_keep)
        self.nb_iter = 0
        self.iter_to_save = np.asarray(iter_to_save)
        self.writer = tf.compat.v1.Summary.FileWriter(file_base_name, session.graph)

    def next_iter(self):
        self.nb_iter += 1
        if np.any((self.iter_to_save - self.nb_iter) == 0) or self.verbose:

            # self.saver.saved_model.simple_save(
            #     self.sess,
            #     self.file_base_name,
            #     inputs={"x": "policy/y", "y": self.sess.graph.asd2},
            #     outputs={"z": self.sess.graph.asd},
            #     legacy_init_op=None
            # )
            self.saver.save(self.sess, self.file_base_name + '/' + self.file_base_name,
                            meta_graph_suffix="meta",
                            #global_step=self.nb_iter,
                            write_meta_graph=True)
            # merge = tf.compat.v1.Summary.merge_all()
        if (self.nb_iter % 100) == 0:
            print('--> saving iteration {}'.format(self.nb_iter))



class PolicyIterationLogger:
    def __init__(self, file_base_name, adv_stats=False):
        self.rwd_fname, self.pup_fname = (file_base_name + ext for ext in ['.rwd', '.pupdate'])
        self.trans_fname = file_base_name + '.trans'
        self.iter = 0
        self.nb_trans = 0
        self.partial_return_prev_iter = 0
        with open(self.rwd_fname, "w") as file:
            print("# .rwd contains number of transitions:cumulative reward", file=file)
        with open(self.pup_fname, "w") as file:
            print('# .pupdate contains iteration:nb_trans:entropy:min:quantile25:median:mean:std:quantile75:max '
                  '(stats are over |pi(a|s) / q(a|s) - 1|)', file=file)
            print('# ratio of probability increase:mean proba diff if proba increased:std:mean proba diff if proba '
                  'decreased:std', file=file)
        if adv_stats:
            self.adv_fname = file_base_name + '.adv'
            with open(self.adv_fname, "w") as file:
                print('# .adv contains iteration:nb_trans:min:quantile25:median:mean:std:quantile75:max (stats are '
                      'over adv function)', file=file)
                print('# ratio pos adv:mean pos adv:std:mean neg adv:std ', file=file)

    def next_iter_path(self, rwd, done, entropy, policy_diff, adv=None):
        for k, r in enumerate(rwd):
            self.partial_return_prev_iter += r
            self.nb_trans += 1
            if done[k]:
                with open(self.rwd_fname, "a") as file:
                    print("{} {}".format(self.nb_trans, self.partial_return_prev_iter), file=file)
                self.partial_return_prev_iter = 0
        self.iter += 1
        self.write_update_info(entropy, policy_diff)
        if adv is not None:
            self.write_adv_info(adv)

    def next_iter_ret(self, rets, lens, entropy, policy_diff, adv=None):
        for k, ret in enumerate(rets):
            self.nb_trans += lens[k]
            with open(self.rwd_fname, "a") as file:
                print("{} {}".format(self.nb_trans, ret), file=file)
        self.iter += 1
        self.write_update_info(entropy, policy_diff)
        if adv is not None:
            self.write_adv_info(adv)

    def write_update_info(self, entropy, policy_diff):
        with open(self.pup_fname, "a") as file:
            abs_diff = np.abs(policy_diff)
            msg = "{} {} {} {} {} {} {} {} {} {}".format(self.iter, self.nb_trans, entropy, np.min(abs_diff), np.percentile(abs_diff, 25), np.median(abs_diff),
                                                         np.mean(abs_diff), np.std(abs_diff), np.percentile(abs_diff, 75), np.max(abs_diff))
            proba_inc = policy_diff[policy_diff >= 0]
            proba_dec = policy_diff[policy_diff < 0]
            msg += " {} {} {} {} {}".format(len(proba_inc) / len(policy_diff), np.mean(proba_inc), np.std(proba_inc), np.mean(proba_dec), np.std(proba_dec))
            print(msg, file=file)

    def write_adv_info(self, adv):
        with open(self.adv_fname, "a") as file:
            msg = "{} {} {} {} {} {} {} {} {}".format(self.iter, self.nb_trans, np.min(adv), np.percentile(adv, 25), np.median(adv),
                                                         np.mean(adv), np.std(adv), np.percentile(adv, 75), np.max(adv))
            adv_pos = adv[adv >= 0]
            adv_neg = adv[adv < 0]
            msg += " {} {} {} {} {}".format(len(adv_pos) / len(adv), np.mean(adv_pos), np.std(adv_pos), np.mean(adv_neg), np.std(adv_neg))
            print(msg, file=file)

    def write_transistion(self, paths, advs):
        try:
            ar = np.concatenate([paths['obs'], paths['rwd'], advs], axis=1)
            df = pd.DataFrame(ar)
            df.to_csv(self.trans_fname, index=False, mode='a+')
            print('saved last iteration for visualization purpose!')
        except OSError:
            print('WARNING: Could not save last iteration for visualization purpose.')
            pass

