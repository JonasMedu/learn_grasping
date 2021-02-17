"""
This document saves the different runs to learn various different reward flavours.
The learned policies shall be analyzed and visualized.
"""

import os
import time

from hand_env.allegro_env_unsupervised import AllegroCollisionFromTactileHand
from hand_env.noisy_hand import AllegronNoiseFromUpper
from setting_utils.param_handler import TB_DIR, OriginalParams

import baselines.common.tf_util as U
from ausy_base.learn_policy import learn_on_env as learn_ppo
import numpy as np

from setting_utils.param_handler import Parmap
import tensorflow as tf
from itertools import product
from multiprocessing import Pool

from setting_utils.rewards import close_to_init_no_debug, or_dist, \
    close_to_init, weighted_or_dist

os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
os.environ['OPENAI_LOGDIR'] = TB_DIR


def multi_wrapper(args):
    pars = OriginalParams(Parmap(time_step=1e-2, nb_iter=500, discount=.99))
    rwd_function = args[1]
    pars.rwd_func = rwd_function

    t = time.asctime().replace(" ", "_").replace(":", "_")
    run_name_suffix = rwd_function.__name__ + str(t)
    pars.run_name = run_name_suffix + pars.run_name

    env = AllegroCollisionFromTactileHand(pars, gui=False, show_fingers=False)
    tf.reset_default_graph()
    sess = tf.Session()
    learn_ppo(
        session=sess,
        env=env,
        parameters=pars,
        dir_out="True")
    print("reset")
    time.sleep(0.1)
    U.get_session().close()
    env.close()
    time.sleep(0.1)
    return args[0]


def multi_wrapper_noisy(args):
    """learns a noisy environment"""
    pars = OriginalParams(Parmap(time_step=1e-2, nb_iter=500, discount=.99, movement_noise=args))
    pars.rwd_func = weighted_or_dist

    t = time.asctime().replace(" ", "_").replace(":", "_")
    run_name_suffix = "NoisyEnv_{}".format(t)
    pars.run_name = run_name_suffix + pars.run_name

    env = AllegronNoiseFromUpper(pars, gui=False, show_fingers=False)
    tf.reset_default_graph()
    sess = tf.Session()
    learn_ppo(
        session=sess,
        env=env,
        parameters=pars,
        dir_out="True")
    print("reset")
    time.sleep(0.1)
    U.get_session().close()
    env.close()
    time.sleep(0.1)
    return args


def learn_close_to_init_no_debug():
    """ Shows the effect the object_fell definition without "debug" information """
    processors = 5
    N = 5
    iter = np.linspace(1, N, N)
    rwd_function = close_to_init_no_debug
    p = Pool(processors)
    print(p.map(multi_wrapper, product(iter, [rwd_function])))
    p.terminate()
    exit(0)


def learn_close_to_init():
    """ Shows the effect the object_fell definition with "debug" information, but unweighted finger signals """
    processors = 5
    N = 5
    iter = np.linspace(1, N, N)
    rwd_function = close_to_init
    p = Pool(processors)
    print(p.map(multi_wrapper, product(iter, [rwd_function])))
    p.terminate()
    exit(0)


def learn_or_dist_no_other_debug():
    """ Shows the effect if all fingers """
    processors = 5
    N = 5
    iter = np.linspace(1, N, N)
    rwd_function = or_dist
    p = Pool(processors)
    print(p.map(multi_wrapper, product(iter, [rwd_function])))
    p.terminate()
    exit(0)


def learn_break_to_noise():
    """ Shows the effect of upper policy noise """
    processors = 5
    N = 20
    iter = np.geomspace(.01, 20, N)
    p = Pool(processors)
    print(p.map(multi_wrapper_noisy, iter))
    p.terminate()
    exit(0)


if __name__ == '__main__':
    learn_break_to_noise()