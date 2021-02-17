import os
import time

from hand_env.noisy_hand import AllegronNoiseFromUpper
from setting_utils.param_handler import TB_DIR, OriginalParams

import baselines.common.tf_util as U
from ausy_base.learn_policy import learn_on_env as learn_ppo
import numpy as np

from setting_utils.param_handler import Parmap
import tensorflow as tf

from setting_utils.rewards import weighted_or_dist

os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
os.environ['OPENAI_LOGDIR'] = TB_DIR


def multi_wrapper(args):
    pars = OriginalParams(Parmap(time_step=1e-2, nb_iter=600, movement_noise=args))
    pars.rwd_func = weighted_or_dist

    t = time.asctime().replace(" ", "_").replace(":", "_")
    run_name_suffix = "SimulatedNoise_{}_".format(str(t))
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


if __name__ == '__main__':
    from multiprocessing import Pool
    processors = 6

    from itertools import product

    N = 12
    iter = np.geomspace(0.01, 1, N)
    p = Pool(processors)
    print(p.map(multi_wrapper, iter))
    p.terminate()
    exit(0)