import os

from hand_env.allegro_env_unsupervised import AllegroCollisionFromTactileHand
from hand_env.noisy_hand import AllegronNoiseFromUpper
from setting_utils.param_handler import Parmap, TB_DIR, OriginalParams
import time
import baselines.common.tf_util as U

from setting_utils.rewards import move_connection_to_object
import numpy as np

os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
os.environ['OPENAI_LOGDIR'] = TB_DIR
import tensorflow as tf
from ausy_base.learn_policy import learn_on_env as learn_ppo

if __name__ == "__main__":
    t = time.asctime().replace(" ", "_").replace(":", "_")
    print('Start run at %s'%t)
    pars = OriginalParams(Parmap(nb_iter=300))
    pars.rwd_func = move_connection_to_object

    env = AllegroCollisionFromTactileHand(pars, gui=False, show_fingers=False)
    # logging
    run_name_suffix = "TestingNewSaveDir_{}_".format(str(t))
    pars.run_name = run_name_suffix + pars.run_name
    print(dict(**vars(pars)))
    # init session
    tf.reset_default_graph()
    sess = tf.Session()
    # learn
    learn_ppo(
        session=sess,
        env=env,
        parameters=pars,
        dir_out="True")
    # clean session close
    time.sleep(0.1)
    U.get_session().close()
    time.sleep(0.1)
    exit(0)
