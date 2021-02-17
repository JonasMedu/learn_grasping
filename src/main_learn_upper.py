import os
import uuid

from setting_utils.param_handler import UpperParmap, Parmap, TB_DIR
from hand_env.trained_env import TrainedEnv
from scripts.policy_manager import get_result_df, load_trained_model
from setting_utils.rewards import move_object

os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
os.environ['OPENAI_LOGDIR'] = TB_DIR
import tensorflow as tf
from hand_env.allegro_env import AllegroHand
from ausy_base.learn_policy import learn_on_env as learn_ppo
import time
import baselines.common.tf_util as U
import numpy as np


def find_good_pre_policy(lower_pi_suffix = "No_Thumb"):
    df = get_result_df(lower_pi_suffix=lower_pi_suffix)
    assert df.shape[0]>0, "There are no available polcies with the prefix %s" %lower_pi_suffix
    best_choices = df[df["score"] == df["score"].min()]
    return best_choices


def get_trained_env(return_policy, sess=None, lower_run_suffix='intrinsicTerm', **NewParamArgs):
    """
    returns the best performing lower policy and its parameters
    :param return_policy: if true, returns policy, else Environment for learning upper policy
    :return: (policy | or environment) , parameter map which belongs to policy
    """
    best_policy = find_good_pre_policy(lower_pi_suffix=lower_run_suffix)
    lower_log_name = best_policy["log_name"].iloc[0]
    if not sess:
        sess = tf.Session()
    low_pi = load_trained_model(os.path.join(TB_DIR, lower_log_name), lower_log_name, sess)
    parMap = best_policy["ParamObject"].values[0]
    vars(parMap).update(NewParamArgs)
    if return_policy:
        return low_pi, parMap
    else:
        env = TrainedEnv(low_pi, paramObject=parMap)
        return env, env.paramMap


def get_trained_joined_policy(run_suffix_lower='intrinsicTerm', run_suffix_upper='UpperTest'):
    """
    returns the best performing lower policy and its parameters
    :return: (policy | or environment) , parameter map which belongs to policy
    """
    best_policy_lower = find_good_pre_policy(lower_pi_suffix=run_suffix_lower)
    best_policy_upper = find_good_pre_policy(lower_pi_suffix=run_suffix_upper)
    lower_run_name = best_policy_lower["pol_saving_name"].iloc[0]
    upper_run_name = best_policy_upper["pol_saving_name"].iloc[0]
    sess = tf.Session()
    low_pi = load_trained_model(TB_DIR, lower_run_name, sess)
    high_pi = load_trained_model(TB_DIR, upper_run_name, sess)
    parMap = best_policy_upper["ParamObject"].values[0]
    #policy = lambda obs: high_pi(low_pi(obs))
    # since lower is now prior (commit 2bb95eef8978e2122cc0bcef74708cd9b5f7306d)
    # the observation for the upper has to be advanced. (i guess?)
    policy = lambda obs: high_pi(np.concatenate([low_pi(obs), obs], axis=0))
    return policy, parMap

def get_non_trained_env():
    pars = UpperParmap(Parmap())
    env = AllegroHand(pars)
    return env, pars


if __name__ == "__main__":
    # t = threading.Thread(target=start_tensorboard)
    # t.start()
    for i in range(100):
        #for goal_reached_reward in np.linspace(-500, -.1, 7):
        sess = tf.Session()
        n_pars = {}
        n_pars['goal_reached_reward'] = -350
        n_pars['run_name'] = 'Upper_'+ str(uuid.uuid4())
        n_pars['rwd_func'] = move_object
        traind_env, pars = get_trained_env(False, lower_run_suffix="noise_weighted_or_dist", **n_pars)
        try:
            pi = learn_ppo(
                session=sess,
                env=traind_env,
                parameters=pars,
                dir_out="True")
            time.sleep(0.1)
        except ValueError as ve:
            print(ve)
            pass
        print("reset")
        tf.reset_default_graph()
        time.sleep(0.1)
        U.get_session().close()
    exit(0)

# if __name__ == "__main__":
#     t = threading.Thread(target=start_tensorboard)
#     t.start()
#     sess = tf.Session()
#     #env, pars = get_non_trained_env()
#     env, pars = get_trained_env(False)
#     # for logging
#     run_name_suffix = "reachingTest2"
#     pars['run_name'] = run_name_suffix + pars['run_name']
#     pars['nb_iter'] = 200
#     pi = learn_ppo(
#         session=sess,
#         min_trans_per_iter=pars['min_trans_per_iter'],
#         e_clip=pars['e_clip'],
#         run_name=pars['run_name'] ,
#         discount=pars['discount'],
#         lam_trace=pars['lam_trace'],
#         max_v_data_size=pars['max_v_data_size'],
#         epochs_per_iter=pars['epochs_per_iter'],
#         batch_size=pars['batch_size'],
#         nb_iter=pars['nb_iter'],
#         s_dim=env.observation_space.shape[0],
#         a_dim=env.action_space.shape[0],
#         env=env,
#         args_dict=pars,
#         dir_out="True")
#     time.sleep(0.1)
#     print("reset")
#     tf.reset_default_graph()
#     time.sleep(0.1)
#     U.get_session().close()