"""
File holds methods used to analyse trained policies.
"""

import matplotlib
import pybullet
from PIL import Image

from scripts.policy_manager import loadParamMap, load_trained_model
from ausy_base.policies import MLPGaussianPolicy
from hand_env.envs import allegro_gyms
from performance_analysis.utils import extract_list, run_lower_with_environment

matplotlib.use('TkAgg')

import os
import random
import pandas as pd
import tensorflow as tf
import baselines.common.tf_util as U
from hand_env.allegro_env import AllegroHand
from setting_utils.param_handler import Parmap, TB_DIR, OriginalParams
import numpy as np
from setting_utils.positions import position_list, pos_test, all_pos_list


def run_lower_take_picture(env, policy, image_name, position=pos_test):
    """takes pictures after a series of steps"""
    env = position(env)
    obs = env.reset()
    t = 0
    for step_size in [5, 45, 50]:  # after 5, 50 and 100 setps
        for k in range(step_size):
            act = policy(obs)
            obs, rwd, done, _ = env.step(act)
            t += 1
        camera_image = env.pc.call(pybullet.getCameraImage, 1600, 1600)
        rbg_image = camera_image[2]
        img = Image.fromarray(rbg_image[:, :, 0:3])  # discard alpha channel
        img.save(image_name + '_' + position.__name__ + '_' + str(t) + '.png')


def run_lower_take_picture_and_return_actions(env, policy, image_name, position=pos_test, take_photos=False):
    """takes pictures after a series of steps"""
    env = position(env)
    obs = env.reset()
    t = 0
    actions =[]
    for step_size in [5, 45, 50]:  # after 5, 50 and 100 setps
        for k in range(step_size):
            act = policy(obs)
            obs, rwd, done, _ = env.step(act)
            t += 1
            actions.append(act)
        if take_photos:
            camera_image = env.pc.call(pybullet.getCameraImage, 1600, 1600)
            rbg_image = camera_image[2]
            img = Image.fromarray(rbg_image[:, :, 0:3])  # discard alpha channel
            img.save(image_name + '_' + position.__name__ + '_' + str(t) + '.png')
    return actions


def make_policy_evaluation_images(pol_tag):
    run_names = os.listdir(TB_DIR)
    noMiddle_names = [name for name in run_names if name.lower().find(pol_tag.lower()) > -1]
    for name in noMiddle_names:
        # if not validate_if_image_taken(name):
        tf.reset_default_graph()
        sess = tf.Session()
        params = loadParamMap(name)
        params.rwd_func = lambda x: (0, False)
        if params.gym_env:
            env = allegro_gyms[params.gym_env](params, gui=True, show_fingers=False)
        else:
            # fallback to old
            env = AllegroHand(params, gui=True)
        policy = load_trained_model(TB_DIR + name, name, sess)
        try:
            for pos in all_pos_list:
                run_lower_take_picture(env, policy, name, position=pos)
        except Exception as et:
            print(name, et)
            pass
        env.close()
        U.get_session().close()
    print("Done")


def make_policy_evaluation_imagesV2(pol_tag):
    run_names = os.listdir(TB_DIR)
    noMiddle_names = [name for name in run_names if name.lower().find(pol_tag.lower()) > -1]
    for name in noMiddle_names:
        # if not validate_if_image_taken(name):
        tf.reset_default_graph()
        sess = tf.Session()
        params = loadParamMap(name)
        params.rwd_func = lambda x: (0, False)
        if params.gym_env:
            env = allegro_gyms[params.gym_env](params, gui=True, show_fingers=False)
            policy = load_trained_model(TB_DIR + name, name, sess)
            try:
                for pos in all_pos_list:
                    run_lower_take_picture(env, policy, name, position=pos)
            except Exception as et:
                print(name, et)
                pass
            env.close()
        U.get_session().close()
    print("Done")


def make_policy_evaluation_noise(pol_tag='NoisyEnv_', take_photos=False):
    run_names = os.listdir(TB_DIR)
    noMiddle_names = [name for name in run_names if name.lower().find(pol_tag.lower()) > -1]
    act_potentials = []
    for name in noMiddle_names:
        # if not validate_if_image_taken(name):
        tf.reset_default_graph()
        sess = tf.Session()
        params = loadParamMap(name)
        noise = params.movement_noise
        params.rwd_func = lambda x: (0, False)
        if params.gym_env:
            env = allegro_gyms[params.gym_env](params, gui=take_photos, show_fingers=False)
            policy = load_trained_model(TB_DIR + name, name, sess)
            try:
                for pos in all_pos_list:
                    actions = run_lower_take_picture_and_return_actions(env, policy, name, position=pos, take_photos=take_photos)
                    mn = np.mean(actions)
                    std = np.std(actions)
                    act_potentials.append({'name': name, 'mean': mn, 'std': std, 'noise': noise})
            except Exception as et:
                print(name, et)
                pass
            env.close()
        U.get_session().close()
    if take_photos:
        pd.DataFrame(act_potentials).to_csv('Noise_action_potentials_absolute', index=False)
        print("Done")
    else:
        return pd.DataFrame(act_potentials)


def get_policy_performance(pol_tag):
    run_names = os.listdir(TB_DIR)
    policy_tag_names = [name for name in run_names if name.lower().find(pol_tag.lower()) > -1]
    mdim_vals = {}
    for name in policy_tag_names:
        name_pathl = {}
        # try:
        tf.reset_default_graph()
        sess = tf.Session()
        params = loadParamMap(name)
        env = AllegroHand(params, gui=True)
        num_tra = 100
        policy = load_trained_model(TB_DIR + name, name, sess)
        for position_func in all_pos_list:
            nb_traj_length = run_lower_with_environment(position_func, env, policy, num_trajectories=num_tra)
            # nb_traj_length = run_lower_policy(position_func, params, policy, num_trajectories=num_tra, show=True)
            name_pathl.update({position_func.__name__: nb_traj_length})
        env.close()
        U.get_session().close()
        # except Exception as Id:
        #     print(name, 'not found in meta')
        mdim_vals.update({name: name_pathl})
    return mdim_vals


def random_sample():
    # mean trajectory length over 100 trials for a newly initialized policy
    #                random_sample_policy
    # pos1                           6.28
    # pos2                           6.47
    # pos3                           6.30
    # pos_test                       6.94
    # resetHandEasy                  5.78

    parameters = OriginalParams(Parmap(time_step=1e-2, nb_iter=500))
    env = AllegroHand(parameters)
    mdim_vals = {}
    name_pathl = {}
    # try:
    tf.reset_default_graph()
    sess = tf.Session()
    num_tra = 100
    with tf.compat.v1.variable_scope("random_sample_policy"):
        policy = MLPGaussianPolicy(sess, env.action_space.shape[0], env.observation_space.shape[0],
                                   mean_mult=parameters.mean_mult,
                                   init_sigma=parameters.init_sigma,
                                   min_sigma=parameters.min_sigma)
    sess.run(tf.compat.v1.global_variables_initializer())
    all_positions = position_list.copy()
    all_positions.append(pos_test)
    for position_func in all_positions:
        nb_traj_length = run_lower_with_environment(position_func, env, lambda obs: policy.get_action(obs), num_trajectories=num_tra)
        # nb_traj_length = run_lower_policy(position_func, params, policy, num_trajectories=num_tra, show=True)
        name_pathl.update({position_func.__name__: nb_traj_length})
    env.close()
    U.get_session().close()
    # except Exception as Id:
    #     print(name, 'not found in meta')
    mdim_vals.update({"random_sample_policy": name_pathl})
    return pd.DataFrame(mdim_vals)


def read_performance_array(run_tag):
    """    result mean traj. length for each position:
    0    73.6565
    1    73.6330
    2    69.4670
    3    73.4770
    4    70.8805 %% test position
    """
    perf_df = pd.read_csv('performance_'+run_tag, index_col='Unnamed: 0')
    m_perf = perf_df.applymap(extract_list).applymap(np.mean).mean(axis=0)
    high_performance = perf_df.applymap(extract_list).applymap(np.mean).loc[:, m_perf.index]
    mean_per_pos = high_performance.mean(axis=1)
    mean_per_pos.name = 'mean_per_pos'
    std_per_pos = high_performance.std(axis=1)
    std_per_pos.name = 'std_per_pos'
    return pd.DataFrame([mean_per_pos, std_per_pos])


# if __name__ == '__main__':
#     tag = 'TESTING_or_distThu_Nov_26_16_17'
#     df = read_performance_array(tag)
#     print(df)

if __name__=='__main__':
    random.seed(a=123)
    pol_tag = 'NoisyEnv_'
    # make_policy_evaluation_imagesV2(pol_tag)
    make_policy_evaluation_noise(pol_tag=pol_tag, take_photos=True)
    print("Done")
#
# if __name__ == '__main__':
#     pol_tag = 'TESTING_or_dist'
#     performance_dict = get_policy_performance(pol_tag)
#     performance_df = pd.DataFrame(performance_dict)
#     print(performance_df)
#     performance_df.to_csv('performance_%s' %pol_tag, index=True)
#     time.sleep(3)
#     exit(0)
