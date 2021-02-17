"""
Utilities for the performance analysis
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from ausy_base import data_handling as dat
from hand_env.allegro_env import AllegroHand
from setting_utils.param_handler import TB_DIR, Parmap

#  Returns tuple of handles, labels for axis ax, after reordering them to conform to the label order `order`, and if unique is True, after removing entries with duplicate labels.
# source: https://stackoverflow.com/a/35926913/3707039
def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x, k in zip(seq, key) if not (k in seen or seen_add(k))]
def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    return(handles, labels)


def extract_list(list):
    """ to be able to read from get_policy_performance().."""
    return np.array(list.strip('][').split(', ')).astype(int)


def find_upper_policy_run_suffixes():
    folder_names = os.listdir(TB_DIR)
    par_list = []
    for folder in folder_names:
        try:
            params = Parmap.from_config_file(TB_DIR + folder + "/config")
            if len(params.get_fingers()) == 4:
                par_list.append(params)
        except KeyError as keyr:
            print(folder)
            print(keyr)
    print('implement me')


def run_lower_policy(position_setter, parMap, policy, num_trajectories=100, show=False):
    # since the env is here only used to compote the step(action), we do not need the TrainedEnv
    env = AllegroHand(allegros_params=parMap, gui=show, show_fingers=False)
    pathsl = []
    for i in range(num_trajectories):
        env = position_setter(env)
        done = False
        traj_length = 0
        while not done:
            for trans_vect in dat.rollout(env, policy, render=False):
                traj_length +=1
                obs, act, rwd, done = trans_vect
        pathsl.append(traj_length)
        env.reset()
    return pathsl


def run_lower_with_environment(position_setter, env, policy, num_trajectories=100):
    # since the env is here only used to compote the step(action), we do not need the TrainedEnv
    pathsl = []
    for i in range(num_trajectories):
        env = position_setter(env)
        done = False
        traj_length = 0
        while not done:
            for trans_vect in dat.rollout(env, policy, render=False):
                traj_length += 1
                obs, act, rwd, done = trans_vect
        pathsl.append(traj_length)
        env.reset()
    return pathsl


def validate_if_image_taken(name):
    all_things = os.listdir()
    file_is_pol_im = lambda x: x.find(name) > -1
    return np.sum([file_is_pol_im(file) for file in all_things]) > 0