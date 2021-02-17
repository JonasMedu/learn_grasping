import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.policy_manager import get_result_df
from performance_analysis.paper_graphs import run_names, MakingGraphs
from setting_utils.param_handler import TB_DIR


def unpack_tuples(list_of_tuples):
    ar_list = []
    for item in list_of_tuples:
        if isinstance(item, list) or isinstance(item, tuple):
            return unpack_tuples(item)
        else:
            ar_list.append(np.array(item))
    return ar_list


def plot_performance():
    perf = pd.read_csv('src/performance_NoMiddle')
    df = perf.dropna(axis=1, how='all').applymap(lambda x: json.loads(x))
    num_pos = df.shape[0]
    df_array = df.applymap(np.array).apply(np.concatenate, axis=1)
    ar_per_position = np.concatenate(df_array).reshape(num_pos, -1)
    print("no middle finger mean length", ar_per_position.mean(axis=1))
    print("no middle finger std", ar_per_position.std(axis=1))

    fig = plt.figure(figsize=(20, 8))
    plt.boxplot(ar_per_position.T)
    plt.title('NoMiddle')
    plt.ylabel('trajectory length')
    plt.show()
    fig.savefig('NoMiddle', bbox_inches='tight')


    perf = pd.read_csv('src/performance_NoThumb')
    df = perf.dropna(axis=1, how='all').applymap(lambda x: json.loads(x))
    num_pos = df.shape[0]
    df_array = df.applymap(np.array).apply(np.concatenate, axis=1)
    ar_per_position = np.concatenate(df_array).reshape(num_pos, -1)
    print("no thumb mean length", ar_per_position.mean(axis=1))
    print("no thumb std", ar_per_position.std(axis=1))

    fig = plt.figure(figsize=(20, 8))
    plt.boxplot(ar_per_position.T)
    plt.title('NoThumb')
    plt.show()
    fig.savefig('NoThumb', bbox_inches='tight')

    perf = pd.read_csv('src/performance_TestReward')
    df = perf.dropna(axis=1, how='all').applymap(lambda x: json.loads(x))
    num_pos = df.shape[0]
    df_array = df.applymap(np.array).apply(np.concatenate, axis=1)
    ar_per_position = np.concatenate(df_array).reshape(num_pos, -1)
    print("upper policy goal on lower pol mean length", ar_per_position.mean(axis=1))
    print("upper policy goal on lower pol  std", ar_per_position.std(axis=1))

    fig = plt.figure(figsize=(20, 8))
    plt.boxplot(ar_per_position.T)
    plt.title('Lower with upper reward function')
    plt.show()
    fig.savefig('Lower_with_upper', bbox_inches='tight')


def plot_upper(pol_tag = 'Upper_'):
    run_names = os.listdir(TB_DIR)
    noUpper_names = [name for name in run_names if name.find(pol_tag)>-1]
    ar_list = []
    for name in noUpper_names:
        upper_df = pd.read_csv(os.path.join(TB_DIR, name, 'progress.csv'))
        ar_list.append(upper_df.to_numpy())
    # upper_df.columns
    # Index(['action_entropy', 'avg_rwd_return', 'diff_mean', 'gen_adv_sum',
    #    'iteration', 'mean_tactile_per_state', 'num_trajectories',
    #    'object distance (pos + or)', 'reached_rel',
    #    'v-function_after_training', 'v-function_before_training'],
    #   dtype='object')
    super_ar = np.stack(ar_list)
    print("Upper policy mean # trajectories per run along train cycles", super_ar[:, :, 6].mean(axis=0))
    print("Upper policy std # trajectories per run along train cycles", super_ar[:, :, 6].std(axis=0))
    ###
    fig = plt.figure(figsize=(20, 8))
    plt.boxplot(super_ar[:, :, 6])
    plt.title('Upper policy # trajectories per run along train cycles')
    plt.show()
    fig.savefig('Upper_policy_trajectories', bbox_inches='tight')

    print("Upper policy mean distance object traveled per run along train cycles", super_ar[:, :, 7].mean(axis=0))
    print("Upper policy std  distance object traveled per run along train cycles", super_ar[:, :, 7].std(axis=0))

    fig = plt.figure(figsize=(20, 8))
    plt.boxplot(super_ar[:, :, 7])
    plt.title('Upper policy distance object traveled per run along train cycles')
    plt.show()
    fig.savefig('Upper_policy_distance', bbox_inches='tight')


def plot_time_step_and_score(lower_pi_suffix='Lower_angular'):
    df = get_result_df(lower_pi_suffix=lower_pi_suffix)
    # mask iterations with something learned
    mask = df['score'] < df['nb_iter']
    plt.figure()
    plt.scatter(df.loc[mask, 'time_step'], df.loc[mask, 'score_p'])
    plt.xlabel('time step')  # also action: mean angle dt
    plt.ylabel('mean number trajectories: first - last; (max size {})'.format(max(df.nb_iter)))
    plt.title("step size and scores")
    plt.show()
    # ---> 1e-5 is a adequate step size. 2.3e-5 is the maximum step size.
    # --> 10k hz is the minimum frequency in which the policy learns?


class DeprecatedThings(object):
    @staticmethod
    def image_per_tag():
        # size time string and run iid = 60
        only_tags = set([name[:-61] for name in run_names])
        for tag in only_tags:
            try:
                """
                    prints the no Middle finger run
                    """
                noMiddle_names = [name for name in run_names if name.find(tag) > -1]
                ser_list = [MakingGraphs.get_progress_array(os.path.join(TB_DIR, name), name) for name in
                            noMiddle_names]
                # filter for malicious runs.
                ser_list = [ser for ser in ser_list if ser]
                # construct data frames
                progesses = [pd.DataFrame(ser['progress']) for ser in ser_list]
                # get hyperparameters, assumtion, all runs have the same.
                pars = ser_list[0]['params']
                state_transactions_per_iteration = int(pars['min_trans_per_iter'])
                id_dict = MakingGraphs.get_column_id_dict(progesses)
                # column sequence
                # ['action_entropy', 'avg_rwd_return', 'diff_mean', 'gen_adv_sum',
                #        'iteration', 'mean_tactile_per_state', 'num_trajectories',
                #        'object distance (pos + or)', 'reached_rel',
                #        'v-function_after_training', 'v-function_before_training']
                # into 3 dim
                ar = np.array([prog.to_numpy() for prog in progesses if prog.shape[0] == int(pars.nb_iter)])
                # # action entropy ## derpricated
                m = ar[:, :, 0].mean(0)
                t = range(0, len(m))
                # sd = ar[:, :, 0].std(0)
                # fig, ax = plt.subplots(1)
                # ax.plot(t, m, lw=2, label='action entropy 95% intvl', color='blue')
                # ax.fill_between(t, m+sd, m-sd, facecolor='blue', alpha=0.5)
                # ax.set_title('action entropy')
                # plt.show()

                # num_trajectories
                num_trajectories_id = id_dict['num_trajectories']
                m = ar[:, :, num_trajectories_id].mean(0)
                sd = ar[:, :, num_trajectories_id].std(0)
                fig, ax = plt.subplots(1)
                ax.plot(t, m, lw=2, label='sim steps until drop', color='green')
                ax.fill_between(t, m + sd, np.clip(m - sd, 0, np.inf), facecolor='green', alpha=0.5,
                                label='95% intvl.')
                ax.set_ylabel('Number of trajectories per run')
                ax.setxlabel('Training iteration')
                ax.set_title('Learning progression')
                plt.legend()
                fig.savefig(tag + '.png', dpi=300)
                plt.show()
                pass
            except Exception as e:
                print(tag, e)



def initial_hand_pos(hand):
    u_pos = []
    i_k = 0
    for ft in AllegroFingerType:
        for jt in AllegroFingerJointType:
            u_pos.append(hand.fingers[ft].joints[jt].initial_position)
            i_k += 1
    return np.array(u_pos)


def compute_input_as_map(joint_dict: dict, u, finger_map) -> dict:
    target_position_map = joint_dict.copy()
    i_k = 0
    for ft in finger_map:
        for jt in AllegroFingerJointType:
            target_position_map[ft][jt] = joint_dict[ft][jt] + u[i_k]
            i_k += 1
    return target_position_map
