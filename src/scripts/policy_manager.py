import os

import numpy as np
import pandas as pd
import tensorflow as tf

from setting_utils.param_handler import TB_DIR, Parmap, uuid_s


def get_result_df(lower_pi_suffix='Lower_angular'):
    pis = os.listdir(TB_DIR)
    # filter for suffix
    possible_policies = [name for name in pis if name.find(lower_pi_suffix) > -1]
    # get policy names
    policy_names = [name[len(lower_pi_suffix):len(lower_pi_suffix)+uuid_s] for name in possible_policies if name.find('data')>-1]
    # tb folder
    folder_names = []
    for pol in possible_policies:
        [folder_names.append(folder) for folder in os.listdir(TB_DIR) if folder.find(pol)>-1]
    assert folder_names, "There are no available polcies with the prefix %s" %lower_pi_suffix
    pars_gatherer = []
    # read configurations and scores
    for folder in folder_names:
        params = loadParamMap(folder)
        pars_gatherer.append(vars(params))
    running_scores = pd.DataFrame(pars_gatherer)
    return running_scores


def loadParamMap(folder, uuid_s=36):
    run_folder = os.path.join(TB_DIR, folder)
    params = Parmap.from_config_file(run_folder + "/config")
    num_traj_score = read_num_trajectories_score(run_folder)
    traj_progress = read_num_trajectories_progress(run_folder)
    params.score = num_traj_score
    params.score_p = traj_progress
    params.log_name = folder
    params.pol_saving_name = folder[-uuid_s:]
    params.ParamObject = params
    return params


def read_num_trajectories_score(log_folder, file_name="progress.csv"):
    try:
        df = pd.read_csv(log_folder+"/"+ file_name)
        return df["num_trajectories"].rolling(window=4).mean().iloc[-1]
    except Exception:
        print("Can not get score for {}.".format(
            log_folder
        ))
    return np.nan


def read_num_trajectories_progress(log_folder, file_name="progress.csv"):
    try:
        df = pd.read_csv(log_folder+"/"+ file_name)
        asdf = df["num_trajectories"].rolling(window=4).mean()
        return asdf.iloc[3] - asdf.iloc[-1]
    except Exception:
        print("Can not get score for {}.".format(
            log_folder
        ))
    return np.nan


def load_trained_model(pols_home, lower_run_name, session):
    """
    :param lower_run_name: name of the policy
    :param pols_home: dir in which policy is saved
    :param session: tf session
    :return:
    """
    print('Restoring model')
    files = os.listdir(pols_home)
    file = [met for met in files if met.find('.meta') > - 1].pop()
    file_path = os.path.join(pols_home, file)[:-5]
    saver = tf.train.import_meta_graph(file_path + '.meta')
    saver.restore(session, file_path)
    # after retreving the variables, mark them as not learnable
    tf.get_default_graph().clear_collection(session.graph)
    act_tensor = session.graph.get_tensor_by_name(lower_run_name + '/pi/policy_output:0')
    obs_tensor = session.graph.get_tensor_by_name(lower_run_name + '/pi/x:0')
    act_std = session.graph.get_tensor_by_name(lower_run_name + '/pi/logstd:0')

    # policy = lambda obs: np.squeeze(session.run('policy/policy_output:0', {'policy/x:0': np.asmatrix(obs)}), axis=0)
    policy = lambda obs: np.squeeze(session.run(act_tensor, {obs_tensor: np.asmatrix(obs)}), axis=0)
    return policy