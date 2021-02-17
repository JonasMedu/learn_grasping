import numpy as np
from ausy_base.logs import MyLogger
logger = MyLogger().logger


def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start:min(batch_start + batch_size, data_set_size)]


def rollout(env, policy, render=False):
    # Generates transitions until episode's end
    obs = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        act = policy(obs)
        nobs, rwd, done, _ = env.step(act)
        yield obs, act, rwd, done
        obs = nobs

def rollouts(env, policy, min_trans, render=False):
    # Keep calling rollout and saving the resulting path until at least min_trans transitions are collected
    keys = ["obs", "act", "rwd", "done"]  # must match order of the yield above
    paths = {}
    for k in keys:
        paths[k] = []
    nb_paths = 0
    while len(paths["rwd"]) < min_trans:
        for trans_vect in rollout(env, policy, render):
            for key, val in zip(keys, trans_vect):
                paths[key].append(val)
        nb_paths += 1
    for key in keys:
        paths[key] = np.asarray(paths[key])
        if paths[key].ndim == 1:
            paths[key] = np.expand_dims(paths[key], axis=-1)
    paths["nb_paths"] = nb_paths
    return paths


def merge_data(d1, d2):
    for key in d1.keys():
        if key == "nb_paths":
            d1[key] += d2[key]
        else:
            d1[key] = np.append(d1[key], d2[key], axis=0)


def delete_data(d1, max_len):
    if d1["obs"].shape[0] > max_len:  # we need to delete. find where paths end
        end_paths = [t for t, done in enumerate(d1["done"]) if done]
        nb_path_to_del = next(t for t, p_length in enumerate(end_paths) if d1["obs"].shape[0] - p_length <= max_len) + 1
        if nb_path_to_del == len(end_paths):
            logger.warning('Warning: max_len is too low, will end up deleting all data. Keeping last path instead')
            nb_path_to_del -= 1
        for key in d1.keys():
            if key == "nb_paths":
                d1[key] -= nb_path_to_del
            else:
                d1[key] = d1[key][(end_paths[nb_path_to_del - 1] + 1):]
