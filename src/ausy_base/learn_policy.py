import os

import gym

from ausy_base import data_handling as dat, policies
from ausy_base.policies import MLPGaussianPolicy
from ausy_base.p3o import PolicyUpdateObjective, P3O
from ausy_base.policy_eval import PolicyEvalV
from setting_utils.param_handler import TB_DIR, Parmap
import numpy as np
import tensorflow as tf
from baselines import logger


def learn_on_env(*,
                 session,
                 env: gym.Env,
                 parameters: Parmap,
                 seed=0,
                 dir_out='exp/',
                 tf_saver=None):
    assert parameters is not None, "You have to pass the runs parameter as paramObject!"
    logger.configure(TB_DIR + parameters.run_name)
    parameters.log_run()
    # seeding, comment out if you do not want deterministic behavior across runs
    # np.random.seed(seed)
    # tf.set_random_seed(seed)
    # mlp for policy

    with tf.compat.v1.variable_scope(parameters.run_name):
        policy = MLPGaussianPolicy(session, env.action_space.shape[0], env.observation_space.shape[0],
                                   mean_mult=parameters.mean_mult,
                                   init_sigma=parameters.init_sigma,
                                   min_sigma=parameters.min_sigma)
    p_update = PolicyUpdateObjective(session, policy, e_clip=parameters.e_clip, a_lrate=parameters.a_lrate)

    # mlp for v_function

    with tf.compat.v1.variable_scope("v_func"):
        vmlp = policies.getMLPdef(1, env.observation_space.shape[0])
    p_eval = PolicyEvalV(session, policy, vmlp, v_lrate=parameters.v_lrate)

    # p3o
    delta_logsig = np.float32((np.log(parameters.init_sigma) - np.log(parameters.min_sigma)) / parameters.nb_iter)
    reduce_exploration_op = tf.compat.v1.assign(policy.min_sigma,
                                      tf.maximum(parameters.min_sigma, policy.min_sigma.value() / tf.exp(delta_logsig)))
    reduce_exploration = lambda: session.run(reduce_exploration_op)
    p3o = P3O(session, policy, p_eval, p_update)
    session.run(tf.compat.v1.global_variables_initializer())
    if not tf_saver:
        tf_saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=parameters.run_name + "/pi"),
                                            max_to_keep=1)
    for it in range(parameters.nb_iter):
        logger.record_tabular("iteration", it + 1)

        # Generates transition data by interacting with the env
        new_paths = dat.rollouts(env, policy=policy.get_action, min_trans=parameters.min_trans_per_iter, render=False)
        new_paths["logprob"] = policy.get_log_proba(new_paths["obs"], new_paths["act"])
        logger.record_tabular("num_trajectories", new_paths["nb_paths"])
        rwd_vals_on_paths_end = new_paths["rwd"][new_paths["done"]]
        logger.record_tabular("reached_rel", np.sum(rwd_vals_on_paths_end != parameters.extrinsic_reward) / new_paths['nb_paths'])  #see test2 reward function
        logger.record_tabular("avg / run", np.sum(new_paths["rwd"]) / new_paths["nb_paths"])
        logger.record_tabular("avg rwd", np.mean(new_paths["rwd"]))
        logger.record_tabular("avg rwd non end", np.mean(new_paths["rwd"][~new_paths["done"]]))
        # logger.record_tabular("mean_tactile_per_state",
        #                       new_paths["obs"][:, -TACTILE_OBS.size:].mean(axis=1).sum() / new_paths["obs"].shape[0])
        # Keep the last max_v_data_size for off-policy policy eval
        if it == 0:
            v_paths = new_paths
        else:
            dat.merge_data(v_paths, new_paths)
            dat.delete_data(v_paths, parameters.max_v_data_size)
        # update policy
        gen_adv = p3o.iteration(v_paths, new_paths, parameters.discount, parameters.lam_trace,
                                parameters.epochs_per_iter, parameters.batch_size, reduce_exploration, verbose=True)
        # log update data
        if dir_out is not None:
            entrop_after = policy.get_entropy(None, None)
            log_act_probas_new = policy.get_log_proba(new_paths["obs"], new_paths["act"])
            diff = np.exp(log_act_probas_new - new_paths["logprob"]) - 1

            #logger.record_tabular("log_act_probas_new", log_act_probas_new)
            logger.record_tabular("diff_mean", diff.mean())
            logger.record_tabular("gen_adv_sum", gen_adv.sum())  # not gen_advantage?

            logger.dump_tabular()
            if it % 30 == 0:
                print("saving " + parameters.run_name)
                # regularly save policy
                tf_saver.save(session, os.path.join(TB_DIR, parameters.run_name) + '/',
                              meta_graph_suffix="meta",
                              write_meta_graph=True,
                              global_step=it)
