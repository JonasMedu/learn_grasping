from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv

from setting_utils.param_handler import Parmap

from ausy_base.learn_policy import learn_on_env as learn_ppo
import time
import tensorflow as tf


if __name__ == '__main__':
    pars = Parmap()
    #env = Walker2DBulletEnv(render=True)
    env = HumanoidBulletEnv(render=False)
    # for logging
    t = time.asctime().replace(" ", "_").replace(":", "_")
    run_name_suffix = "PPO_test_{}_".format(str(t))
    pars.run_name = run_name_suffix + pars.run_name

    tf.reset_default_graph()
    sess = tf.Session()
    pars.nb_iter = 300000
    learn_ppo(
        session=sess,
        env=env,
        parameters=pars,
        dir_out="True")
