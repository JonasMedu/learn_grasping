import os
import uuid
import numpy

from setting_utils import rewards
from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerType
from setting_utils.rewards import weighted_or_dist
import pandas as pd

global TB_DIR
TB_DIR = os.path.expanduser('~') + '/Documents/tb/'
uuid_s = 36


class Parmap(object):
    """
    Parameter map controls the experiments parameters.
    In class to enable IDE features
    """
    def __init__(self, **args):
        self.save_name = str(uuid.uuid4())

        self.nb_iter = 300
        self.run_name = self.save_name
        self.extrinsic_reward = -1
        self.goal_reached_reward = -1
        self.time_step = 2e-5
        self.min_trans_per_iter = 3200
        self.e_clip = .2
        self.discount = .99
        self.lam_trace = .95
        self.max_v_data_size = self.min_trans_per_iter * 6
        self.epochs_per_iter = 20
        self.batch_size = 64
        self.rwd_func = weighted_or_dist
        self.index = True
        self.middle = True
        self.small = True
        self.thumb = True
        self.movement_noise = 0.00
        self.entcoeff = 0.2 # only trpo
        self.init_sigma = 1.
        self.min_sigma = 1e-1
        self.mean_mult = 1.
        self.a_lrate = 5e-4
        self.v_lrate = 5e-4
        self.gym_env = None


        vars(self).update(args)

        assert callable(self.rwd_func)

    def get_fingers(self) -> list:
        """
        return a enum list for the fingers which this parameter object deems to be controllable
        :return: list of fingers
        """
        actionable_finger = []
        if self.index:
            actionable_finger.append(AllegroFingerType.INDEX)
        if self.middle:
            actionable_finger.append(AllegroFingerType.MIDDLE)
        if self.small:
            actionable_finger.append(AllegroFingerType.SMALL)
        if self.thumb:
            actionable_finger.append(AllegroFingerType.THUMB)
        return actionable_finger

    def activate_all_fingers(self):
        self.index = True
        self.middle = True
        self.small = True
        self.thumb = True
        return self

    def deactivate_random_finger(self):
        to_deac = numpy.random.choice([self.index, self.middle, self.small, self.thumb])
        to_deac = False

    @staticmethod
    def myBool(val):
        if type(val) == str:
            if 'False' == val:
                return False
            elif 'True' == val:
                return True
            else:
                return numpy.nan
        else:
            print("Unexpected type in boolean conversion: {}".format(
                type(val)
            ))
            return val

    @staticmethod
    def from_config_file(file):
        dtypes = {"save_name": numpy.str,
                  "extrinsic_reward": numpy.float,
                  "time_step": numpy.float,
                  "nb_iter": numpy.int,
                  "min_trans_per_iter": numpy.int,
                  "e_clip": numpy.float,
                  "discount": numpy.float,
                  "lam_trace": numpy.float,
                  "max_kl": numpy.float,
                  "max_v_data_size": numpy.int,
                  "epochs_per_iter": numpy.int,
                  "batch_size": numpy.int,
                  "rwd_name": numpy.str,
                  "index": Parmap.myBool,
                  "middle": Parmap.myBool,
                  "small": Parmap.myBool,
                  "thumb": Parmap.myBool,
                  "movement_noise": numpy.float,
                  "entcoeff": numpy.float,
                  "run_name": numpy.str,
                  "pol_saving_name": numpy.str,
                  "log_name": numpy.str,
                  "ParamObject": numpy.str,
                  "score":numpy.float,
                  "goal_reached_reward":numpy.float,
                  "rwd_func": numpy.str,
                  "min_sigma": numpy.float,
                  "init_sigma": numpy.float,
                  "exploration_sigma": numpy.float,
                  "a_lrate": numpy.float,
                  "v_lrate": numpy.float,
                  "mean_mult": numpy.float,
                  "render_every": numpy.int,
                  "gym_env": numpy.str}

        df = pd.read_csv(file)
        par_dict = {}
        for itm in df.iterrows():
            name = itm[1][0]
            val = itm[1][1]
            par_dict.update({name: dtypes[name](val)})
            if (name == 'rwd_func') or (name == 'rwd_name') :
                # get actual function
                par_dict.update({'rwd_func': rewards.__dict__.get(val)})
        ans = Parmap(**par_dict)
        return ans

    def log_run(self):
        """
        Log the parameter and score to the Tensorboard Log folder
        """
        run_log_score = TB_DIR + self.run_name
        # logger.configure(run_log_score)
        config_file_log = run_log_score + "/config"
        dict = vars(self).copy()
        dict.update({'rwd_func': self.rwd_func.__name__})
        pd.Series(dict).to_csv(config_file_log)


class UpperParmap(Parmap):
    """
    Used in setup which load pre-learned polices
    """
    def __init__(self, parmap=None, **args):
        super(UpperParmap, self).__init__(**vars(parmap), **args)
        self.rwd_func = rewards.test
        self.movement_noise = 0
        self.activate_all_fingers()


class OriginalParams(Parmap):
    """
    proven set of values which works for PPO
    """
    def __init__(self, parmap=None, **args):
        # params
        self.nb_iter = 1000000  # one iter -> at least min_trans_per_iter generated
        self.min_trans_per_iter = 3200
        self.render_every = 100
        self.epochs_per_iter = 20  # for training the v function and updating the policy
        self.exploration_sigma = 1
        self.discount = .99
        self.lam_trace = .95
        self.e_clip = .2  # the 'step size'
        self.batch_size = 64
        super(OriginalParams, self).__init__(**vars(parmap), **args)
