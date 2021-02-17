import pybullet

from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerJointType
from hand_env.allegro_env_unsupervised import AllegroCollisionFromTactileHand
from hand_env.hand_interface import TACTILE_OBS, OTHER_LOW, THUMB_LOW, THUMB_HIGH, OTHER_HIGH, set_joint_pos_direct, \
    get_joint_state, tactile_state
from setting_utils.param_handler import Parmap
from gym import spaces

import numpy as np

from setting_utils.positions import position_list


class AllegronNoiseFromUpper(AllegroCollisionFromTactileHand):

    def __init__(self, allegros_params: Parmap, gui=False, show_fingers=False):
        self.noise_magnitude = allegros_params.movement_noise
        self.noise_frequency = 10  # every 10 time steps of this lower policy the upper policy recomputes its planning task
        self.upper_target = None
        super(AllegronNoiseFromUpper, self).__init__(allegros_params, gui, show_fingers)

    def define_observation_space(self):
        """
        defines state space / observation space box
        Lower observation has always all fingers as input disregarding the amount of fingers used.
        :return: None
        """
        tactile_low = TACTILE_OBS
        tactile_high = np.ones(TACTILE_OBS.shape) * 100
        noise_from_upper_low = np.ones(len(self.fingerMap) * len(AllegroFingerJointType)).ravel() * -1
        noise_from_upper_high = np.ones(len(self.fingerMap) * len(AllegroFingerJointType)).ravel()
        self.observation_space = spaces.Box(
            np.concatenate((np.array(OTHER_LOW * 3 + THUMB_LOW).ravel(),
                            tactile_low,
                            noise_from_upper_low)),
            # low
            np.concatenate((np.array(OTHER_HIGH * 3 + THUMB_HIGH).ravel(),
                            tactile_high,
                            noise_from_upper_high)),
            # high
            dtype=np.float64)

    def step(self, u):
        # u does not have to be clipped
        # self.state is current position
        # joint_input = compute_input_as_map(self.hand.get_finger_joint_angles(), u, self.paramMap.get_fingers())

        # 1 is bar -> need actually ids from fingertips?
        self.dets_reward.old_obj_pos, self.dets_reward.old_obj_or = self.pc.call(pybullet.getBasePositionAndOrientation,
                                                                                 self.bar.body_unique_id)

        u = self.compute_upper_noise(u)

        set_joint_pos_direct(self.hand, u, self.fingerMap)

        self.pc.step_simulation()

        self.joint_state = get_joint_state(self.hand)
        ts = tactile_state(self.hand)
        full_state = self.state_for_policy_input(ts)
        self.dets_reward.policy_action.append(u)
        self.dets_reward.tactile_state.append(ts)
        reward, done = self.reward()
        self.reset_upper_policy_intervention()
        # time.sleep(0.1)
        return full_state, reward, done, 0

    def compute_upper_noise(self, u):
        current_state = get_joint_state(self.hand)
        u += ((current_state-self.upper_target) / self.noise_frequency)
        return u

    def reset(self):
        # Set of states M as in set of states from which to learn the MDP from
        pos_set_func = np.random.choice(position_list)
        self = pos_set_func(self)

        self.joint_state = get_joint_state(self.hand)
        # for reward
        self.dets_reward.init_joint_pos = self.joint_state.copy()
        self.dets_reward.init_obj_pos = np.array(self.bar.base_link.initial_position)
        self.dets_reward.init_obj_or = np.array(self.bar.base_link.initial_orientation)
        self.dets_reward.counter = 0

        ts = tactile_state(self.hand) + 1  # debug fix.
        # draw new noise
        self.dets_reward.counter = 0
        self.upper_target = self.upper_policy_planning_target()
        full_state = self.state_for_policy_input(ts)

        return full_state

    def upper_policy_planning_target(self):
        """simulation from the upper policy; as noise"""
        current_joint_state = get_joint_state(self.hand)
        upper_plan = np.random.random(len(self.fingerMap) * len(AllegroFingerJointType)) * self.noise_magnitude
        return current_joint_state - upper_plan

    def state_for_policy_input(self, ts):
        full_state = np.concatenate([self.joint_state, ts, self.upper_target])
        return full_state

    def reset_upper_policy_intervention(self):
        if self.dets_reward.counter % self.noise_frequency == 0:
            self.upper_target = self.upper_policy_planning_target()
        self.dets_reward.counter += 1
