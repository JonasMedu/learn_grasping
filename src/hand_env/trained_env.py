import pybullet
from gym import spaces

from setting_utils import param_handler
from hand_env.allegro_env import AllegroHand
from hand_env.hand_interface import get_joint_state, tactile_state, set_joint_pos, TACTILE_OBS, \
    build_action_space_from_type_list, compute_input
import numpy as np
from setting_utils.param_handler import UpperParmap


class TrainedEnv(AllegroHand):
    def __init__(self, pi, paramObject=None, gui=False):
        """
        Environmental Class which takes a pre-trained policy to transforms the transition function
        T(S, A, S') of an MDP into T(pi(S,A),A, S')
        :param pi: pre-trained policy
        :param paramObject: given ParamMap to train
        :param maxTorque: for the Allegrohand
        :param gui: Displays the pybullet Interface to watch the hand.
        """
        assert isinstance(paramObject, param_handler.Parmap), "Need to know at least the lower policy action space of the lower pretrained " \
                                     "policy. "

        self.fingerMap_lower = paramObject.get_fingers().copy()
        pars_upper = UpperParmap(paramObject)
        super(TrainedEnv, self).__init__(pars_upper, gui=gui)
        self.lower_pi = pi

    def low_policy_step(self, tactile_state):
        full_state_lower = np.concatenate([self.joint_state, tactile_state])
        u_inner = self.lower_pi(full_state_lower)
        return u_inner

    def step(self, u):
        # policy_action does not have to be clipped
        # self.state is current position

        self.dets_reward.old_obj_pos, self.dets_reward.old_obj_or = self.pc.call(pybullet.getBasePositionAndOrientation,
                                                                                 self.bar.body_unique_id)

        joint_input = compute_input(self.joint_state, u, self.fingerMap)

        set_joint_pos(self.hand, joint_input, self.fingerMap, 0) # no movement noise in upper policy

        self.pc.step_simulation()

        self.joint_state = get_joint_state(self.hand)
        ts = tactile_state(self.hand)
        full_state = self.state_for_policy_input(ts)

        self.dets_reward.policy_action.append(u)
        self.dets_reward.tactile_state.append(ts)

        reward, done = self.reward()

        # time.sleep(0.1)
        return full_state, reward, done, 0

    def state_for_policy_input(self, ts):
        lower_output = self.low_policy_step(ts)
        full_state = np.concatenate([lower_output, self.joint_state, ts])
        return full_state

    def define_observation_space(self):
        tactile_low = TACTILE_OBS
        tactile_high = np.ones(TACTILE_OBS.shape) * 100

        obs_from_lower_low, obs_from_lower_high = build_action_space_from_type_list(self.fingerMap_lower)
        obs_from_higher_low, obs_from_higher_high = build_action_space_from_type_list(self.paramMap.get_fingers())
        self.observation_space = spaces.Box(
            # first, input form lower pi, than current finger position, than tactile observations
            np.concatenate([obs_from_lower_low, obs_from_higher_low, tactile_low]),
            # low
            np.concatenate([obs_from_lower_high, obs_from_higher_high, tactile_high]),
            # high
            dtype=np.float64)

