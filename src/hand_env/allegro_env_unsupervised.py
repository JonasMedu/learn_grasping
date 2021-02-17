import pybullet
from gym import spaces

import hand_env.hand_interface
from hand_env.allegro_env import AllegroHand
from hand_env.hand_interface import get_joint_state, tactile_state_dict, THUMB_LOW, THUMB_HIGH, OTHER_LOW, OTHER_HIGH, \
    TACTILE_OBS
import numpy as np

from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerType

from setting_utils.param_handler import Parmap

FINGER_TO_LINK_ID = {AllegroFingerType.INDEX:  4,
                     AllegroFingerType.MIDDLE: 9,
                     AllegroFingerType.SMALL: 14,
                     AllegroFingerType.THUMB: 19}
LINK_ID_TO_FINGER = {4: AllegroFingerType.INDEX,
                     9: AllegroFingerType.MIDDLE,
                     14: AllegroFingerType.SMALL,
                     19: AllegroFingerType.THUMB}


class AllegroCollisionFromTactileHand(AllegroHand):

    def __init__(self, allegros_params: Parmap, gui=False, show_fingers=False):
        self.objects_with_collision = allegros_params.get_fingers()
        super(AllegroCollisionFromTactileHand, self).__init__(allegros_params, gui, show_fingers)

    def collisions_not_object(self) -> dict:
        """collisions of fingers, which are not between fingers and bar"""
        collision_list = self.pc.call(pybullet.getContactPoints)
        number_collisions = len(collision_list)
        collisions = dict(zip(self.objects_with_collision, np.zeros(len(self.objects_with_collision))))
        if number_collisions > 0:
            for entry in collision_list:
                if entry[3] in FINGER_TO_LINK_ID.values() and (entry[2] != self.bar_link_id):
                    collisions[LINK_ID_TO_FINGER[entry[3]]] = entry[9] > 0.01
        return collisions

    def show_collisions(self):
        ts_dict = tactile_state_dict(self.hand)
        if self.gui_connected:
            for finger in AllegroFingerType:
                blue = [0, 0, 255] if np.sum(ts_dict[finger]) > 0 else [0, 0, 0]
                # # One needs to press "w" to activate the wireframe and see the colors
                self.pc.call(pybullet.setDebugObjectColor, 0, FINGER_TO_LINK_ID[finger], blue)

    def reward(self):

        self.dets_reward.tactile_state_dict = hand_env.hand_interface.tactile_state_dict(self.hand)

        self.dets_reward.current_minus1_joint_position = self.dets_reward.current_joint_position

        self.dets_reward.current_joint_position = get_joint_state(self.hand)
        self.dets_reward.new_obj_pos, self.dets_reward.new_obj_or = self.pc.call(pybullet.getBasePositionAndOrientation,
                                                                                 self.bar.body_unique_id)
        self.dets_reward.collisions_not_bar_dict = self.collisions_not_object()
        # probably not right here, but well
        self.show_collisions()

        self.dets_reward.counter += 1

        return self.paramMap.rwd_func(self.dets_reward)


class AllegroCollisionPositionHand(AllegroCollisionFromTactileHand):
    """
    AllegroCollisionFromTactileHand which includes position, orientation of the object
    """
    def __init__(self, allegros_params: Parmap, gui=False, show_fingers=False):
        super(AllegroCollisionPositionHand, self).__init__(allegros_params, gui, show_fingers)

    def state_for_policy_input(self, ts):
        obj_pos, obj_or = self.pc.call(pybullet.getBasePositionAndOrientation, self.bar.body_unique_id)
        full_state = np.concatenate([self.joint_state, ts, obj_pos, obj_or])
        return full_state

    def define_observation_space(self):
        """
        defines state space / observation space box
        Lower observation has always all fingers as input disregarding the amount of fingers used.
        :return: None
        """
        tactile_low = TACTILE_OBS
        tactile_high = np.ones(TACTILE_OBS.shape) * 100
        obj_pos_low = np.ones(3) * -1
        obj_pos_high = np.ones(3) * 3
        obj_or_low = np.ones(4) * -1
        obj_or_high = np.ones(4)
        self.observation_space = spaces.Box(
            np.concatenate((np.array(OTHER_LOW * 3 + THUMB_LOW).ravel(), tactile_low, obj_pos_low, obj_or_low)),
            # low
            np.concatenate((np.array(OTHER_HIGH * 3 + THUMB_HIGH).ravel(), tactile_high, obj_pos_high, obj_or_high)),
            # high
            dtype=np.float64)