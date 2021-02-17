"""
Helper functions to compute rewards from the current state of the simulation
"""
import numpy as np
from scipy.spatial.distance import cosine

from setting_utils.positions import goaling_position, goaling_orientation


def angular_similarity(x, y):
    # https://en.wikipedia.org/wiki/Cosine_similarity -> angular similarity
    angular_distance = np.arccos(cosine(x, y)) / np.pi
    return 1 - angular_distance


class DetailsForReward(object):
    """
    Object holds data for the rewards.
    Enables You to make heavily use of code completion features of Your IDE.
    """
    def __init__(self, action_space=0, observation_space=0):
        self.action_space = action_space
        self.observation_space = observation_space
        self.goal_reached_reward = None
        self.extrinsic_reward = None
        self.min_trans_per_iter = None
        self.init_obj_pos = None
        self.init_obj_or = None
        self.init_joint_pos = None
        self.old_obj_or = None
        self.old_obj_pos = None
        self.new_obj_pos = None
        self.new_obj_or = None
        self.current_joint_position = None
        self.current_joint_velocities = None
        self.current_minus1_joint_position = None
        self.policy_action = []
        self.tactile_state = []
        self.goal_pos_orientation = GoalPositionOrientation(obj_pos=goaling_position, obj_or=goaling_orientation)
        self.action_fingers = []
        self.tactile_state_dict = {}
        self.collision_dict = {}
        self.collisions_not_bar_dict = {}
        self.counter = 0


class GoalPositionOrientation(object):
    def __init__(self, **args):
        if args['obj_pos']:
            self.obj_pos = args['obj_pos']
        else:
            self.obj_pos = None
        if args['obj_or']:
            self.obj_or = args['obj_or']
        else:
            self.obj_or = None


def all_action_fingers_have_contact(rwdobj: DetailsForReward, tactile_threshold=0.05) -> bool:
    """
    checks the tactile sensors of the fingers
    :param rwdobj: reward object
    :param tactile_threshold: of sum of tactile state to return false
    :return:
    """
    assert len(rwdobj.action_fingers) > 1, "Found less than two fingers in action finger list."
    for finger in rwdobj.action_fingers:
        if np.sum(rwdobj.tactile_state_dict[finger]) < tactile_threshold:
            return False
    return True


def any_action_finger_has_contact(rwdobj: DetailsForReward, tactile_threshold=0.05) -> bool:
    """
    checks the tactile sensors of the fingers
    :param rwdobj: reward object
    :param tactile_threshold: of sum of tactile state to return false
    :return:
    """
    assert len(rwdobj.action_fingers) > 1, "Found less than two fingers in action finger list."
    return np.sum([np.sum(rwdobj.tactile_state_dict[finger]) for finger in rwdobj.action_fingers]) > tactile_threshold


def for_weighted_connect(rwdobj):
    # number of fingers connected from tactile feedback
    tac_threshold = .005
    num_fingers_connected = np.sum(
        [np.sum(rwdobj.tactile_state_dict[finger]) > tac_threshold for finger in rwdobj.action_fingers])
    object_in_grasp = num_fingers_connected > 1  # at least two fingers
    obj_fell = (sum(rwdobj.collisions_not_bar_dict.values()) > 0) or (not object_in_grasp)
    return num_fingers_connected, obj_fell