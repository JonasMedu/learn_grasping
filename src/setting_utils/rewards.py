import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from pyquaternion import Quaternion
from setting_utils.reward_utils import angular_similarity, DetailsForReward, all_action_fingers_have_contact, \
    for_weighted_connect

""" begin here, reward functions learned."""


def close_to_init(rwdobj: DetailsForReward):
    "stay close to current joint position"
    ir = angular_similarity(rwdobj.current_joint_position, rwdobj.init_joint_pos)
    pos_init_dis = euclidean(rwdobj.new_obj_pos, rwdobj.init_obj_pos)
    grip_strength = np.round(np.sum(np.abs(rwdobj.tactile_state[-1])), 2)
    obj_fell = (pos_init_dis > 0.1) or (grip_strength < (0.004 * rwdobj.action_space))
    if obj_fell:
        return rwdobj.extrinsic_reward, obj_fell
    else:
        return ir, obj_fell


def close_to_init_no_debug(rwdobj: DetailsForReward):
    "stay close to current joint position, policy may cheats, because the object fell definition only relies on tactile information."
    ir = angular_similarity(rwdobj.current_joint_position, rwdobj.init_joint_pos)
    grip_strength = np.round(np.sum(np.abs(rwdobj.tactile_state[-1])), 2)
    obj_fell = (grip_strength < (0.004 * rwdobj.action_space))
    if obj_fell:
        return rwdobj.extrinsic_reward, obj_fell
    else:
        return ir, obj_fell


def weighted_close_to_init(rwdobj: DetailsForReward):
    ir = cosine(rwdobj.current_joint_position, rwdobj.init_joint_pos) + 1
    num_fingers_connected, obj_fell = for_weighted_connect(rwdobj)
    if obj_fell:
        return rwdobj.extrinsic_reward, obj_fell
    else:
        weighted_distance = num_fingers_connected/np.exp(ir)
        return weighted_distance, obj_fell


def weighted_or_dist(rwdobj: DetailsForReward):
    """Take orientation as distance metric. Reason: position can easily  be manipulated by the hands position."""
    distance_or = Quaternion.distance(Quaternion(rwdobj.new_obj_or), Quaternion(rwdobj.init_obj_or))
    num_fingers_connected, obj_fell = for_weighted_connect(rwdobj)
    if obj_fell:
        return rwdobj.extrinsic_reward, obj_fell
    else:
        weighted_distance = num_fingers_connected/np.exp(distance_or)
        return weighted_distance, obj_fell


def or_dist(rwdobj: DetailsForReward):
    """Take orientation as distance metric. Reason: position can easily  be manipulated by the hands position."""
    distance_or = Quaternion.distance(Quaternion(rwdobj.new_obj_or), Quaternion(rwdobj.init_obj_or))
    obj_fell = not all_action_fingers_have_contact(rwdobj)
    if obj_fell:
        return rwdobj.extrinsic_reward, obj_fell
    else:
        distance = 1/np.exp(distance_or)
        return distance, obj_fell


def move_connection_to_object(rwdobj: DetailsForReward):
    """move object to a position"""
    distance_pos = euclidean(rwdobj.new_obj_pos, rwdobj.goal_pos_orientation.obj_pos)
    # norm positional distance
    distance_pos = distance_pos / euclidean(rwdobj.init_obj_pos, rwdobj.goal_pos_orientation.obj_pos)   # (x-min) / ( max-min)-> normalize form 0..1, min=0

    distance_or = Quaternion.distance(Quaternion(rwdobj.new_obj_or), Quaternion(rwdobj.goal_pos_orientation.obj_or))
    # norm orientational distance
    distance_or = distance_or / Quaternion.distance(Quaternion(rwdobj.init_obj_or), Quaternion(rwdobj.goal_pos_orientation.obj_or))

    num_fingers_connected, obj_fell = for_weighted_connect(rwdobj)
    reached = (distance_pos < 0.01) and (distance_or < 0.01)
    if reached:
        print("reached")
    if obj_fell:
        return -1, True  # reward finish
    else:
        assert distance_pos >= 0, "positional distance is negative."
        assert distance_or >= 0, "Quaternion distance is negative."
        weighted_distance = num_fingers_connected/np.exp(distance_pos + distance_or)
        return weighted_distance, reached


"""begin here, reward functions, not learned for paper visuals"""


def close_to_init_collision(rwdobj: DetailsForReward):
    ir = angular_similarity(rwdobj.current_joint_position, rwdobj.init_joint_pos)
    pos_init_dis = euclidean(rwdobj.new_obj_pos, rwdobj.init_obj_pos)
    obj_fell = sum(rwdobj.collision_dict.values()) < len(rwdobj.action_fingers)
    if obj_fell:
        return rwdobj.extrinsic_reward, obj_fell
    else:
        return ir, obj_fell


def close_to_init_collision_2(rwdobj: DetailsForReward):
    ir = np.abs(cosine(rwdobj.current_joint_position, rwdobj.init_joint_pos)) *-1
    pos_init_dis = euclidean(rwdobj.new_obj_pos, rwdobj.init_obj_pos)
    obj_fell = sum(rwdobj.collision_dict.values()) < len(rwdobj.action_fingers)
    if obj_fell:
        return rwdobj.extrinsic_reward, obj_fell
    else:
        return ir, obj_fell


def connected_to_object(rwdobj: DetailsForReward):
    """ high reward, if finger tips connect to bar. """
    pos_init_dis = euclidean(rwdobj.new_obj_pos, rwdobj.init_obj_pos)
    grip_strength = np.round(np.sum(np.abs(rwdobj.tactile_state[-1])), 2)
    obj_fell = (pos_init_dis > 0.1) or (grip_strength < (0.004 * rwdobj.action_space))
    stuck = False
    if len(rwdobj.tactile_state) > 2:
        sames = np.round(rwdobj.current_joint_position, 2) == np.round(rwdobj.current_minus1_joint_position, 2)
        stuck = all(sames)
    if obj_fell or stuck:
        return rwdobj.extrinsic_reward, (obj_fell or stuck)
    else:
        return sum(rwdobj.collision_dict.values()), obj_fell


def move_connection(rwdobj: DetailsForReward):
    """ high reward, if finger tips connect to bar. """
    dis_traveled = euclidean(rwdobj.new_obj_pos, rwdobj.init_obj_pos)
    or_traveled = Quaternion.sym_distance(Quaternion(rwdobj.new_obj_or), Quaternion(rwdobj.init_obj_or))
    # symmetric orientation scaling
    or_traveled /= 1000
    obj_fell = sum(rwdobj.collision_dict.values()) < 4
    stuck = False
    if len(rwdobj.tactile_state) > 2:
        sames = np.round(rwdobj.current_joint_position, 2) == np.round(rwdobj.current_minus1_joint_position, 2)
        stuck = all(sames)
    if obj_fell or stuck:
        return rwdobj.extrinsic_reward, (obj_fell or stuck)
    else:
        return np.exp(dis_traveled + or_traveled), obj_fell


def move_connection_to_object_only_position(rwdobj: DetailsForReward):
    """move object to a position"""
    distance_pos = euclidean(rwdobj.new_obj_pos, rwdobj.goal_pos_orientation.obj_pos)

    distance_pos = distance_pos / euclidean(rwdobj.init_obj_pos,
                                            rwdobj.goal_pos_orientation.obj_pos)  # (x-min) / ( max-min)-> normalize form 0..1, min=0
    # number of fingers connected from tactile feedback

    # limited amount of steps per trajectory

    max_traj_length_reached = rwdobj.counter > 399

    tac_threshold = .005
    num_fingers_connected = np.sum([np.sum(rwdobj.tactile_state_dict[finger]) > tac_threshold for finger in rwdobj.action_fingers])
    object_in_grasp = num_fingers_connected > 1  # at least two fingers
    obj_fell = (sum(rwdobj.collisions_not_bar_dict.values()) > 1) or (not object_in_grasp)
    reached = (distance_pos < 0.01)
    if reached:
        print("reached")
    if obj_fell:
        return -1, True  # reward finish
    else:
        assert distance_pos >= 0, "positional distance is negative."
        weighted_distance = num_fingers_connected/np.exp(distance_pos)
        return weighted_distance, (reached or max_traj_length_reached)


def test(rwdobj: DetailsForReward):
    """MOVE AS MUCH AS POSSIBLE"""
    dis_traveled = euclidean(rwdobj.new_obj_pos, rwdobj.init_obj_pos)
    or_traveled = Quaternion.absolute_distance(Quaternion(rwdobj.new_obj_or), Quaternion(rwdobj.init_obj_or))
    grip_strength = np.round(np.sum(np.abs(rwdobj.tactile_state[-1])), 2)
    obj_fell = grip_strength < 0.004 * rwdobj.action_space
    if obj_fell:
        return rwdobj.extrinsic_reward, True # reward finish
    else:
        return dis_traveled + or_traveled, False


def move_object(rwdobj: DetailsForReward):
    """move object to a position"""
    distance_pos = euclidean(rwdobj.new_obj_pos, rwdobj.goal_pos_orientation.obj_pos)
    # norm positional distance
    distance_pos = distance_pos / euclidean(rwdobj.init_obj_pos, rwdobj.goal_pos_orientation.obj_pos)   # (x-min) / ( max-min)-> normalize form 0..1, min=0

    distance_or = Quaternion.distance(Quaternion(rwdobj.new_obj_or), Quaternion(rwdobj.goal_pos_orientation.obj_or))
    # norm orientational distance
    distance_or = distance_or / Quaternion.distance(Quaternion(rwdobj.init_obj_or), Quaternion(rwdobj.goal_pos_orientation.obj_or))

    grip_strength = np.round(np.sum(np.abs(rwdobj.tactile_state[-1])), 2)
    obj_fell = grip_strength < 0.01 * rwdobj.action_space
    reached = (distance_pos < 0.1) and (distance_or < 0.1)
    if reached:
        print("reached")
    # print(distance_or, distance_pos, reached)
    if obj_fell:
        return rwdobj.extrinsic_reward, True  # reward finish
    else:
        assert distance_pos >= 0, "positional distance is negative."
        assert distance_or >= 0, "Quaternion distance is negative."
        return 1/np.exp(distance_pos + distance_or), reached
