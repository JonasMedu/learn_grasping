import numpy as np

from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerType, AllegroFingerJointType
from setting_utils.param_handler import Parmap


def get_joint_state(hand):
    angles = np.array([hand.fingers[ft].angles for ft in AllegroFingerType])
    return angles.flatten()


def get_joint_velocities(hand):
    velocity_dict = hand.get_finger_joint_angles()
    return np.array([list(finger.values()) for finger in velocity_dict.values()]).ravel()


def get_joint_targets(hand):
    ll = []
    for ft in AllegroFingerType:
        for jt in AllegroFingerJointType:
            ll.append(hand.fingers[ft].joints[jt].target_position)
    return np.array(ll)


def get_joint_torques(hand):
    torques = np.array([hand.fingers[ft].torques for ft in AllegroFingerType])
    return torques.flatten()


def tactile_state(hand):
    forces = np.array([hand.fingers[ft].tactile_sensor.tactel_forces for ft in AllegroFingerType])
    return forces.flatten()


def tactile_state_dict(hand):
    res_dict = {}
    [res_dict.update({ft:hand.fingers[ft].tactile_sensor.tactel_forces}) for ft in AllegroFingerType]
    return res_dict


def set_joint_pos(hand, pos_tensor, finger_map, movement_noise):
    i_k = 0
    passive_joints = set(finger_map).symmetric_difference(FINGER_SET)
    for ft in finger_map:
        for jt in AllegroFingerJointType:
            hand.fingers[ft].joints[jt].target_position = pos_tensor[i_k]
            i_k += 1
    for pft in passive_joints:
        for jt in AllegroFingerJointType:
            hand.fingers[pft].joints[jt].target_position += np.random.normal(loc=0, scale=movement_noise)
            i_k += 1


def set_joint_pos_direct(hand, pos_tensor, finger_map):
    i_k = 0
    passive_joints = set(finger_map).symmetric_difference(FINGER_SET)
    for ft in finger_map:
        for jt in AllegroFingerJointType:
            hand.fingers[ft].joints[jt].target_position = hand.fingers[ft].joints[jt].observed_position + pos_tensor[i_k]
            i_k += 1


FINGER_SET = set([i for i in AllegroFingerType])


class CamProps(object):
    camera_distance = 0.45
    camera_yaw = -90.0
    camera_pitch = 10
    camera_target_position = [0.0, 0.0, 0.32]


THUMB_LOW = [0.26, -0.18, -0.19, -0.16]
THUMB_HIGH = [1.1, 1.16, 1.64, 1.72]
OTHER_LOW = [-0.47, -0.2, -0.17, -0.23]
OTHER_HIGH = [0.47, 1.61, 1.71, 1.62]
BASE_POSITION = np.array([0, 0, .3245])
BASE_ORIENTATION = np.array([0, 180, 0, 1])
TACTILE_OBS = np.zeros(92)


def build_action_space_absolute_angular_control(param_map: Parmap):
    joint_low = []
    joint_high = []
    if param_map.index:
        joint_low.append(OTHER_LOW)
        joint_high.append(OTHER_HIGH)
    if param_map.middle:
        joint_low.append(OTHER_LOW)
        joint_high.append(OTHER_HIGH)
    if param_map.small:
        joint_low.append(OTHER_LOW)
        joint_high.append(OTHER_HIGH)
    if param_map.thumb:
        joint_low.append(THUMB_LOW)
        joint_high.append(THUMB_HIGH)
    return np.array(joint_low).ravel(), np.array(joint_high).ravel()


def build_action_space(param_map: Parmap):
    """low, high"""
    return np.ones(len(param_map.get_fingers())*len(OTHER_LOW))*-1, np.ones(len(param_map.get_fingers())*len(OTHER_HIGH))


def build_action_space_from_type_list(finger_map: list):
    joint_low = []
    joint_high = []
    for allegro_finger_type in finger_map:
        if allegro_finger_type == AllegroFingerType.INDEX:
            joint_low.append(OTHER_LOW)
            joint_high.append(OTHER_HIGH)
        if allegro_finger_type == AllegroFingerType.MIDDLE:
            joint_low.append(OTHER_LOW)
            joint_high.append(OTHER_HIGH)
        if allegro_finger_type == AllegroFingerType.SMALL:
            joint_low.append(OTHER_LOW)
            joint_high.append(OTHER_HIGH)
        if allegro_finger_type == AllegroFingerType.THUMB:
            joint_low.append(THUMB_LOW)
            joint_high.append(THUMB_HIGH)
    return sum(joint_low, []), sum(joint_high, [])


def compute_input(joint_state, u, finger_map):
    new_pos = joint_state.copy()
    for f_type in finger_map:
        u_out = u[:4]
        u = u[4:]
        # if clause needs to stay sorted
        if f_type == AllegroFingerType.INDEX:
            new_pos[:4] += u_out
        elif f_type == AllegroFingerType.MIDDLE:
            new_pos[4:8] += u_out
        elif f_type == AllegroFingerType.SMALL:
            new_pos[8:12] += u_out
        elif f_type == AllegroFingerType.THUMB:
            new_pos[12:16] += u_out
    return new_pos