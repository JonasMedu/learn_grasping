import gym
import configparser
from gym import spaces
import numpy as np
import pybullet
import pybullet_data
from allegro_pybullet import PhysicsClient
from allegro_pybullet.bar_description.BarObject import Bar, BarUncollide
from allegro_pybullet.simulation_body import URDFBody
from allegro_pybullet.simulation_body.allegro_hand import AllegroRightHand
from allegro_pybullet.simulation_body.allegro_hand.allegro_right_hand import AllegroRightHandUncollide
from allegro_pybullet.simulation_object import JointControlMode
from setting_utils.reward_utils import DetailsForReward
from setting_utils.param_handler import Parmap
from setting_utils.positions import position_list
from hand_env.hand_interface import get_joint_state, tactile_state, tactile_state_dict, set_joint_pos_direct, CamProps, \
    THUMB_LOW, THUMB_HIGH, OTHER_LOW, OTHER_HIGH, BASE_POSITION, BASE_ORIENTATION, TACTILE_OBS, build_action_space

config = configparser.ConfigParser()
config.read('config.ini')


class AllegroHand(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, allegros_params: Parmap, gui=False, show_fingers=False):

        # setting experiment parameters
        self.paramMap = allegros_params
        self.paramMap.gym_env = self.__class__.__name__
        # init finger map, dependent on params / options
        self.fingerMap = self.paramMap.get_fingers()

        # init physics simulation
        self.state = None
        self.pc = PhysicsClient()
        if gui:
            self.connect_gui()
        else:
            self.pc.connect_direct()
            self.gui_connected = False

        self.add_simulation_bodies()

        # simulation parameters
        pybullet.setGravity(0, 0, -9.8)
        self.pc.time_step = self.paramMap.time_step

        # action: joint_cmd(pos)
        joint_low, joint_high = build_action_space(param_map=self.paramMap)
        self.action_space = spaces.Box(joint_low, joint_high, dtype=np.float64)
        # state: current get_joint_state(pos,vel), object pos&orientation and (biotacs)
        self.define_observation_space()
        # init logging
        self.dets_reward = DetailsForReward(self.action_space.shape[0], TACTILE_OBS.size)
        self.dets_reward.extrinsic_reward = self.paramMap.extrinsic_reward
        self.dets_reward.min_trans_per_iter = self.paramMap.min_trans_per_iter
        self.dets_reward.goal_reached_reward = self.paramMap.goal_reached_reward
        self.dets_reward.action_fingers = self.fingerMap.copy()

        self.joint_state = np.zeros(self.action_space.shape[0])

        # if gui:
        #     self.pc = AllegroHandVisual.visualize_bar_goal(self.pc, self.dets_reward)

        if show_fingers:
            assert gui, "You can not display the finger tactile pressures without connecting to the gui"
            self.prep_tactile_display()

    def add_simulation_bodies(self):
        """connect items to physics simulation"""
        self.hand = AllegroRightHand(
            base_position=BASE_POSITION,
            base_orientation=np.array(BASE_ORIENTATION),
            joint_control_mode=JointControlMode.POSITION_CONTROL)
        self.bar = Bar(
            base_position=np.array([0, 0, 1]),
            base_orientation=np.array([1, 1, 0, 0]))
        # id by order
        self.pc.add_body(self.hand)
        self.pc.add_body(self.bar)
        # self.initialize_state(random=False)
        self.pc.set_additional_search_path(pybullet_data.getDataPath())
        self.pc.add_body(URDFBody("plane.urdf"))
        # id numeration
        self.hand_link_id  = 0
        self.bar_link_id   = 1
        self.plane_link_id = 2

    def connect_gui(self):
        """
        starts the visualization of the simulation
        :return: None
        """
        self.pc.connect_gui()
        self.pc.reset_debug_visualizer_camera(CamProps.camera_distance,
                                              CamProps.camera_yaw,
                                              CamProps.camera_pitch,
                                              CamProps.camera_target_position)
        self.pc.configure_debug_visualizer(pybullet.COV_ENABLE_MOUSE_PICKING, False)
        self.gui_connected = True

    def prep_tactile_display(self):
        """
        complicated method to display the tactile pressures
        :return: None
        """
        new_finger_dict = self.hand._AllegroHand__fingers.copy()
        for finger_name in self.hand._AllegroHand__fingers:
            finger_obj = self.hand._AllegroHand__fingers[finger_name]
            finger_obj._AllegroFinger__tactile_sensor.display_tactels = True
            new_finger_dict.update({finger_name: finger_obj})
        self.hand._AllegroHand__fingers = new_finger_dict

    def define_observation_space(self):
        """
        defines state space / observation space box
        Lower observation has always all fingers as input disregarding the amount of fingers used.
        :return: None
        """
        tactile_low = TACTILE_OBS
        tactile_high = np.ones(TACTILE_OBS.shape) * 100
        self.observation_space = spaces.Box(
            np.concatenate((np.array(OTHER_LOW * 3 + THUMB_LOW).ravel(), tactile_low)),
            # low
            np.concatenate((np.array(OTHER_HIGH * 3 + THUMB_HIGH).ravel(), tactile_high)),
            # high
            dtype=np.float64)

    def reward(self):

        self.dets_reward.tactile_state_dict = tactile_state_dict(self.hand)

        self.dets_reward.current_joint_position = get_joint_state(self.hand)
        self.dets_reward.new_obj_pos, self.dets_reward.new_obj_or = self.pc.call(pybullet.getBasePositionAndOrientation,
                                                                                 self.bar.body_unique_id)

        return self.paramMap.rwd_func(self.dets_reward)

    def step(self, u):
        # u does not have to be clipped
        # self.state is current position
        # joint_input = compute_input_as_map(self.hand.get_finger_joint_angles(), u, self.paramMap.get_fingers())

        # 1 is bar -> need actually ids from fingertips?
        self.dets_reward.old_obj_pos, self.dets_reward.old_obj_or = self.pc.call(pybullet.getBasePositionAndOrientation,
                                                                                 self.bar.body_unique_id)

        set_joint_pos_direct(self.hand, u, self.fingerMap)

        self.pc.step_simulation()

        self.joint_state = get_joint_state(self.hand)
        ts = tactile_state(self.hand)
        full_state = self.state_for_policy_input(ts)
        self.dets_reward.policy_action.append(u)
        self.dets_reward.tactile_state.append(ts)
        reward, done = self.reward()

        # time.sleep(0.1)
        return full_state, reward, done, 0

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

        ts = tactile_state(self.hand)+1 # debug fix.
        full_state = self.state_for_policy_input(ts)
        return full_state

    def state_for_policy_input(self, ts):
        full_state = np.concatenate([self.joint_state, ts])
        return full_state

    def close(self):
        self.pc.disconnect()


class AllegroHandVisual():
    def __init__(self, pc: PhysicsClient, dets_reward: DetailsForReward):
        self.hand = AllegroRightHandUncollide(
            base_position=BASE_POSITION,
            base_orientation=BASE_ORIENTATION)

        self.bar = BarUncollide(
            base_position=dets_reward.goal_pos_orientation.obj_pos,
            base_orientation=dets_reward.goal_pos_orientation.obj_or)

        pc.add_body(self.hand)
        pc.add_body(self.bar)

        self.pc = pc

    @staticmethod
    def visualize_bar_goal(pc, dets_reward: DetailsForReward):
        bar = BarUncollide(
            base_position=dets_reward.goal_pos_orientation.obj_pos,
            base_orientation=dets_reward.goal_pos_orientation.obj_or)

        pc.add_body(bar)
        return pc