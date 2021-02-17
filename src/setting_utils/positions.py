"""
Set of functions which hold the positions of the initial states.
The
"""
from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerType, AllegroFingerJointType

"""
dummy step size is used to measure tactile response after position reset.
Warning do not set to small (e.g. 1e-10), this will break the simulation and explodes some torques.
@deprecated, it is more stable to include an artificial dummy amount of tactile pressure (see env.reset)
"""
dummy_step_size = 0.0001
# fixme: include in position resetHandEasy


goaling_position = [-0.05, -.04, .4]  # x, y, z
goaling_orientation = [0.6, .1, 0, .3]

base_position_pos3 = [-0.1, 0.015, .332]  # x, y, z
base_orientation_pos3 = [0.1, .48, 0.08, .75]


def adhere_target_to_initial(self):
    for ft in AllegroFingerType:
        for jt in AllegroFingerJointType:
            self.hand.fingers[ft].joints[jt].target_position = self.hand.fingers[ft].joints[jt].initial_position
    return self


def resetHandEasy(self):
    # assert issubclass(self.__class__, pp_env.AllegroHand)
    o_dt = self.pc.time_step
    self.pc.time_step = 1e-4

    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.TWIST].initial_position     = 0
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.5
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.MIDDLE].initial_position    = -.15
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.DISTAL].initial_position    = .5

    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.TWIST].initial_position     = 0
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.5
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.MIDDLE].initial_position    = -.15
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.DISTAL].initial_position    = .65

    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.TWIST].initial_position     = 0
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.5
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.MIDDLE].initial_position    = -.15
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.DISTAL].initial_position    = .6

    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.TWIST].initial_position     = .3
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 2
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.31
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.21

    base_position = [-0.12, 0, .333]  # x, y, z
    base_orientation = [9.00E+01, 0.00E+00, 9.00E+01, 0.266150]
    self.bar.base_link.initial_position = base_position
    self.bar.base_link.initial_orientation = base_orientation

    self.pc.reset_to_initial_state()
    self = adhere_target_to_initial(self)

    # self.pc.step_simulation()

    self.pc.time_step = o_dt

    return self


def pos1(self):
    # assert issubclass(self.__class__, pp_env.AllegroHand)

    o_dt = self.pc.time_step
    self.pc.time_step = dummy_step_size

    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.TWIST].initial_position     = .13
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.45
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.2
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.33
    
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.TWIST].initial_position     = .2
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.5
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.15
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.335
    
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.TWIST].initial_position     = .1
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.5
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.06
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.5
    
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.TWIST].initial_position     = 1.2
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 0.9
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.8
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.5
    
    base_position = [-0.12, 0.02, .34] # x, y, z
    base_orientation = [9.50E+01, 0.1, 9.00E+01, 0.266150]
    self.bar.base_link.initial_position = base_position
    self.bar.base_link.initial_orientation = base_orientation
    
    self.pc.reset_to_initial_state()
    self = adhere_target_to_initial(self)
    # self.pc.step_simulation()

    self.pc.time_step = o_dt

    return self


def pos2(self):
    # assert issubclass(self.__class__, pp_env.AllegroHand)
    o_dt = self.pc.time_step
    self.pc.time_step = dummy_step_size

    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.TWIST].initial_position     = .3
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.MIDDLE].initial_position    = 1
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.DISTAL].initial_position    = .8

    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.TWIST].initial_position     = 0
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.3
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.MIDDLE].initial_position    = .1
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.DISTAL].initial_position    = .85

    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.TWIST].initial_position     = -.05
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.PROXIMAL].initial_position  = .7
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.MIDDLE].initial_position    = 1
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.DISTAL].initial_position    = 1.1

    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.TWIST].initial_position     = -.8
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.5
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.MIDDLE].initial_position    = .01
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.DISTAL].initial_position    = 1.1

    base_position = base_position_pos3  # x, y, z
    base_orientation = [0.1, .48, 0.08, .75]  # fixed
    self.bar.base_link.initial_position = base_position
    self.bar.base_link.initial_orientation = base_orientation

    self.pc.reset_to_initial_state()
    self = adhere_target_to_initial(self)
    # self.pc.step_simulation()
    self.pc.time_step = o_dt

    return self


def pos3(self):
    # assert issubclass(self.__class__, pp_env.AllegroHand)
    o_dt = self.pc.time_step
    self.pc.time_step = dummy_step_size

    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.TWIST].initial_position     = .3
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.05
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.MIDDLE].initial_position    = 1
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.DISTAL].initial_position    = .8

    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.TWIST].initial_position     = 0
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.3
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.MIDDLE].initial_position    = .1
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.DISTAL].initial_position    = .85

    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.TWIST].initial_position     = -.13
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.PROXIMAL].initial_position  = .8
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.MIDDLE].initial_position    = .9
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.DISTAL].initial_position    = 1

    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.TWIST].initial_position     = -1
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.PROXIMAL].initial_position  =  1
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.MIDDLE].initial_position    = .01
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.DISTAL].initial_position    = 1.1

    self.bar.base_link.initial_position = base_position_pos3
    self.bar.base_link.initial_orientation = base_orientation_pos3

    self.pc.reset_to_initial_state()
    self = adhere_target_to_initial(self)
    # self.pc.step_simulation()
    self.pc.time_step = o_dt

    return self


bar_test_base_pos = [-.1095, .015, .34]  # x, y, z


def pos_test(self):
    # assert issubclass(self.__class__, pp_env.AllegroHand)
    # debug, with out of rotation links
    o_dt = self.pc.time_step
    self.pc.time_step = dummy_step_size

    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.TWIST].initial_position     = .13
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.45
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.2
    self.hand.fingers[AllegroFingerType.INDEX].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.33

    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.TWIST].initial_position     = .2
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.5
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.15
    self.hand.fingers[AllegroFingerType.MIDDLE].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.335

    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.TWIST].initial_position     = -.25
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 1.5
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.1
    self.hand.fingers[AllegroFingerType.SMALL].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.7

    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.TWIST].initial_position     = 1.163
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.PROXIMAL].initial_position  = 0.85
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.MIDDLE].initial_position    = 0.95
    self.hand.fingers[AllegroFingerType.THUMB].joints[AllegroFingerJointType.DISTAL].initial_position    = 0.5

    base_position = bar_test_base_pos  # x, y, z
    base_orientation = [0.1, .48, 0.08, .75]  # fixed
    self.bar.base_link.initial_position = base_position
    self.bar.base_link.initial_orientation = base_orientation
    self.pc.reset_to_initial_state()
    self = adhere_target_to_initial(self)
    # self.pc.step_simulation()

    self.pc.time_step = o_dt
    return self


position_list = [resetHandEasy, pos1, pos2, pos3]
all_pos_list = [resetHandEasy, pos1, pos2, pos3, pos_test]

