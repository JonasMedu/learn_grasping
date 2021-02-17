import time
from PIL import Image

import pybullet

from hand_env.allegro_env_unsupervised import AllegroCollisionFromTactileHand

from hand_env.allegro_env import AllegroHand
from hand_env.hand_interface import tactile_state_dict, get_joint_state
from setting_utils.param_handler import Parmap
from setting_utils.positions import all_pos_list, adhere_target_to_initial
import numpy as np


def save_position_as_images(res=1600):
    env = AllegroHand(Parmap(), gui=True, show_fingers=True)
    for pos in all_pos_list:
        # env.pc.reset_debug_visualizer_camera(CamProps.camera_distance,
        #                                      CamProps.camera_yaw,
        #                                      CamProps.camera_pitch,
        #                                      CamProps.camera_target_position)
        env = pos(env)
        env = adhere_target_to_initial(env)
        env.pc.time_step = 1e-3
        env.pc.step_simulation()
        camera_image = env.pc.call(pybullet.getCameraImage, res, res)
        rbg_image = camera_image[2]
        img = Image.fromarray(rbg_image[:, :, 0:3])  # discard alpha channel
        img.save(''+pos.__name__+'.pdf')
        print("break")
        time.sleep(1)
    env.close()

if __name__ == '__main__':
    pars = Parmap(time_step=1e-2)
    pars.rwd_func = lambda x: (0, False)
    env = AllegroCollisionFromTactileHand(pars, gui=True, show_fingers=True)

    all_pos = all_pos_list.copy()
    for pos in all_pos:
        env = pos(env)
        zero_act = get_joint_state(env.hand)
        env.pc.step_simulation()
        for i in range(3):
            # put debugger here, and use console to prototype new hand-object configurations
            env.show_collisions()
            tac_state_dict = tactile_state_dict(env.hand)
            print(env.reward())
            for finger in tac_state_dict:
                print(finger, np.sum(tac_state_dict[finger]))
            env.step(np.random.normal(zero_act, scale=.001))

        time.sleep(1)
    #### save_position_as_images()
    exit(0)