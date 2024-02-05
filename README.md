## In-hand manipulation with proximal policy optimization (PPO)
This project aims to train the Allegro hand for in-hand manipulation (in simulation). The robotic Allegro hand is an analogy of human anatomy. Its fingertips measure haptic pressure via tactile information. The real hand makes use of fluids, which are hard to model. This motivates model free reinforcement learning. In this project the Allegro hand learns to find a stabilizing grasp for a small bar.
The project defines two experimental setup. The first one aims to find a reward function definition to enable training progression. The second setup simulates a trajectory optimization in form of noise. 
#### framework overview
**tensorflow 1.x** for machine learning
**pybullet** for the Allegro hand simulation
**gym(.Env)** as superclass for hand  

#### packages overview
ausy_base:
  1. Learning script
  1. PPO implementation
  1. Policy and value function model

allegro_pybullet:
  1. Meshes for the hand and the grasping object
  1. Pybullet interface, manipulation interface for the hand

hand_env:
  1. Gym classes for the Allegro hand. (defines state and action spaces) 
      1. Grasping learning hand (**allegro_env**)
      1. Grasping learning hand under the influence of noise (**noisy_hand**)
      1. Grasping learning hand which stacks a previously trained (**trained_env**)

setting_utils:
  1. controls the experiments parameters
  1. holds the positions of the initial state distribution
  1. defines the reward functions

performance_analysis:
  1. holds code to make graphs for the training progression etc.

### quick start (curerntly unavailable; I do not have any licence information about the allegro Hand implementation)
  1. install requirements (e.g. via pip), the requirements file
  2. adapt hard coded logging folder in setting_utils.paramhandler
  3. run learn_lower with gui=True
  4. wait for learning progression 
     1. You can measure the learning progression by the "number of trajectories" per training data gathering cycle. You can inspect this variable via tensorboard.


### implementation notes
- **setting_utils** defines tensorboard
- project ran only on local machines -> no cli arguments, alter source code directly
- tf v 1. requires messy model saving/loading handling
- files saved (standard out: documents/tb, new folder per experiment) per run:
  1. progress.csv with the training progress
  1. config file saves the parameter settings
- pybullet works in one thread. The easiest possibility to speed up the experiments, is to run them in parallel.    
- the show_finger option of the Envs shows the tactile pressure, but slow down the simulation hard
- because of pybullet, you only render every time step, or none.
- lots of tf warnings are thrown because of the project age
- the policy loading and saving is mostly managed via naming conventions
#### model notes
2-hidden layer (size 64) MLP, for the value function and the policy.
#### How to enable Logging in tensorboard
run in your comand line
*tensorboard --logdir=path/to/tensorboard/data* \
Where as the fixed standard out: "path/to/tensorboard/data" = "~/documents/tb"


#### Results
After around 600 training and data gathering iterations the policy learns a stabilizing grasp.
The following images show a policy trained with the reward function **weighted_or_dist**.
<p align="center">
  <img src="https://github.com/JonasMedu/learn_grasping/blob/main/read_me_images/noise_weighted_or_distMon_Dec__7_13_01_05_2020_28fea92c-be03-41fd-b6f4-95713e8b46c7_pos1_50_cropped.png" width="200" title="stable position a">
  <img src="https://github.com/JonasMedu/learn_grasping/blob/main/read_me_images/noise_weighted_or_distMon_Dec__7_13_01_05_2020_28fea92c-be03-41fd-b6f4-95713e8b46c7_pos2_50_cropped.png" width="200" alt="stable position b">
  <img src="https://github.com/JonasMedu/learn_grasping/blob/main/read_me_images/noise_weighted_or_distMon_Dec__7_13_01_05_2020_28fea92c-be03-41fd-b6f4-95713e8b46c7_pos3_50_cropped.png" width="200" alt="stable position c">
  <img src="https://github.com/JonasMedu/learn_grasping/blob/main/read_me_images/noise_weighted_or_distMon_Dec__7_13_01_05_2020_28fea92c-be03-41fd-b6f4-95713e8b46c7_pos_test_50_cropped.png" width="200" alt="stable position d">  
</p>

The second experiment shows how a simulation of the *upper* trajectory optimization influences the learning capabilities of the stabilizing grasp. The graph shows the training progression with the influence of noise. The graph shows 20 training sessions, each of which as a different noise (simulated *upper* trajectory optimization signal) input. The noise is spaced between .1 and 20 in a geometric progression. The color of the lines help to indicate the noise magnitude. The higher the noise, the darker the line. A gradient from bright, at the button, to dark (at the top) appears in the graph.   
<p align="center">
  <img src="https://github.com/JonasMedu/learn_grasping/blob/main/read_me_images/Training_progress_noisy_run.png" width="600" title="Grapsing performance under the influence of noise.">
</p>

Each iteration has a fixed number of simulation steps. The lower the number of trajectories, the more stable is the grasp of the bar.
