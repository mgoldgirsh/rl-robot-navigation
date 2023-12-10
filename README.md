# rl-robot-navigation

**This project is based off the `DRL-robot-navigation` Github.** 

https://github.com/reiniscimurs/DRL-robot-navigation/tree/main  


This project creates a deep reinforcement learning environment with Python's `tkinter` library, and does not require any usage of ROS. 


The goal is for the robot to navigation to a random goal point in the environment with obstacle avoidance. Obstacles are detected with point cloud distance readings and the goal is randomly generated at the start of each episode of simulation.

## Modifications from `DRL-robot-navigation` Github

The environment has been simplified from the ROS environment. 

The `observation space` of the simulation is 
1. the robots position x
2. the robots position y
3. the angle robot is facing
4. the fov of the robot (the point cloud) (a `np.array` of length 180)

The `action space` of the simulation has been simplified to a discrete action space with 
1. `RobotAction.FORWARD` - sets the linear velocity of the robot to 1
2. `RobotAction.ROTATE_LEFT` - sets the angular velocity of the rotation to 1
3. `RobotAction.ROTATE_RIGHT` - sets the angular velocity of the rotation to -1


## Algorithms used for Training

The following algorithms are used to train the robot and generate an optimal policy for obtaining the goal. 

1. DQN - Deep Q-Learning
    - This reinforcment algorithm was chosen to act as a baseline for how the robot would perform in this updated environment

2. Imitation Learning 
    - This reinforcement algorithm was chosen to see how the robot would perform when given expert (`ExpertPolicy`) provided data to train on and see the results of such as policy.

3. TODO: add more?


## Citations
    @ARTICLE{9645287,
      author={Cimurs, Reinis and Suh, Il Hong and Lee, Jin Han},
      journal={IEEE Robotics and Automation Letters}, 
      title={Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning}, 
      year={2022},
      volume={7},
      number={2},
      pages={730-737},
      doi={10.1109/LRA.2021.3133591}}
