# rl-robot-navigation

**This project is based off the `DRL-robot-navigation` Github.** 

https://github.com/reiniscimurs/DRL-robot-navigation/tree/main  


This project creates a deep reinforcement learning environment with Python's `tkinter` library, and does not require any usage of ROS. 


The goal is for the robot to navigation to a random goal point in the environment with obstacle avoidance. Obstacles are detected with point cloud distance readings and the goal is randomly generated at the start of each episode of simulation.

## Modifications from `DRL-robot-navigation` Github

The environment has been simplified from the ROS environment. 

The `observation space` of the simulation is 
1. the distance from the goal
2. the angle robot is facing
3. the fov of the robot (the point cloud) (a `np.array` of length 61)s

The `action space` of the simulation has been simplified to a discrete action space with 
1. `RobotAction.FORWARD` - sets the linear velocity of the robot to 1
2. `RobotAction.ROTATE_LEFT` - sets the angular velocity of the rotation to 3
3. `RobotAction.ROTATE_RIGHT` - sets the angular velocity of the rotation to -3

The `reward` calculation of the simulation looks like the following: 
1. When the goal position is reached the reward is $R = 100$
2. When a collision occurs the reward is $R = -100$
3. On every other step if the action is the move forward action the reward is defined with the following:
    $$
    R = 1 - \frac {\texttt{distance to goal}} {\texttt{max distance possible}}
    $$

    Otherwise when the action is a roatation the reward is: 
    $$
    R = 0
    $$



## Installation 
Git clone into this repository using the following command.
```bash
git clone git@github.com:mgoldgirsh/rl-robot-navigation.git
```

### Requirements
Need you need to have Python 3.8 >= to run this program

Install PyTorch on your computer using the following website. \
https://pytorch.org/get-started/locally/

Then 
```bash
pip3 install -r requirements.txt
```

Run `dqn.py` for DQN learning, `point_and_shoot.py` for "point and shoot" algorithm, or run `imitation.py` for imitation learning. Train and see results! :)

## Algorithms used for Training

The following algorithms are used to train the robot and generate an optimal policy for obtaining the goal. 

1. DQN - Deep Q-Learning
    - This reinforcment algorithm was chosen to act as a baseline for how the robot would perform in this updated environment

2. "point and shoot" algorithm
	- This algorithm is used as a baseline to compare the results obtained by reinforcement learning methods with a simple goto goal algorithm.

3. Imitation Learning 
	- A combination between the "point and shoot" algorithm and DQN
	- the replay buffer is trained with point and shoot and then trained with DQN
    - This reinforcement algorithm was chosen to see how the robot would perform when given expert (`ExpertPolicy`) provided data to train on and see the results of such as policy.

4. TODO: add more?

## TODO
see `todo.md` 

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
