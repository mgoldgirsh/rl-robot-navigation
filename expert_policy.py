from typing import Any
from envs.obstacles_env import ObstaclesWorld
from envs.generic_env import GenericWorld, RobotAction
import numpy as np

class ExpertPolicy():
    def __init__(self, env: GenericWorld) -> None:
        self.env = env
    
    def find_action(self) -> RobotAction:
        robot_pos = self.env.pos
        goal_pos = self.env.goal

        angle_to_go = round(-np.arctan2(goal_pos[1] - robot_pos[1], (goal_pos[0] - robot_pos[0])) * 180/np.pi)
        print(self.env.angle, angle_to_go)
        
        updated_pos = robot_pos[0] + np.cos(angle_to_go * np.pi/180), robot_pos[1] + np.sin(angle_to_go * np.pi/180)
        if (self.env.has_collision(updated_pos)):
            return RobotAction.ROTATE
        
        if (self.env.angle - 360 == angle_to_go):
            return RobotAction.FORWARD
        else:
            return RobotAction.ROTATE
        

        
        
if __name__ == "__main__":
    env = ObstaclesWorld(500, 500, see_all=True)
    expert = ExpertPolicy(env)
    
    episodes = 5000
    gamma = 0.99
    for episode_num in range(episodes):
        env.reset()
        G = 0
        done = False
        timestep = 0
        while not done:
            action = expert.find_action()
            next_state, reward, done = env.step(action, render=True)
            G += reward + G * gamma
            if (done): 
                print('discounted return', G, 'timestep', timestep)
                env.reset()

            timestep += 1