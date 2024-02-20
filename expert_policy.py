from typing import Any
from envs.obstacles_env import ObstaclesWorld
from envs.generic_env import GenericWorld, RobotAction
import numpy as np


class ExpertPolicy:
    def __init__(self, env: GenericWorld) -> None:
        self.env = env
        self.has_collision_count = 0

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.find_action()

    def find_action(self) -> RobotAction:
        robot_pos = self.env.pos
        goal_pos = self.env.goal

        angle_to_go = round(
            -np.arctan2(goal_pos[1] - robot_pos[1], (goal_pos[0] - robot_pos[0]))
            * 180
            / np.pi
        )
        # print(self.env.angle, angle_to_go)

        cur_angle = 0
        if self.env.angle >= 360:
            cur_angle = self.env.angle - 360
        elif self.env.angle <= -360:
            cur_angle = self.env.angle + 360
        else:
            cur_angle = self.env.angle

        updated_pos = (
            robot_pos[0] + 4 * np.cos(angle_to_go * np.pi / 180),
            robot_pos[1] + 4 * np.sin(angle_to_go * np.pi / 180),
        )

        # if (env.has_collision(updated_pos)):
        #     print('has collision please')
        #     self.has_collision_count += 5
        #     return RobotAction.ROTATE_LEFT

        # if (self.has_collision_count > 0):
        #     self.has_collision_count -= 1
        #     return RobotAction.FORWARD

        # test the angle to go
        while angle_to_go % 3 != 0:
            angle_to_go += 1

        if cur_angle < angle_to_go:
            return RobotAction.ROTATE_LEFT
        elif cur_angle > angle_to_go:
            return RobotAction.ROTATE_RIGHT
        else:
            return RobotAction.FORWARD


if __name__ == "__main__":
    env = ObstaclesWorld(500, 500, see_all=True)
    expert = ExpertPolicy(env)

    episodes = 5000
    gamma = 0.99
    for episode_num in range(episodes):
        env.reset(render=True)
        G = 0
        done = False
        timestep = 0
        while not done:
            action = expert()
            next_state, reward, done = env.step(action, render=True)
            G = reward + gamma * G
            if done:
                print("discounted return", G, "timestep", timestep)
                env.reset(render=True)

            timestep += 1
