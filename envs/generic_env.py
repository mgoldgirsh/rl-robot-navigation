import tkinter as tk
from tkinter import *
import numpy as np
from typing import List, Tuple
from enum import IntEnum

from envs.obstacle import Obstacle
from utils import convert_to_radians


class RobotAction(IntEnum):
    FORWARD = 0  # vel = 1, rotation = 0
    ROTATE_LEFT = 1  # vel = 0, rotation = 3s
    ROTATE_RIGHT = 2  # vel = 0, rotation = -3


class GenericWorld:
    def __init__(
            self,
            width: int,
            height: int,
            refresh_rate: int = 10,
            manual: bool = False,
            see_all: bool = False,
    ) -> None:
        self.width = width
        self.height = height

        # ---- robot parameters
        self.pos = (width // 2, height // 2)  # the position of the robot
        self.angle = 0  # the angle of the robot in degrees (0 represents the right/EAST direction)

        self.vel = 0  # the linear velocity
        self.rotational_vel = 0  # the angular velocity
        self.fov = 180  # the field of view of a robot (from its orientiation angle) (current 180 fov)

        # obstacles
        self._obstacles = []

        # the goal to goto
        self.goal = self._generate_random_position()
        self.goalbox = 20  # the full box side length around the goal

        # the distances objects are from the location of the robot
        # this is called the point cloud
        # the first index is the minimum most location of the point cloud
        self.point_cloud = [1] * (self.fov + 1)

        # whether to accept manual inputs
        self.manual = manual
        # whether to see all obstacles and not just point cloud
        self.see_all = see_all

        # refresh rate
        self.refresh_rate = refresh_rate
        self.tick = 0

        # rendering elmeents
        self.top = Tk()
        self.top.geometry("600x600")
        self.canvas = Canvas(self.top, height=height, width=width)

        # on key press
        if manual:
            # background tick update
            self.top.after(self.refresh_rate, self._update)
            self.top.bind("<KeyPress>", self._onKeyPress)

    def _draw_cursor(self, size=30) -> None:
        coord = (
            self.pos[0] - size // 2,
            self.pos[1] - size // 2,
            self.pos[0] + size // 2,
            self.pos[1] + size // 2,
        )
        self.canvas.create_arc(
            coord, start=self.angle + 157.5, extent=45, fill="red", tags=("robot")
        )
        self.canvas.create_text(self.width-85, 20, text=f'Position: {round(self.pos[0], 2), round(self.pos[1],2)}')
        self.canvas.create_text(self.width-44, 35, text=f'Angle: {round(self.angle, 2)}°')

    def _draw_obstacles(self) -> None:
        for obstacle in self._obstacles:
            obstacle.create_obstacle(self.canvas)

    def _draw_goal(self) -> None:
        box_side = self.goalbox / 2
        goal_rect = (
            self.goal[0] - box_side,
            self.goal[1] - box_side,
            self.goal[0] + box_side,
            self.goal[1] + box_side,
        )
        self.canvas.create_rectangle(goal_rect, fill="green", tags=("goal"))

    def _draw_fov(self, size=2) -> None:
        i = 0

        for fov_angle in range(
                self.angle - self.fov // 2, self.angle + self.fov // 2 + 1
        ):
            updated_pos = (
                self.pos[0]
                + np.cos(self._normalize_angle(fov_angle) * np.pi / 180)
                * self.point_cloud[i],
                self.pos[1]
                - np.sin(self._normalize_angle(fov_angle) * np.pi / 180)
                * self.point_cloud[i])
            self.canvas.create_oval(
                min(max(updated_pos[0], 5), self.width - 1) - size / 2,
                min(max(updated_pos[1], 5), self.height - 1) - size / 2,
                min(max(updated_pos[0], 5), self.width - 1) + size / 2,
                min(max(updated_pos[1], 5), self.height - 1) + size / 2,
                fill="black",
                tags="fov",
            )
            i += 1

    def _update_position(self) -> None:
        # update the angle first
        self.angle += self.rotational_vel

        # standardize the fov_angle
        # the range of the fov_angle should -180 to 180
        self.angle = self._normalize_angle(self.angle)

        updated_vel = (
            self.vel * np.cos(convert_to_radians(self.angle)),
            self.vel * np.sin(convert_to_radians(-self.angle)),
        )
        updated_pos = (self.pos[0] + updated_vel[0], self.pos[1] + updated_vel[1])

        if self.manual:
            if not self.has_collision(updated_pos):
                self.pos = updated_pos
        else:
            self.pos = updated_pos

    def _distance_to_wall(self, fov_angle: int) -> Tuple[int, int]:
        # given the fov_angle in degrees calculate the distance to the nearest wall
        # this calculated using the quadratic formula
        if 0 <= fov_angle < 90:
            # you have to be looking at top/right walls
            vert_dist = self.pos[1]
            horiz_dist = self.width - self.pos[0]
        elif 90 <= fov_angle < 180:
            # looking at top/left walls
            vert_dist = self.pos[1]
            horiz_dist = self.pos[0]

        elif -90 <= fov_angle < 0:
            # looking at bottom/right walls
            vert_dist = self.height - self.pos[1]
            horiz_dist = self.width - self.pos[0]
        else:
            # looking at bottom/left walls
            vert_dist = self.height - self.pos[1]
            horiz_dist = self.pos[0]

        if fov_angle == 0:
            return horiz_dist
        if fov_angle == 90 or fov_angle == -90:
            return vert_dist

        # calculate vert/horiz distance from the top and left points of intersection
        calc_vert = np.tan(fov_angle * np.pi / 180) * horiz_dist
        calc_horiz = 1/np.tan(fov_angle * np.pi / 180) * vert_dist

        if abs(calc_vert) < abs(calc_horiz):
            return (calc_vert ** 2 + horiz_dist ** 2) ** .5
        else:
            return (calc_horiz ** 2 + vert_dist ** 2) ** .5

    def _update_point_cloud(self) -> np.array:
        # the goal of the point cloud is to generate list of distances of how far way something is from the
        # robot based on the angle

        fov_angles = range(self.angle - self.fov // 2, self.angle + self.fov // 2 + 1)
        i = 0
        for fov_angle in fov_angles:
            self.point_cloud[i] = min(
                [obs.distance_to_robot(self.pos, self._normalize_angle(fov_angle)) for obs in self._obstacles] +
                [self._distance_to_wall(self._normalize_angle(fov_angle))]
            )
            i += 1

        return self.point_cloud

    def _update_data(self) -> None:
        # update position and point cloud
        self._update_position()
        self._update_point_cloud()

    def _update(self) -> None:
        # reset canvas
        self.canvas.delete("all")
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.canvas.create_rectangle(4, 4, self.width, self.height, outline='black', width=1)

        # update all data
        self._update_data()

        # re-render pos and obstacles
        self._draw_cursor()
        self._draw_goal()
        if self.see_all:
            self._draw_obstacles()
        self._draw_fov()

        self.canvas.update()
        self.canvas.pack()
        if self.manual:
            self.top.after(self.refresh_rate, self._update)

    def _onKeyPress(self, event) -> None:
        if event.keysym == "Left":
            self.vel = 0
            self.rotational_vel = 1
        elif event.keysym == "Right":
            self.vel = 0
            self.rotational_vel = -1
        elif event.keysym == "Up":
            self.rotational_vel = 0
            self.vel = 1
        elif event.keysym == "Down":
            self.rotational_vel = 0
            self.vel = -1
        elif event.keysym == "0":
            self.vel = 0
            self.rotational_vel = 0

    def render(self) -> None:
        self.top.mainloop()

    def add_obstacles(self, obstacles: List[Obstacle]) -> None:
        for obstacle in obstacles:
            self._obstacles.append(obstacle)

    def clear_obstacles(self):
        self._obstacles.clear()

    def has_collision(self, position) -> bool:
        collides = any(
            [obstacle.collides_with(position) for obstacle in self._obstacles]
        )

        on_border = False
        if (
                position[0] <= 0
                or position[0] >= self.width
                or position[1] <= 0
                or position[1] >= self.height
        ):
            on_border = True

        return collides or on_border

    def _generate_random_position(self) -> Tuple[int, int]:
        random_pos = (np.random.randint(self.width), np.random.randint(self.height))

        collision = False
        if len(self._obstacles) != 0:
            collision = self.has_collision(random_pos)

        while collision:
            random_pos = (np.random.randint(self.width), np.random.randint(self.height))
            if len(self._obstacles) != 0:
                collision = self.has_collision(random_pos)

        return random_pos

    def _within_goal(self) -> bool:
        box_side = self.goalbox / 2
        if (
                self.goal[0] - box_side <= self.pos[0] <= self.goal[0] + box_side
        ) and (
                self.goal[1] - box_side <= self.pos[1] <= self.goal[1] + box_side
        ):
            return True
        else:
            return False

    @staticmethod
    def _normalize_angle(angle: int) -> int:
        if angle < -180:
            angle += 360
        elif angle > 180:
            angle -= 360
        return angle

    def _distance_to_goal(self) -> float:
        return ((self.pos[0] - self.goal[0]) ** 2 + (self.pos[1] - self.goal[1]) ** 2) ** 0.5

    def _generate_new_world(self):
        pass

    def reset(self, render: bool = False) -> np.array:
        self._generate_new_world()
        self.pos = self._generate_random_position()
        self.goal = self._generate_random_position()
        self.angle = 0
        self.vel = 0
        self.rotational_vel = 0
        if render or self.manual:
            self._update()
        else:
            self._update_data()

        distance_to_goal = self._distance_to_goal()
        observation = np.append([distance_to_goal, self.angle], self.point_cloud)
        return observation

    def step(
            self, action: RobotAction, render: bool = False
    ) -> Tuple[np.array, int, bool]:
        # actions are a linear velocity and an angular velocity
        # reward of 100 is given for getting to goal
        # reward of -100 is given for collision
        if action == RobotAction.FORWARD:
            self.vel = 1
            self.rotational_vel = 0
        elif action == RobotAction.ROTATE_LEFT:
            self.vel = 0
            self.rotational_vel = 1
        elif action == RobotAction.ROTATE_RIGHT:
            self.vel = 0
            self.rotational_vel = -1

        # update the simulation env
        # tick one timestep
        if render or self.manual:
            self._update()
        else:
            self._update_data()

        # the parameters to return
        reward = 0
        done = False
        max_dist = (self.width ** 2 + self.height ** 2) ** .5
        optimal_angle = self._normalize_angle(round(-np.arctan2(self.goal[1] - self.pos[1],
                                                                (self.goal[0] - self.pos[0])) * 180/np.pi))
        angle_delta = abs(self._normalize_angle(self.angle)) - abs(self._normalize_angle(optimal_angle))

        distance_to_goal = self._distance_to_goal()

        if self._within_goal():
            self.reset(render=render)
            reward = 500.0
            done = True

        elif self.has_collision(self.pos):
            self.reset(render=render)
            reward = -100.0
            done = True
        else:
            # return the next state
            # calculate how close the goal is to the pos
            dist_reward = 5 * (distance_to_goal / max_dist)

            # calculate if the robot is facing towards the goal
            angle_reward = (abs(angle_delta) / 180)

            # print(optimal_angle, self.angle)
            # print('dist', dist_reward, 'angle', angle_reward)

            if action == RobotAction.ROTATE_RIGHT or action == RobotAction.ROTATE_LEFT:
                # can make reward more specific for angles here too
                reward = -angle_reward
            else:
                # if abs(angle_delta) < 10:
                reward = dist_reward
                # else:
                #     reward = -dist_reward - angle_reward
            # else:
            #     reward = -dist_reward

            # print(self.angle, optimal_angle)
            # print(dist_reward, angle_reward)

            # reward = -(dist_reward + angle_reward)
            done = False

        observation = np.append([distance_to_goal, angle_delta], self.point_cloud)
        return observation, reward, done


if __name__ == "__main__":
    world = GenericWorld(500, 500, manual=True)
    world.render()
    # world.reset()

    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
    # print(world.step(RobotAction.FORWARD))
