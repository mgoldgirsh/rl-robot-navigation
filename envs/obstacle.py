import tkinter as tk
from typing import Tuple
import numpy as np
import random


class Obstacle:
    def __init__(self, x, y, width, height) -> None:
        self.x1 = x
        self.y1 = y
        self.x2 = x + width
        self.y2 = y + height
        self.intersections = []

    def distance_to_robot(self, robot_pos: Tuple[int, int], fov_angle: int):
        # determines the distance between the robot and nearest collision of the obstacle using
        # the fov angle as the assumed robot angle
        if 0 <= fov_angle < 90:
            # you have to be looking at top/right of the obs
            horiz_dist = self.x1 - robot_pos[0]
            vert_dist = robot_pos[1] - self.y2
        elif 90 <= fov_angle < 180:
            # looking at top/left walls
            vert_dist = robot_pos[1] - self.y2
            horiz_dist = robot_pos[0] - self.x2
        elif -90 <= fov_angle < 0:
            # looking at bottom/right walls
            vert_dist = self.y1 - robot_pos[1]
            horiz_dist = self.x1 - robot_pos[0]
        else:
            # looking at bottom/left walls
            vert_dist = self.y1 - robot_pos[1]
            horiz_dist = robot_pos[0] - self.x2

        if fov_angle == 0 or fov_angle == -180 or fov_angle == 180:
            dist = horiz_dist
        if fov_angle == 90 or fov_angle == -90:
            dist = vert_dist
        else:
            # calculate vert/horiz distance from the top and left points of intersection
            calc_vert = np.tan(fov_angle * np.pi / 180) * horiz_dist
            calc_horiz = 1 / np.tan(fov_angle * np.pi / 180) * vert_dist

            if abs(calc_vert) < abs(calc_horiz):
                dist = (calc_vert ** 2 + horiz_dist ** 2) ** .5
            else:
                dist = (calc_horiz ** 2 + vert_dist ** 2) ** .5

        new_pos = (robot_pos[0] + dist * np.cos(fov_angle * np.pi / 180),
                   robot_pos[1] - dist * np.sin(fov_angle * np.pi / 180))

        if self.collides_with(new_pos):
            return dist
        else:
            return 2 ** 31

    def collides_with(self, position: Tuple[int, int]) -> bool:
        if (self.x1 <= position[0] <= self.x2) and (self.y1 <= position[1] <= self.y2):
            return True
        else:
            return False

    def create_obstacle(self, canvas: tk.Canvas):
        canvas.create_rectangle(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            fill="black",
            tags=("obstacle"),
        )

    @staticmethod
    def random_obstacle(board_width, board_height) -> "Obstacle":
        random_x = random.randint(10, board_width)
        random_y = random.randint(10, board_height)
        random_width = random.randint(10, max((board_width - random_x) // 3, 10))
        random_height = random.randint(10, max((board_height - random_y) // 3, 10))
        return Obstacle(random_x, random_y, random_width, random_height)


if __name__ == "__main__":
    o = Obstacle(100, 100, 50, 50)
