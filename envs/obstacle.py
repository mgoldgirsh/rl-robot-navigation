import math
from typing import Tuple
import numpy as np


class Obstacle:
    def __init__(self, x, y, width, height) -> None:
        self.x1 = x
        self.y1 = y
        self.x2 = x + width
        self.y2 = y + height
        self.intersections = []

    def collides_with(self, position: Tuple[int, int]) -> bool:
        if (self.x1 <= position[0] <= self.x2) and (self.y1 <= position[1] <= self.y2):
            return True
        else:
            return False


if __name__ == "__main__":
    o = Obstacle(100, 100, 50, 50)
    print(o.calculate_intersection(-90, (20, 20), 500 * np.sqrt(2)))
