from typing import Tuple
import numpy as np
from threading import Thread

class Obstacle: 
    
    
    def __init__(self, x, y, width, height) -> None:
        self.x1 = x
        self.y1 = y
        self.x2 = x + width
        self.y2 = y + height
        
    def collides_with(self, position: Tuple[int, int]) -> bool:
        if ((position[0] >= self.x1 and position[0] <= self.x2) and 
            (position[1] >= self.y1 and position[1] <= self.y2)):
            return True
        else:
            return False
    
    
    def dist_interval(self, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        test_points = [(self.x1, self.y1), (self.x1, self.y2), (self.x2, self.y1), (self.x2, self.y2)]
        distance_betweens = [self.dist_between(current_pos, test) for test in test_points]
        min_dist = min(distance_betweens)
        max_dist = max(distance_betweens)
        return min_dist, max_dist
    
        
    def calculate_intersection(self, fov_angle, current_pos: Tuple[float, float], max_view: float) -> float:
        min_dist, max_dist = self.dist_interval(current_pos)
        if (min_dist < 50):
            min_dist = 0
        # print(min_dist, max_dist)
        intersection = min_dist
        intervals = 100
        delta = (max_dist - min_dist) / intervals
        
        
        for _ in range(intervals):
            update_pos = (current_pos[0] + intersection * np.cos(-fov_angle * np.pi/180), 
                        current_pos[1] + intersection * np.sin(-fov_angle * np.pi/180))
            if (self.collides_with(update_pos)):
                return intersection
            else:
                intersection += delta
        
        # if no intersection return the default value
        return max_view
    @staticmethod
    def absolute_min(i1, i2):
        if (abs(i1) < abs(i2)):
            return i1
        else:
            return i2
        
    @staticmethod
    def dist_between(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return ((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) ** 0.5
    
    
if __name__ == "__main__":
    o = Obstacle(100, 100, 50, 50)
    print(o.calculate_intersection(-90, (20, 20)))