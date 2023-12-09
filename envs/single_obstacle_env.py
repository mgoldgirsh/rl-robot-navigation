from generic_env import GenericWorld
from obstacle import Obstacle

class SingleObstacleWorld(GenericWorld):
    
    def __init__(self, width: int, height: int, refresh_rate: int = 10, manual: bool = False, see_all: bool = False) -> None:
        super().__init__(width, height, refresh_rate, manual, see_all)
        self.pos = (20, 20)
        self.add_obstacles([
            Obstacle(100, 100, 50, 50)
        ])

if __name__ == "__main__":
    world = SingleObstacleWorld(width=500, height=500, manual=True)
    world.render()