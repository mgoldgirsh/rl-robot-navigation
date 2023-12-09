from .generic_env import GenericWorld
from .obstacle import Obstacle

class ObstaclesWorld(GenericWorld):
    def __init__(self, width: int, height: int, refresh_rate: int = 10, manual: bool = False, see_all: bool = False) -> None:
        super().__init__(width, height, refresh_rate, manual, see_all)
        self.reset()
        self.add_obstacles([
            # three boxes
            Obstacle(100, 100, 50, 50),
            Obstacle(350, 350, 50, 50),
            Obstacle(350, 50, 50, 50),
            Obstacle(50, 400, 50, 50),
        
            
            # l shape
            Obstacle(200, 200, 25, 100),
            Obstacle(225, 275, 75, 25)
        ])

if __name__ == "__main__":
    world = ObstaclesWorld(width=500, height=500, manual=True, see_all=False)
    world.render()