from generic_env import GenericWorld
from obstacle import Obstacle

class MazeWorld(GenericWorld):
    def __init__(self, width: int, height: int, refresh_rate: int = 10, manual: bool = False, see_all: bool = False):
        super().__init__(width, height, refresh_rate, manual, see_all=see_all)
        
        # modify the starting pos
        self.pos = (20, 20)
        
        layers = 10
        
        # add the obstacles to the maze world
        self.add_obstacles([
            # borders
            Obstacle(x=0, y=0, width=width, height=5), 
            Obstacle(x=0, y=0, width=5, height=height), 
            Obstacle(x=width-5, y=0, width=5, height=height), 
            Obstacle(x=0, y=height-5, width=width, height=5),
            
            # obstacle maze
            Obstacle(x=0, y=height / layers, width=width*.9, height=5),
            Obstacle(x=width *.1, y=height / layers * 2, width=width*.9, height=5),
            Obstacle(x=0, y=height / layers * 3, width=width*.9, height=5),
            Obstacle(x=width *.1, y=height / layers * 4, width=width*.9, height=5),
            Obstacle(x=0, y=height / layers * 5, width=width*.9, height=5),
            Obstacle(x=width *.1, y=height / layers * 6, width=width*.9, height=5),
            Obstacle(x=0, y=height / layers * 7, width=width*.9, height=5),
            Obstacle(x=width *.1, y=height / layers * 8, width=width*.9, height=5),
            Obstacle(x=0, y=height / layers * 9, width=width*.9, height=5),
        ])


if __name__ == "__main__":
    maze_world = MazeWorld(500, 500, manual=True)
    maze_world.reset()
    maze_world.render()