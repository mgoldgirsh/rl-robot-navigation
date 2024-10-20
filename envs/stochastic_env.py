from envs.generic_env import GenericWorld
from envs.obstacle import Obstacle

import random


class StochasticWorld(GenericWorld):
    def __init__(
            self,
            width: int,
            height: int,
            refresh_rate: int = 10,
            manual: bool = False,
            see_all: bool = False,
    ) -> None:
        super().__init__(width, height, refresh_rate, manual, see_all)
        num_obstacles = random.randint(1, 10)

        obstacles = []
        for obs in range(num_obstacles):
            obstacles.append(Obstacle.random_obstacle(self.width, self.height))

        self.add_obstacles(obstacles)

        self.reset()

    def _generate_new_world(self):
        super()._generate_new_world()
        self.canvas.delete("all")
        self.clear_obstacles()
        num_obstacles = random.randint(1, 10)

        obstacles = []
        for obs in range(num_obstacles):
            obstacles.append(Obstacle.random_obstacle(self.width, self.height))

        self.add_obstacles(obstacles)


if __name__ == "__main__":
    world = StochasticWorld(width=500, height=500, manual=True, see_all=True)
    world.render()
