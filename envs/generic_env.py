from tkinter import *
import numpy as np
from typing import List, Tuple
from .obstacle import Obstacle
from threading import Thread
from enum import IntEnum

class RobotAction(IntEnum):
    FORWARD = 0, # vel = 1, rotation = 0
    ROTATE_LEFT = 1 # vel = 0, rotation = 3s
    ROTATE_RIGHT = 2 # vel = 0, rotation = -3
    

class GenericWorld: 
    

    def __init__(self, width: int, height: int, refresh_rate:int=10, manual:bool=False, see_all:bool=False) -> None:
        self.width = width
        self.height = height
        
        # ---- robot parameters
        self.pos = (width//2, height//2) # the position of the robot
        self.angle = 0 # the angle of the robot in degrees (0 represents the right/EAST direction)
        
        self.vel = 0 # the linear velocity
        self.rotational_vel = 0 # the angular velocity
        self.fov = 180 # the field of view of a robot (from its orientiation angle) (current 180 fov)
        
        # obstacles
        self._obstacles = []
        
        # the goal to goto
        self.goal = self._generate_random_position()
        self.goalbox = 10 # the full box side length around the goal
        
        # the maximum view that the robot can see 
        # this assumes that the robot if possible can see the whole environment
        self.max_view = np.sqrt(self.width ** 2 + self.height ** 2) 
        
        # the distances objects are from the location of the robot
        # this is called the point cloud
        # the first index is the minimum most location of the point cloud
        self.point_cloud = np.ones(shape=self.fov//3+1) 
        
        # whethere to accept manual inputs
        self.manual = manual
        # whether to see all obstacles and not just point cloud
        self.see_all = see_all
        
        # refresh rate
        self.refresh_rate = refresh_rate
        self.tick = 0
        
        # rendering elmeents
        self.top = Tk()
        self.canvas = Canvas(self.top, height=height, width=width)
        
        # on key press
        if (manual):
            # background tick update
            self.top.after(self.refresh_rate, self._update)
            self.top.bind('<KeyPress>', self._onKeyPress)
    
        
    def _draw_cursor(self, size=20) -> None:
        coord = self.pos[0] - size//2, self.pos[1] - size//2, self.pos[0] + size, self.pos[1] + size
        self.canvas.create_arc(coord, start=157.5 + self.angle, extent=45, fill="red", tags=('robot'))


    def _draw_obstacles(self) -> None:
        for obstacle in self._obstacles:
            self.canvas.create_rectangle(obstacle.x1, obstacle.y1, obstacle.x2, obstacle.y2, fill='black', tags=('obstacle'))


    def _draw_goal(self) -> None: 
        box_side = self.goalbox / 2
        goal_rect = self.goal[0] - box_side, self.goal[1] - box_side, self.goal[0] + box_side, self.goal[1] + box_side
        self.canvas.create_rectangle(goal_rect, fill='green', tags=('goal'))      

    
    def _draw_fov(self, size=2) -> None:
        i = 0
        
        for fov_angle in range(self.angle - self.fov // 2, self.angle + self.fov // 2+1, 3):
            updated_pos = (self.pos[0] + np.cos(-fov_angle * np.pi/180) * self.point_cloud[i],
                           self.pos[1] + np.sin(-fov_angle * np.pi/180) * self.point_cloud[i])
            self.canvas.create_oval(updated_pos[0] - size/2, updated_pos[1] - size/2, 
                                    updated_pos[0] + size/2, updated_pos[1] + size/2, fill='black', tags='fov')
            i += 1


    def _update_position(self) -> None:
        # update the angle first 
        self.angle += self.rotational_vel
        
        updated_vel = (self.vel * np.cos(-self.angle * np.pi / 180), 
                       self.vel * np.sin(-self.angle * np.pi / 180))
        updated_pos = (self.pos[0] + updated_vel[0], self.pos[1] + updated_vel[1])        
        
        if (self.manual):
            if not self.has_collision(updated_pos):
                self.pos = updated_pos
        else:
            self.pos = updated_pos
            
    
    def _calculate_distance_from(self, fov_angle: int, index: int) -> float:        
        constants = (self.pos[0], self.pos[1])
        
        min_distance = self.max_view
        for obstacle in self._obstacles:
            test_distance = obstacle.calculate_intersection(fov_angle, constants, self.max_view)

            if (test_distance < min_distance):
                min_distance = test_distance
            
        self.point_cloud[index] = min_distance
        return min_distance
    
    
    def _update_point_cloud(self) -> None:
        i = 0
        
        fov_angles = range(self.angle - self.fov // 2, self.angle + self.fov // 2 + 1, 3)

        # Threaded code
        for fov_angle in fov_angles:
            t = Thread(target = lambda: self._calculate_distance_from(fov_angle, i))
            t.start()
            t.join()
            i += 1


    def _update_data(self) -> None:
        # update position and point cloud
        self._update_position()
        self._update_point_cloud()

            
    def _update(self) -> None:
        # print(self.angle, self.point_cloud[0], self.point_cloud[90], self.point_cloud[180])
        # print(self.point_cloud)
        # print(self.pos, self.vel, self.angle)
        # reset canvas
        self.canvas.delete("all")
        
        # update all data
        self._update_data()
        
        # re-render pos and obstacles
        self._draw_cursor()
        self._draw_goal()
        if (self.see_all):
            self._draw_obstacles()
        self._draw_fov()

        self.canvas.update()
        self.canvas.pack()
        if (self.manual):
            self.top.after(self.refresh_rate, self._update)
        
        
    def _onKeyPress(self, event) -> None:
        if (event.keysym == "Left"):
            print('rotating')
            self.vel = 0
            self.rotational_vel = 1
        elif (event.keysym == "Right"):
            print('rotating')
            self.vel = 0 
            self.rotational_vel = -1
        elif (event.keysym == "Up"):
            print('moving forward')
            self.rotational_vel = 0
            self.vel = 1
        elif (event.keysym == "Down"):
            print('moving backwards')
            self.rotational_vel = 0
            self.vel = -1
        elif (event.keysym == '0'):
            print('0 pressed | velocity reset | angle reset')
            self.vel = 0
            self.rotational_vel = 0


    def render(self) -> None:
        self.top.mainloop()
    
    
    def add_obstacles(self, obstacles: List[Obstacle]) -> None:
        for obstacle in obstacles:
            self._obstacles.append(obstacle)    


    def has_collision(self, position) -> bool:
        collides = any([obstacle.collides_with(position) for obstacle in self._obstacles])
        
        on_border = False
        if (position[0] <= 0 or position[0] >= self.width or position[1] <= 0 or position[1] >= self.height):
            on_border = True
        
        return collides or on_border


    def _generate_random_position(self) -> Tuple[int, int]:
        random_pos = (np.random.randint(self.width), np.random.randint(self.height))
        
        collision = False
        if (len(self._obstacles) != 0):
            collision = self.has_collision(random_pos)
    
        while (collision):
            random_pos = (np.random.randint(self.width), np.random.randint(self.height))
            if (len(self._obstacles) != 0):
                collision = self.has_collision(random_pos)
        
        return random_pos


    def _within_goal(self) -> bool:
        box_side = self.goalbox / 2
        if ((self.pos[0] >= self.goal[0] - box_side and self.pos[0] <= self.goal[0] + box_side)
            and (self.pos[1] >= self.goal[1] - box_side and self.pos[1] <= self.goal[1] + box_side)):
            return True
        else:
            return False
    
                
    def reset(self, render:bool = False) -> np.array:
        self.pos = self._generate_random_position()
        self.goal = self._generate_random_position()
        self.angle = 0
        self.vel = 0 
        self.rotational_vel = 0
        if (render or self.manual):
            self._update()
        else:    
            self._update_data()
        
        distance_to_goal = ((self.pos[0] - self.goal[0]) ** 2 + (self.pos[1] - self.goal[1]) ** 2) ** 0.5
        observation = np.append([distance_to_goal, self.angle], self.point_cloud)
        return observation


    def step(self, action: RobotAction, render:bool = False) -> Tuple[np.array, int, bool]: 
        # actions are a linear velocity and an angular velocity
        # reward of 100 is given for getting to goal
        # reward of -100 is given for collision
        if (action == RobotAction.FORWARD):
            self.vel = 1
            self.rotational_vel = 0
        elif (action == RobotAction.ROTATE_LEFT):
            self.vel = 0
            self.rotational_vel = 3
        elif (action == RobotAction.ROTATE_RIGHT):
            self.vel = 0
            self.rotational_vel = -3
            
        # update the simulation env
        # tick one timestep
        if (render or self.manual):
            self._update()
        else:
            self._update_data()
        
        # the parameters to return
        reward = 0
        done = False
        distance_to_goal = ((self.pos[0] - self.goal[0]) ** 2 + (self.pos[1] - self.goal[1]) ** 2) ** 0.5
        
        if (self._within_goal()):
            self.reset()
            reward = 100.0
            done = True
        
        # wrong here
        elif (self.has_collision(self.pos)):
            self.reset()
            reward = -100.0
            done = True
        else: 
            # return the next state 
            # calculate how close the goal is to the pos 
            if (action == RobotAction.ROTATE_LEFT or action == RobotAction.ROTATE_RIGHT):
                reward = -0.5
            else:
                reward =  1 - (distance_to_goal / self.max_view)
            done = False
            
        
        observation = np.append([distance_to_goal, self.angle], self.point_cloud)
        return observation, reward, done


if __name__ == "__main__":
    world = GenericWorld(500, 500, manual=False)
    world.reset()
    
    print(world.step(RobotAction.FORWARD))
    print(world.step(RobotAction.FORWARD))
    print(world.step(RobotAction.FORWARD))
    print(world.step(RobotAction.FORWARD))
    print(world.step(RobotAction.FORWARD))
    print(world.step(RobotAction.FORWARD))
    print(world.step(RobotAction.FORWARD))



