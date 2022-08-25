import random
import numpy as np
from simulation import vector2D
import simulation.vector2D as vector

water_flow_force_factor = 0.2


# radial_direction_factor = 0.5

class Agent(object):
    def __init__(self,
                 agent_arg,
                 position=[120, 0],
                 direction=[0, 1],
                 color=[1, 0.5, 0.5],
                 color_name='yellow',
                 water_flow_force=[10, -1],
                 dt=5,
                 mode="up_down",

                 ):
        self.mode = mode
        # start initialize
        self.start = np.array(position)
        self.start_direction = np.array(direction)

        # absolute coordinate
        self.position, self.trajectory = [], []

        # ego coordinate
        self.ego_position, self.ego_trajectory = [], []

        self.direction = []
        self.color = color
        self.color_name = color_name

        self.speed = 1

        self.directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.action_size = len(self.directions)
        if self.mode != "up_down":
            self.steer = np.array([i for i in range(-2, 3)])
            self.angle = np.pi / 4
            self.action_size = len(self.steer)
        self.last_action = 0
        self.dt = dt

        # set mode
        self.random_range = [[0.4, 1], [0, 1]]
        for key, value in agent_arg.items():
            setattr(self, key, value)

        self.reset()

    def getObservation(self):
        return np.array(self.ego_position)
        return np.concatenate((np.array(self.ego_position), np.array([self.previous_action])))

    def getEgoPosition(self):
        return self.position - self.start

    def randomPosition(self):
        for i in range(2):
            bound_min, bound_max = self.boundary[i]
            bound_range = bound_max - bound_min

            random_bound_min = bound_min + bound_range * self.random_range[i][0]
            random_bound_max = bound_min + bound_range * self.random_range[i][1]
            self.position[i] = random.uniform(random_bound_min, random_bound_max)

        # self.position[0] = self.boundary[0][1]
        self.start = self.position.copy()

    def randomDirection(self):
        direction_index = np.random.randint(len(self.directions))
        self.direction = self.directions[direction_index]
        self.start_direction = self.direction

    def randomStart(self):
        self.randomPosition()
        self.randomDirection()
        self.reset()

    def reset(self):
        self.hit_wall = 0
        self.position = self.start.copy()
        self.direction = self.start_direction.copy()
        self.ego_position = self.getEgoPosition()
        self.trajectory = []
        self.ego_trajectory = []

    def expectPosition(self, direction, dt):
        return np.array(self.position) + np.array(direction) * dt

    def boundaryLimit(self, pos_temp):
        for i in range(len(self.boundary)):
            bound_min, bound_max = self.boundary[i]
            if pos_temp[i] < bound_min:
                pos_temp[i] = bound_min
                self.hit_wall = 1
            if pos_temp[i] > bound_max:
                pos_temp[i] = bound_max
                self.hit_wall = 1

        return pos_temp

    def act(self, action):
        self.last_action = action



    def move(self, dt):
        self.direction = vector2D.limit_magnitude(self.direction, self.speed, self.speed)
        new_pos = self.expectPosition(self.direction, dt)

        self.position = new_pos
        self.ego_position = self.getEgoPosition()

    def update(self):
        # discrete movement
        self.trajectory.append(self.position)
        self.ego_trajectory.append(self.getEgoPosition())
