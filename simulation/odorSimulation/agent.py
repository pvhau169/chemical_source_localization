import random
import numpy as np

from simulation.odorSimulation import vector


class Agent:
    def __init__(self, agent_arg, env_arg):
        self.agent_arg = agent_arg

        for key, value in env_arg.items():
            setattr(self, key, value)

        for key, value in agent_arg.items():
            setattr(self, key, value)

        self.reset()

    def reset(self):
        def randomStart():
            for i in range(2):
                bound_min, bound_max = self.boundary[i]
                bound_range = bound_max - bound_min

                random_bound_min = bound_min + bound_range * self.random_range[i][0]
                random_bound_max = bound_min + bound_range * self.random_range[i][1]
                self.position[i] = random.uniform(random_bound_min, random_bound_max)

        # random direction
        if self.random_start:
            randomStart()
            self.start_position = self.position.copy()

        self.position = self.start_position.copy()

        # random direction
        self.direction_index = np.random.randint(len(self.directions)) if self.random_direction else 0
        self.direction = self.directions[self.direction_index]
        self.last_direction_index = np.random.randint(len(self.directions))


        self.trajectory = []
        self.ego_trajectory = []

        #updown action
        self.updown_boundary = self.boundary.copy()
        self.updown_boundary[1] = self.updown_boundary[1, 0] + (self.updown_boundary[1, 1] - self.updown_boundary[1, 0]) * self.updown_range
        # cyclone action
        self.count_switch = 0
        self.count_cur_step = 0
        self.max_dir_step = 1

    def getAction(self):
        if self.action_mode == 1:
            return self.getCycloneAction()
        elif self.action_mode == 0:
            return self.getUpDownAction()
        elif self.action_mode == 2:
            return self.getStochasticAction()
        else:
            return self.getRandomAction()

    def getRandomAction(self):
        self.direction_index = np.random.randint(len(self.directions))
        return self.direction_index

    def getCycloneAction(self):
        self.count_cur_step += 1

        if self.count_cur_step == self.max_dir_step:
            self.direction_index = (self.direction_index + 1) % len(self.directions)
            self.count_switch += 1
            self.count_cur_step = 0
            if self.count_switch == 2:
                self.count_switch = 0
                self.max_dir_step += 1

        return self.direction_index

    def getStochasticAction(self):
        random_num = np.random.rand()
        if random_num<0.6:
            self.direction_index = self.last_direction_index
        # elif random_num<0.9:
        else:
            # self.direction_index = (self.last_direction_index + 1) % len(self.directions)
            self.direction_index = np.random.randint(len(self.directions))
        # else:
        #     self.direction_index = (self.last_direction_index + 2) % len(self.directions)
        self.last_direction_index = self.direction_index
        return self.direction_index

    def getUpDownAction(self):
        def validPosition(position, boundary):
            for i in range(len(boundary)):
                bound_min, bound_max = boundary[i]
                if position[i] < bound_min or position[i] > bound_max:
                    return 0
            return 1

        return self.direction_index
        if validPosition(self.expectPosition(self.direction), self.updown_boundary) == 0:
            self.direction_index = (self.direction_index + 2) % len(self.directions)

        return self.direction_index

    def getRandomAction(self):
        random_index = np.random.randint(len(self.directions))
        while self.validPosition(self.expectPosition(self.directions[random_index])) == 0:
            random_index = np.random.randint(len(self.directions))
        return random_index

    def validPosition(self, position, boundary = None):
        for i in range(len(self.boundary)):
            bound_min, bound_max = self.boundary[i]
            if position[i] < bound_min or position[i] > bound_max:
                return 0
        return 1

    def expectPosition(self, direction):
        return self.position + direction * self.dt

    def act(self, action):
        self.last_action = action

        # steering action
        self.direction = self.directions[action]

        # moving
        self.direction = vector.limit_magnitude(self.direction, self.speed, self.speed)
        self.position = self.expectPosition(self.direction)
        self.ego_position = self.position - self.start_position

        # update trajectory
        self.trajectory.append(self.position)
        self.ego_trajectory.append(self.ego_position)

    def setWind(self, wind):
        self.wind = wind
