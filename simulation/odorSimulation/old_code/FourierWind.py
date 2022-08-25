import torch
from matplotlib import pyplot as plt

from simulation.odor2D import OdorSource, OdorPacket
from simulation.agent2D import Agent

from simulation.aquatic_world_heat_map_func import *
import random
import simulation.vector2D as vector
from gym import spaces
from celluloid import Camera
from tqdm.notebook import tqdm


class FourierWind:
    def __init__(self, turbulence_arg={}):
        self.energy = 0.42
        self.length_scale = 10
        self.tau_f = 5

        self.sampling_interval = 1 / 5
        self.time_step = 130
        self.k_x, self.k_y, self.a_c_k_t = None, None, None
        self.water_flow_force = [1, 0]

        for key, value in turbulence_arg.items():
            setattr(self, key, value)

        ts = np.arange(0, self.time_step + 100, self.sampling_interval)
        self.ts = np.atleast_2d(ts).T
        self.getTurbulenceMap()

    def reset(self):
        self.t = -1

    def randomWaterFlow(self):
        self.water_flow_force = np.array([random.uniform(0, 5), 0])

    def getTurbulenceMap(self):
        k_s = 2 * np.pi / self.length_scale  # spatial frequency in radian

        k1 = np.array([[0, k_s],
                       [k_s, 0]])
        k2 = np.array([[k_s, k_s],
                       [k_s, -k_s]])

        k = np.concatenate((k1, k2), axis=0)

        self.k_x = k[:, 0]
        self.k_y = k[:, 1]

        a = np.zeros(len(self.ts))
        c = np.exp(-np.abs(self.ts - self.ts.T) / self.tau_f)
        a_k_t = np.array([np.random.multivariate_normal(a, c) for i in range(8)]).T.reshape((len(self.ts), 4, 2))

        self.a_c_k_t = np.vectorize(complex)(a_k_t[..., 0], a_k_t[..., 1])

    def multiply(self, pos, t_index, k_variable):
        x, y = pos
        temp1 = np.vectorize(complex)(np.zeros(k_variable.shape), k_variable)

        temp2 = x * self.k_x + y * self.k_y
        temp2 = np.exp(np.vectorize(complex)(np.zeros(temp2.shape), temp2))
        result = np.matmul(np.atleast_2d(temp1 * temp2), np.atleast_2d(self.a_c_k_t[t_index]).T)
        result = result[0][0].real
        return result

    def getTurbulenceVector(self, pos, t):
        t_index = int(t / self.sampling_interval)

        result_x = self.multiply(pos, t_index, self.k_x)
        result_y = self.multiply(pos, t_index, self.k_y)
        return np.array([result_x, result_y]) + self.water_flow_force
