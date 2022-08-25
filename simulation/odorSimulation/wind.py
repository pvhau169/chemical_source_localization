import numpy as np
import random

class FourierWind:
    def __init__(self, wind_arg, env_arg):
        #set wind parameters
        self.energy = 0.42,
        self.length_scale = 10
        self.tau_f = 5
        self.sampling_interval = 1/5

        self.episode_limit = 130

        for key, value in env_arg.items():
            setattr(self, key, value)
        for key, value in wind_arg.items():
            setattr(self, key, value)

        self.reset()

    def reset(self):
        self.t = -1

        self.ts = np.atleast_2d(np.arange(0, self.episode_limit + 100, self.sampling_interval)).T

        #random Water Flow
        self.randomWaterFlow()


        self.createTurbulenceMap()

    def randomWaterFlow(self):
        # if self.random_water_flow:
        #     self.water_flow_force = np.array([random.uniform(self.min_water_flow_force, self.max_water_flow_force), 0])
        # else: self.water_flow_force = np.array([self.max_water_flow_force, 0])

        if self.random_water_flow:
            radius = np.random.gamma(self.water_flow_force_shape, self.water_flow_force_scale)
            angle = np.random.uniform(0, 2*np.pi)
            self.water_flow_force = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        else: self.water_flow_force = np.array([self.max_water_flow_force, 0])


    def switchWind(self, source_position):
        # position = (2)
        # boundary = (2, 2)
        boundary = self.boundary
        if source_position[0] > boundary[0, 0] + (boundary[0, 1] - boundary[0, 0]) /2:
            self.water_flow_force *= -1

    def createTurbulenceMap(self):
        k_s = 2 * np.pi / self.length_scale  # spatial frequency in radian

        k1 = np.array([[0, k_s],
                       [k_s, 0]])
        k2 = np.array([[k_s, k_s],
                       [k_s, -k_s]])

        k = np.concatenate((k1, k2), axis=0)
        self.k = k

        self.k_x = k[:, 0]
        self.k_y = k[:, 1]

        a = np.zeros(len(self.ts))
        c = np.exp(-np.abs(self.ts - self.ts.T) / self.tau_f)
        a_k_t = np.array([np.random.multivariate_normal(a, c) for i in range(8)]).T.reshape((len(self.ts), 4, 2))

        self.a_c_k_t = np.vectorize(complex)(a_k_t[..., 0], a_k_t[..., 1])

    def getTurbulenceVector(self, pos, t):
        t_index = int(t/self.sampling_interval)
        # pos = (n_packet, 2), t
        temp1 = np.vectorize(complex)(np.zeros_like(self.k.T), self.k.T)
        temp2 = np.exp(np.vectorize(complex)(np.zeros_like(pos @ self.k.T), pos @ self.k.T))
        temp = temp1[np.newaxis, ...] * temp2[:, np.newaxis, :]
        result = temp @ np.atleast_2d(self.a_c_k_t[t_index]).T
        return result.squeeze(-1).real + self.water_flow_force #(n_packet, 2)




