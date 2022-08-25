import random
import numpy as np

from simulation.odorSimulation import vector


class OdorSource:
    def __init__(self, source_arg, env_arg):
        self.source_arg = source_arg

        for key, value in env_arg.items():
            setattr(self, key, value)
        for key, value in source_arg.items():
            setattr(self, key, value)

        self.reset()

    def reset(self):
        def randomStart():
            # for i in range(2):
            #     bound_min, bound_max = self.boundary[i]
            #     bound_range = bound_max - bound_min
            #
            #     random_bound_min = bound_min + bound_range * self.random_range[i][0]
            #     random_bound_max = bound_min + bound_range * self.random_range[i][1]
            #     self.position[i] = random.uniform(random_bound_min, random_bound_max)
            radius = np.random.gamma(self.source_pos_shape, self.source_pos_scale)
            angle = np.random.uniform(0, 2* np.pi)
            self.position = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        # reset parameters
        self.t = -1
        self.n_packet = 0

        # reset concentration
        if self.random_concentration:
            self.concentration = random.uniform(self.min_concentration, self.max_concentration)
        self.spawn_rate = int(self.concentration / self.spawn_scale)

        #random Position
        if self.random_start:
            randomStart()
            self.wind.switchWind(self.position)

        # reset burn in
        self.in_boundary = 0
        self.fill = 0
        self.update()
        if self.burn_in:
            while self.fill == 0:
                self.update()

    def update(self):
        def validPosition(positions, boundary):
            final_mask = np.ones(len(positions)).astype(np.bool)
            for i in range(len(boundary)):
                bound_min, bound_max = boundary[i]
                axis_mask = (positions[:, i] >= bound_min) & (positions[:, i] <= bound_max)
                final_mask = final_mask & axis_mask
            return final_mask
        def spawnPackets():
            n_new_packet = max(np.random.poisson(self.spawn_rate), 1)
            new_packets_position = np.random.uniform(-5, 5, (n_new_packet, 2)) + self.position  # n_packets , 2
            new_packets_concentration = np.ones((n_new_packet, 1)) * self.concentration
            new_packets_time_step = np.ones((n_new_packet, 1))

            if self.t == -1:
                self.packets_position = new_packets_position
                self.packets_concentration = new_packets_concentration
                self.packets_time_step = new_packets_time_step
            else:
                self.packets_position = np.concatenate((self.packets_position, new_packets_position), axis=0)
                self.packets_concentration = np.concatenate((self.packets_concentration, new_packets_concentration),
                                                            axis=0)
                self.packets_time_step = np.concatenate((self.packets_time_step, new_packets_time_step), axis=0)

            self.n_packet += n_new_packet

        def applyMask(mask):
            mask = mask.squeeze(1)
            if np.sum(mask) < self.n_packet:
                self.fill = 1
            self.packets_position = self.packets_position[mask]
            self.packets_concentration = self.packets_concentration[mask]
            self.packets_time_step = self.packets_time_step[mask]
            self.n_packet = len(self.packets_position)

        # spawn more packets
        spawnPackets()

        # update position, concentration, time_step
        self.packets_time_step += 1
        self.t += 1
        self.packets_concentration = self.packets_concentration * self.decay_rate
        packets_water_flow_force = self.wind.getTurbulenceVector(self.packets_position, self.t)  # n_packet, 2
        packets_random_turbulence = np.random.uniform(-self.max_random_turbulence, self.max_random_turbulence,
                                                      (self.n_packet, 2))
        self.packets_position = self.packets_position + (packets_water_flow_force+ packets_random_turbulence) * self.dt

        # remove low concentration packets
        concentration_mask = self.packets_concentration >= self.concentration_threshold
        applyMask(concentration_mask)
        position_mask = validPosition(self.packets_position, self.boundary)
        if np.sum(position_mask)>20:
            self.in_boundary = 1

    def setWind(self, wind):
        self.wind = wind
