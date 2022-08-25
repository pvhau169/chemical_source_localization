import random
import numpy as np

from simulation import vector2D

# from aquatic_world import aquaEnv

_MAX_SPEED = 8
_MIN_SPEED = 0


# _GRAVITY_FACTOR = 0.5


class OdorPacket:
    def __init__(self,
                 packet_arg,
                 wind,
                 position=[0, 0],
                 concentration=1,
                 decay_rate=0.97,
                 color=[1, 0, 0],
                 t=0
                 ):
        self.wind = wind
        self.position = position
        # self.water_flow_force = water_flow_force

        # Generate random Turbulence force
        self.turbulence_force = np.array([0, 0])

        self.concentration = concentration
        self.decay_rate = decay_rate

        self.color = color

        self.t = t
        self.random = True

        self.size = 1
        self.size_growth = 1.03

        for key, value in packet_arg.items():
            setattr(self, key, value)

    def randomTurbulence(self):
        max_magnitude = 2
        return np.array([random.uniform(-max_magnitude, max_magnitude), random.uniform(-max_magnitude, max_magnitude)])

    def update(self, t, dt=1):
        # decay concentration
        self.concentration *= self.decay_rate
        water_flow_force = self.wind.getTurbulenceVector(self.position, t)
        velocity = np.array(water_flow_force) + np.array(self.randomTurbulence())

        # velocity = np.array(turbulence_force)
        # velocity = vector2D.limit_magnitude(velocity, _MAX_SPEED, _MIN_SPEED)

        self.position = self.position + dt * velocity
        self.size *= self.size_growth


class OdorSource:
    def __init__(self,
                 source_arg,
                 wind,
                 position=[20, 0],
                 boundary=[(-20, 500), (-200, 200)],
                 water_flow_force=[10, -1],
                 water_flow_force_threshold=10,
                 concentration=1,
                 decay_rate=0.97,
                 color=[1, 0, 0],
                 max_packet=1000,
                 concentration_threshold=0.20,
                 ):
        self.wind = wind
        self.position = np.array(position)

        self.spawn_rate = 10
        self.max_packet = max_packet
        self.concentration = concentration
        self.concentration_threshold = concentration_threshold
        self.decay_rate = decay_rate

        # odor packets list
        self.packets = np.array([])
        self.color = color

        self.boundary = boundary

        self.water_flow_force = water_flow_force
        # self.water_flow_force_threshold = water_flow_force_threshold
        # self.water_flow_force = vector2D.limit_magnitude(self.water_flow_force, self.water_flow_force_threshold,
        #                                                  self.water_flow_force_threshold)
        self.fill = False
        self.t = 0

        for key, value in source_arg.items():
            setattr(self, key, value)

        self.spawn_rate = int(self.concentration / self.spawn_scale)
        print(self.spawn_rate)

    # generate random start position
    def randomStart(self):

        for i in range(2):
            bound_min, bound_max = self.boundary[i]
            if i == 0:
                rate = 0.6
            else:
                rate = 1
            self.position[i] = random.uniform(bound_min, bound_max * rate)
        return self.position

    # generate random Concentration
    def randomConcentration(self):
        self.concentration = random.uniform(0.8, 1)
        self.spawn_rate = int(self.concentration / self.spawn_scale)

    # produce more odor packets
    def spawnPacket(self):
        num_create = min(np.random.poisson(self.spawn_rate), self.max_packet - len(self.packets))
        num_create = max(num_create, 0)
        for i in range(num_create):
            packet_arg = {
                "position": self.position.copy() + np.array([random.uniform(-5, 5), random.uniform(-5, 5)]),
                "concentration": self.concentration,
                "decay_rate": self.decay_rate,
                "water_flow_force": self.water_flow_force.copy(),
                "t": self.t,
            }
            self.packets = np.append(self.packets, [
                OdorPacket(packet_arg, self.wind)])

    @property
    def packet_position(self):
        return np.array([packet.position for packet in self.packets])

    @property
    def packet_concentration(self):
        return np.array([packet.concentration for packet in self.packets])

    @property
    def packet_size(self):
        return np.array([packet.size for packet in self.packets])

    @property
    def current_t(self):
        return self.t

    # odor source is updated for each time step
    # update includes: produce more odor packets, eliminate odorpacket that have concentration under threshold
    def update(self):
        self.t += 1
        for packet in self.packets:
            packet.update(self.t)

        # eliminate Odor Packet whose concentration under threshold
        packets_concentration = self.getConcentration()
        concentration_mask = packets_concentration > self.concentration_threshold
        if np.sum(concentration_mask) < len(self.packets):
            self.fill = True
        self.packets = self.packets[concentration_mask]

        # eliminate OdorPacket that are out of boundary
        position_mask = self.getBoundaryIndex()

        if np.sum(position_mask) < len(self.packets):
            self.fill = True
        self.packets = self.packets[position_mask]

        self.spawnPacket()

    def getBoundaryIndex(self):
        pos = self.packet_position
        position_mask = np.ones((len(pos)), dtype=bool)
        if len(pos > 0):
            for i in range(2):
                bound_min, bound_max = self.boundary[i]
                position_mask = position_mask & (pos[:, i] >= bound_min) & (pos[:, i] <= bound_max)

        return position_mask

    def burnIn(self):
        while not self.fill:
            self.update()

    def reset(self, burn_in=True):
        self.t = self.t_start
        self.packets = np.array([])
        self.fill = False
        if burn_in:
            self.burnIn()

    def getConcentration(self):
        concentrations = np.array([packet.concentration for packet in self.packets])
        return concentrations

    def getScatterInformation(self):
        position_mask = self.getBoundaryIndex()
        positions = self.packet_position[position_mask]
        colors = np.tile(self.color, (len(positions), 1))[position_mask]
        concentrations = np.atleast_2d(self.packet_concentration).T[position_mask]
        sizes = np.atleast_2d(self.packet_size).T[position_mask]
        return positions, np.concatenate((colors, concentrations), axis=1), sizes

    def getSourceScatterInformation(self):
        return self.position, np.concatenate((self.color, self.concentration), axis=1)
