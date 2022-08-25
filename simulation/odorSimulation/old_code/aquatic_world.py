import torch
from matplotlib import pyplot as plt

from simulation.odor2D import OdorSource, OdorPacket
from simulation.agent2D import Agent
from simulation.FourierWind import FourierWind
from simulation.GaussianWind import WindModel
from simulation.FixedGaussianWind import FixedWindModel
from simulation.Rectangle import Rectangle

from simulation.aquatic_world_heat_map_func import *

import random
import simulation.vector2D as vector
from gym import spaces
from celluloid import Camera
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class aquaEnv:
    num_env = 0

    def __init__(self, control_arg, agent_arg, source_arg):

        self.success_reward = 100
        self.t = 0
        self.limit_t = 50
        self.boundary = [(0, 200), (-50, 50)]

        self.map_x, self.map_y = 5, 5
        self.total_map_x, self.total_map_y = 20, 10

        # ini sources
        self.sources = []

        # ini agent
        self.agents = []
        self.partial_total_concentration, self.partial_heat_maps, self.full_heat_maps, self.t_agent, self.agents_mask, self.agents_done = [], [], [], [], [], []
        self.concentration_per_source = []
        self.num_packet_per_source = []

        # init fourier wind

        for key, value in control_arg.items():
            setattr(self, key, value)

        self.wind = None
        self.agent_arg = agent_arg
        self.source_arg = source_arg

        self.action_space = spaces.Discrete(5)
        self.observation_space = np.ones(100)
        self.agents_memory_input = None
        self.initialize()

    def getObservation(self, index=-1, remote_sensing=False):
        agent = self.agents[index]
        heat_map = self.getPartialMap(index)
        total_concen = self.getPartialTotalConcen(index)

        # observation = heatmap + position
        agent_last_action = agent.getLastAction

        heat_map = heat_map.reshape(-1)
        total_concen = total_concen.reshape(-1)

        turbulence_at_agent = self.wind.getTurbulenceVector(agent.position, self.t)
        direction = agent.getDirection
        angle_between = vector.angle_between(turbulence_at_agent, direction)

        turbulence_input = vector.rotate(turbulence_at_agent, angle_between)

        observation = np.concatenate((heat_map, turbulence_input))

        return observation

    def getAllObservation(self):
        obs = []
        for (i, agent) in enumerate(self.agents):
            obs.append(self.getObservation(i))

        return np.array(obs)

    def generatePartialMaps(self):
        map_x, map_y = self.map_x, self.map_y
        for (i, agent) in enumerate(self.agents):
            if self.agents_done[i]:
                continue
            boundary = getAgentBoundary(agent)
            total_concentration, num_packet, mask = getHeatMap(self.sources, boundary, map_x, map_y)
            heat_map = getAvgGridMap(total_concentration, num_packet)
            self.partial_total_concentration[i].append(np.sum(total_concentration, axis=0))
            self.partial_heat_maps[i].append(heat_map)

            self.agents_mask[i] = self.agents_mask[i] | mask

    def getPartialMap(self, index):
        return self.partial_heat_maps[index][-1]

    def getPartialTotalConcen(self, index):
        return self.partial_total_concentration[index][-1]

    def generateFullMaps(self):
        map_x, map_y = self.total_map_x, self.total_map_y
        boundary = self.boundary

        total_concentration, num_packet, mask = getHeatMap(self.sources, boundary, map_x, map_y)
        self.concentration_per_source = (self.concentration_per_source * self.t + total_concentration) / (self.t + 1)
        self.num_packet_per_source = (self.num_packet_per_source * self.t + num_packet) / (self.t + 1)

    def getFullMap(self, index=-1):
        mask = self.agents_mask[index]
        total_concentration, num_packet = self.concentration_per_source[mask].copy(), self.num_packet_per_source[
            mask].copy()

        heat_map = getAvgGridMap(total_concentration, num_packet)
        return heat_map

    def step(self, actions=None, t=1):
        self.success_reward = 200
        reward = -1
        self.t += 1
        old_position = self.agents[0].position
        for source in self.sources:
            source.update()

        if actions is not None:
            for (i, agent) in enumerate(self.agents):
                if self.agents_done[i]:
                    continue
                agent.act(actions[i])

        # encourage rewards
        # agent = self.agents[0]
        # distance_old_agent_to_source = vector.distance(old_position, self.sources[-1].position)
        # distance_current_agent_to_source = vector.distance(agent.position, self.sources[-1].position)
        # reward+= (distance_old_agent_to_source - distance_current_agent_to_source)

        self.generatePartialMaps()
        current_obs = self.getObservation()
        # concentration = current_obs[0]
        # if concentration != 0:
        #     reward = 0

        for (i, agent) in enumerate(self.agents):
            if vector.distance(agent.position, self.sources[-1].position) <= agent.radius:
                reward = self.success_reward
                self.agents_done[i] = True

        if t == self.timer - 1:
            for (i, agent) in enumerate(self.agents):
                self.agents_done[i] = True
                if reward != self.success_reward:
                    distance_IS = vector.distance(agent.start, self.sources[-1].position)
                    distance_DS = vector.distance(agent.position, self.sources[-1].position)
                    distance_ID = vector.distance(agent.start, agent.position)

                    # if distance_ID < distance_IS and distance_DS < distance_IS:
                    #     reward = abs(distance_IS - distance_DS)
                    # else:
                    #     reward = abs(distance_IS - distance_DS) * -1
                    reward = distance_IS - distance_DS

        return self.getObservation(), reward, self.agents_done[0], {}

    def initialize(self):
        self.t = -1

        self.source_arg["t_start"] = self.t
        self.source_arg["boundary"] = self.boundary
        self.source_arg["water_flow_force"] = self.water_flow_force

        # ini sources
        self.sources = [OdorSource(self.source_arg, self.wind) for i in range(self.num_source)]

        for (i, source) in enumerate(self.sources):
            if self.random_source_position:
                source.randomStart()
            if self.random_source_concentration:
                source.randomConcentration()

        # ini agent
        self.agent_arg["boundary"] = self.boundary
        self.agents = [Agent(self.agent_arg) for i in range(self.num_agent)]
        for agent in self.agents:
            if self.random_agent_position:
                agent.randomStart()

        self.wind = FixedWindModel(velocity_init_vector=self.water_flow_force, limit_time=self.limit_t)
        turbulence_arg = {"energy": 0.1,
                          "length_scale": 20,
                          "tau_f": 5,
                          "sampling_interval": 1 / 2,
                          "water_flow_force": self.water_flow_force,

                          }
        self.wind = FourierWind(turbulence_arg)
        self.reset()

    def reset(self, random_reset=True):
        self.t = -1

        if self.random_water_flow:
            self.wind.randomWaterFlow()
        self.wind.reset()
        self.max_t = -1
        for (i, source) in enumerate(self.sources):
            source.wind = self.wind
            source.reset()
            # print(source.t)
            self.max_t = max(self.max_t, source.current_t)
            if self.random_source_concentration and random_reset:
                source.randomConcentration()

        # print(max_t)
        # synchronize all sources
        for (i, source) in enumerate(self.sources):
            source.t = self.max_t
        self.t = self.max_t

        for (i, agent) in enumerate(self.agents):
            agent.reset()
            if self.random_agent_position and random_reset:
                agent.randomStart()

        # ini agent
        self.partial_heat_maps = [[] for i in range(self.num_agent)]
        self.partial_total_concentration = [[] for i in range(self.num_agent)]

        self.concentration_per_source, self.num_packet_per_source = np.array(
            [np.zeros((self.total_map_y, self.total_map_x)) for i in range(len(self.sources))]), np.array(
            [np.zeros((self.total_map_y, self.total_map_x)) for i in range(len(self.sources))])

        self.agents_done = np.zeros(self.num_agent).astype(np.bool8)
        self.agents_mask = [np.zeros(len(self.sources)).astype(np.bool8) for i in range(len(self.agents))]
        self.agents_mask.append(np.ones(len(self.sources)).astype(np.bool8))
        self.agents_mask = np.array(self.agents_mask)

        self.step()
        self.action_space = spaces.Discrete(self.agents[-1].action_size)
        self.observation_space = self.getObservation()

        return self.getObservation()

    def get_start_state(self):
        return self.getObservation()

    def processInputShortTermMemory(self, obses):
        for (i, obs) in enumerate(obses):
            self.agents_memory_input[i].addState(obs)

        output = [self.agents_memory_input[i].getState() for i in range(self.num_agent)]
        return output

    def createGif(self, alg=None, file_name='gif/test.gif', timer=200, epsilon=0.01, recurrent=False):
        fig = plt.figure(figsize=(15, 15))

        show_simulation = True
        show_quiver = False
        show_concentration = True
        show_trajectory = True

        ax_simulation = plt.subplot2grid((15, 9), (0, 0), rowspan=5, colspan=9)
        ax_quiver = plt.subplot2grid((15, 9), (6, 0), rowspan=5, colspan=9)
        ax_partial_heat_map = []
        ax_trajectory = []
        for i in range(self.num_agent):
            ax_partial_heat_map.append(plt.subplot2grid((15, 9), (6, 3 * i), rowspan=3, colspan=3))
            ax_trajectory.append(plt.subplot2grid((15, 9), (12, 3 * i), rowspan=3, colspan=3))

        camera = Camera(fig)

        # load axis box
        ax_simulation.set_xlabel('X Label')
        ax_simulation.set_ylabel('Y Label')

        boundary_x, boundary_y = self.boundary
        ax_simulation.set_xlim(boundary_x[0], boundary_x[1])
        ax_simulation.set_ylim(boundary_y[0], boundary_y[1])

        print(self.boundary)
        for i in range(self.num_agent):
            ax_trajectory[i].set_xlim(boundary_x[0], boundary_x[1])
            ax_trajectory[i].set_ylim(boundary_y[0], boundary_y[1])

        self.reset()
        # fixed-length memory input
        if alg is not None:
            self.agents_memory_input = [ShortTermMemory(alg.memory_size, alg.feature_dim) for i in
                                        range(self.num_agent)]

        max_time_step = min(timer, self.limit_t)

        hidden = alg.memory.hiddenInitialize(bsize=self.num_agent)
        for time_step in tqdm(range(max_time_step), position=0, leave=True):
            # plot quiver windself.step(actions)
            if show_quiver:
                ax_quiver.quiver(self.wind.x_points, self.wind.y_points,
                                 self.wind.velocity_field_created[time_step].T[0],
                                 self.wind.velocity_field_created[time_step].T[1], width=0.003)

                ax_quiver.title.set_text("mean_flow = " + str(self.wind.velocity_init_vector))

            observations = self.getAllObservation()

            if alg is None:
                actions = [torch.randint(0, self.agents[-1].action_size, (1,)).type(torch.LongTensor).item() for i in
                           range(len(self.agents))]
            else:
                num = np.random.rand()
                if num < epsilon:
                    actions = [torch.randint(0, self.agents[-1].action_size, (1,)).type(torch.LongTensor).item() for i
                               in range(len(self.agents))]
                else:
                    # if recurrent:
                    belief_state, hidden = alg.memory.forward(observations, hidden)
                    actions = alg.model(belief_state, bsize=self.num_agent)
                    actions = [torch.argmax(actions, dim=1).cpu().item()]
                    # else:
                    #     observations_input = self.processInputShortTermMemory(observations)
                    #     actions = torch.argmax(alg.model(observations_input, bsize=self.num_agent), dim=1)
                # print(actions)

            self.step(actions)

            if show_simulation:
                # simulate source
                for source in self.sources:
                    pos, colors = source.getScatterInformation()
                    x, y = pos[:, 0], pos[:, 1]
                    x_source, y_source = source.position
                    ax_simulation.scatter(x_source, y_source, s=int(400 * source.concentration), color=source.color,
                                          marker='o')
                    ax_simulation.scatter(x, y, s=[5] * len(x), color=colors, marker='o')

                    ax_simulation.text(0.5, 0.85, 'Timestamp: {time_step}'.format(time_step=time_step),
                                       transform=ax_simulation.transAxes, ha="center")

                ax_simulation.title.set_text('Environment Simulation')

                # simulate agent
                for (i, agent) in enumerate(self.agents):
                    agent_pos = agent.position
                    agent_direction = agent.direction
                    radius = agent.radius
                    color = agent.color
                    x, y = agent_pos
                    x1, y1 = agent.expectPosition(agent_direction, agent.dt)

                    circle = plt.Circle((x, y), radius, color='y', alpha=0.3)
                    ax_simulation.add_artist(circle)

                    ax_simulation.scatter([x], [y], s=20, color=source.color, marker='o')
                    ax_simulation.plot([x, x1], [y, y1], color=color)
                    ax_simulation.annotate(str(i), xy=agent_pos, fontsize=15, ha="center")

            # simulate Agent View

            # partial view
            if show_concentration:
                for (i, agent) in enumerate(self.agents):
                    obs = self.getObservation()
                    heat_map = obs[0].reshape((1, 1))
                    turbulence_input = obs[1:]
                    # heat_map = self.getPartialMap(i)
                    ax_partial_heat_map[i].imshow(heat_map, cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0,
                                                  vmax=1)
                    ax_partial_heat_map[i].title.set_text('Agent s Partial View')
                    ax_partial_heat_map[i].text(0.5, 0.85, 'position: ' + str(turbulence_input), color='white',
                                                transform=ax_partial_heat_map[i].transAxes, ha="center")

            # trajectory
            if show_trajectory:
                for (i, agent) in enumerate(self.agents):
                    agent.update()
                    trajectory = np.array(agent.trajectory)
                    ax_trajectory[i].plot(trajectory[:, 0], trajectory[:, 1], 'r')
                    ax_trajectory[i].set_title(str(agent.position), y=-0.01)
                    ax_trajectory[i].text(0.5, 0.85, 'position: ' + str(agent.position),
                                          transform=ax_trajectory[i].transAxes, ha="center")

            camera.snap()

        animation = camera.animate()
        animation.save(file_name, writer='Pillow', dpi=100, fps=10)

    def createGifGamma(self, alg=None, file_name='gif/test.gif', timer=200, epsilon=0.01, recurrent=False):
        fig = plt.figure(figsize=(10, 10))

        show_simulation = True
        show_quiver = False
        show_concentration = True
        show_trajectory = True

        ax_simulation = plt.subplot2grid((7, 7), (0, 0), rowspan=4, colspan=7)

        ax_partial_heat_map = []
        ax_trajectory = []
        for i in range(self.num_agent):
            ax_partial_heat_map.append(plt.subplot2grid((7, 7), (4, 3 * i), rowspan=3, colspan=3))
            # ax_trajectory.append(plt.subplot2grid((15, 9), (12, 3 * i), rowspan=3, colspan=3))

        camera = Camera(fig)

        # load axis box
        ax_simulation.set_xlabel('X Label')
        ax_simulation.set_ylabel('Y Label')

        boundary_x, boundary_y = self.boundary
        ax_simulation.set_xlim(boundary_x[0], boundary_x[1])
        ax_simulation.set_ylim(boundary_y[0], boundary_y[1])

        # print(self.boundary)
        # for i in range(self.num_agent):
        #     ax_trajectory[i].set_xlim(boundary_x[0], boundary_x[1])
        #     ax_trajectory[i].set_ylim(boundary_y[0], boundary_y[1])

        self.reset()

        max_time_step = min(timer, self.limit_t)

        hidden = alg.memory.hiddenInitialize(bsize=self.num_agent)
        for time_step in tqdm(range(max_time_step), position=0, leave=True):

            observations = self.getObservation()

            num = np.random.rand()
            belief_state, hidden = alg.memory.forward(observations, hidden, action=self.agents[0].direction)
            if num < epsilon:
                actions = [torch.randint(0, self.agents[-1].action_size, (1,)).type(torch.LongTensor).item() for i
                           in range(len(self.agents))]
            else:
                # if recurrent:

                if torch.is_tensor(belief_state) == False:
                    belief_state = torch.from_numpy(belief_state).float()
                actions = alg.model(belief_state.to(DEVICE), bsize=self.num_agent)
                actions = [torch.argmax(actions, dim=1).cpu().item()]

            self.step(actions)

            if show_simulation:
                # simulate source
                for source in self.sources:
                    pos, colors = source.getScatterInformation()
                    x, y = pos[:, 0], pos[:, 1]
                    x_source, y_source = source.position
                    ax_simulation.scatter(x_source, y_source, s=int(400 * source.concentration), color=source.color,
                                          marker='o')
                    ax_simulation.scatter(x, y, s=[5] * len(x), color=colors, marker='o')

                    ax_simulation.text(0.5, 0.85, 'Timestamp: {time_step}'.format(time_step=time_step),
                                       transform=ax_simulation.transAxes, ha="center")

                ax_simulation.title.set_text('Environment Simulation')

                # simulate agent
                for (i, agent) in enumerate(self.agents):
                    agent_pos = agent.position
                    agent_direction = agent.direction
                    radius = agent.radius
                    color = agent.color
                    x, y = agent_pos
                    x1, y1 = agent.expectPosition(agent_direction, agent.dt)

                    circle = plt.Circle((x, y), radius, color='y', alpha=0.3)
                    ax_simulation.add_artist(circle)

                    ax_simulation.scatter([x], [y], s=20, color=source.color, marker='o')
                    ax_simulation.plot([x, x1], [y, y1], color=color)
                    ax_simulation.annotate(str(i), xy=agent_pos, fontsize=15, ha="center")

            # simulate Agent View

            # partial view
            if alg.space_mem_size != None:
                heat_map = belief_state[0, 0, :, :]
                # print(belief_state.shape)
                # print(heat_map.shape)
                # print(torch.sum(torch.Tensor(heat_map)))
                ax_partial_heat_map[i].imshow(heat_map, cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0,
                                              vmax=1)
                ax_partial_heat_map[i].title.set_text('Agent s HeatMap')
                # ax_partial_heat_map[i].text(0.5, 0.85, 'position: ' + str(turbulence_input), color='white',
                #                             transform=ax_partial_heat_map[i].transAxes, ha="center")

            # # trajectory
            # if show_trajectory:
            #     for (i, agent) in enumerate(self.agents):
            #         agent.update()
            #         trajectory = np.array(agent.trajectory)
            #         ax_trajectory[i].plot(trajectory[:, 0], trajectory[:, 1], 'r')
            #         ax_trajectory[i].set_title(str(agent.position), y=-0.01)
            #         ax_trajectory[i].text(0.5, 0.85, 'position: ' + str(agent.position),
            #                               transform=ax_trajectory[i].transAxes, ha="center")

            camera.snap()

        animation = camera.animate()
        animation.save(file_name, writer='Pillow', dpi=100, fps=10)
