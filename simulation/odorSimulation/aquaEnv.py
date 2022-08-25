import cv2
import numpy as np
import torch
from celluloid import Camera
from matplotlib import pyplot as plt
import matplotlib.animation as ani
from simulation.odorSimulation import vector
from simulation.odorSimulation.agent import Agent
from simulation.odorSimulation.source import OdorSource
from simulation.odorSimulation.wind import FourierWind
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AquaEnv:

    def __init__(self, env_arg, agent_arg, source_arg, wind_arg):

        self.env_arg, self.agent_arg, self.source_arg, self.wind_arg = env_arg, agent_arg, source_arg, wind_arg

        # set parameters
        args = [env_arg, agent_arg, source_arg]
        for arg in args:
            for key, value in arg.items():
                setattr(self, key, value)

        self.initialize()

    def initialize(self):
        self.t = -1

        # create wind
        self.wind = FourierWind(self.wind_arg, self.env_arg)
        self.env_arg['wind'] = self.wind

        # create sources
        self.source = OdorSource(self.source_arg, self.env_arg)

        # create agents
        self.agent = Agent(self.agent_arg, self.env_arg)

        # set wind
        self.source.setWind(self.wind)
        self.agent.setWind(self.wind)

        # heat map
        print(self.boundary_map[0], self.episode_limit)
        self.total_concentration_map = np.zeros((self.boundary_map[0] + 1, self.boundary_map[1]+1, self.episode_limit))
        self.occurence_rate_map = np.zeros((self.boundary_map[0] + 1, self.boundary_map[1]+1, self.episode_limit))
        self.reset()

    def reset(self):
        self.t = -1
        self.wind.reset()

        # reset sources
        max_burn_in_t = -1
        source = self.source
        source.reset()
        while source.in_boundary != True:
            source.reset()

        max_burn_in_t = max(max_burn_in_t, source.t)

        self.t = max_burn_in_t
        self.t_step = -1
        source.t = max_burn_in_t

        # reset agent
        self.agent.reset()

        # generate data purpose
        self.detect_source = 0

        return self.getObservation()

    def getObservation(self):
        def agentDetect(source, agent):
            packets_position, packets_concentration = source.packets_position, source.packets_concentration
            # position.shape = (n_packet, 2); concentration.shape = (n_packet, 1)

            # check position_mask
            position_mask = np.ones(len(packets_position)).astype(np.bool)
            for i_axis in range(packets_position.shape[1]):
                min_range, max_range = agent.position[i_axis] - agent.radius, agent.position[i_axis] + agent.radius
                axis_mask = (packets_position[:, i_axis] <= max_range) & (packets_position[:, i_axis] >= min_range)
                position_mask = position_mask * axis_mask

            # apply mask to find total/ average concentration
            try:
                detected_packets_concentration = packets_concentration[position_mask]
            except:
                print(packets_position.shape, axis_mask.shape)
                print(position_mask.shape)
                quit()

            total_concentration = np.sum(detected_packets_concentration)
            avg_concentration = total_concentration / (
                    len(detected_packets_concentration) + (len(detected_packets_concentration) == 0))
            return total_concentration, avg_concentration

        # observation = (total_concentration, water_flow_force)
        agent_water_flow_force = self.wind.getTurbulenceVector(self.agent.position[np.newaxis, ...], self.t)  # (1,2)
        total_concen, avg_concen = agentDetect(self.source, self.agent)
        total_concen, avg_concen = np.atleast_2d(total_concen), np.atleast_2d(avg_concen)  # (1, 1)

        obs = np.concatenate((total_concen, agent_water_flow_force), axis=-1)  # (1,3)

        # generate data purpose
        self.detect_total_concentration = total_concen
        self.detect_avg_concentration = avg_concen
        self.detect_water_flow_force = agent_water_flow_force
        if total_concen > 0: self.detect_source = 1

        self.source_concentration = self.source.concentration

        torch_obs = torch.from_numpy(obs).type(torch.float)
        return torch_obs

    def getSensingObservation(self):
        def getMapPosition(positions):
            interval_map = (np.sum(np.abs(self.boundary), axis=1) / self.boundary_map)[np.newaxis, ...]  # (1, 2)
            map_positions = (positions - self.boundary[:, 0][np.newaxis, ...]) / interval_map
            return map_positions.astype(np.int)

        def updateEnvMap():
            packets_position, packets_concentration = self.source.packets_position, self.source.packets_concentration
            packets_map_position = getMapPosition(packets_position)

            # create Map
            # self.total_concentration_map *= (self.t_step - 1)
            # map_x, map_y = self.boundary_map
            # for i_x in range(map_x):
            #     for i_y in range(map_y):
            #         cell_position_mask = (packets_map_position[:, 0] == i_x) & (packets_map_position[:, 1] == i_y)
            #         cell_total_concentration = np.sum(packets_concentration[cell_position_mask])
            #         self.total_concentration_map[i_x, i_y] += cell_total_concentration
            # self.total_concentration_map /= self.t_step
            # self.total_concentration_map /= self.t_step
            map_x, map_y = self.boundary_map
            for i_x in range(map_x):
                for i_y in range(map_y):
                    cell_position_mask = (packets_map_position[:, 0] == i_x) & (packets_map_position[:, 1] == i_y)
                    cell_total_concentration = np.sum(packets_concentration[cell_position_mask])
                    cell_n_packet = len(packets_concentration[cell_position_mask])
                    self.total_concentration_map[i_x, i_y][self.t_step] = cell_total_concentration
                    self.occurence_rate_map[i_x, i_y][self.t_step] = cell_n_packet

        updateEnvMap()
        agent_position = getMapPosition(self.agent.position)[0]  # (2)
        total_concen = self.total_concentration_map[agent_position[0], agent_position[1]][:self.t_step+1]  # (1,time_len)
        if len(total_concen) == 0:
            total_concen = 0
        total_concen = np.atleast_2d(np.mean(total_concen))
        # print(total_concen, np.sum(self.total_concentration_map))

        # observation = (total_concentration, water_flow_force)
        agent_water_flow_force = self.wind.getTurbulenceVector(self.agent.position[np.newaxis, ...], self.t)  # (1,2)
        obs = np.concatenate((total_concen, agent_water_flow_force), axis=-1)  # (1,3)

        # generate data purpose
        self.detect_total_concentration = total_concen
        self.detect_water_flow_force = agent_water_flow_force

        torch_obs = torch.from_numpy(obs).type(torch.float)

        # generate data purpose
        self.detect_total_concentration = total_concen
        self.detect_water_flow_force = agent_water_flow_force
        if total_concen > 0: self.detect_source = 1

        self.source_concentration = self.source.concentration
        return torch_obs

    def step(self, action):
        # actions = (1)
        self.t += 1
        self.t_step += 1  # start from 1
        self.source.update()

        self.agent.act(action)

        # process observation map

        # process source_seeking reward
        reward = -1#0
        done = False

        agent = self.agent
        # check source found
        if vector.distance(agent.position, self.source.position) <= agent.radius:
            reward = self.success_reward
            done = True

        # out of time
        if (self.t_step == self.episode_limit-1) and done == False:
            # calculate reward = start_to_source - end_to_source
            distance_start_to_source = vector.distance(agent.start_position, self.source.position)
            distance_end_to_source = vector.distance(agent.position, self.source.position)
            distance_move = vector.distance(agent.start_position, agent.position)

            reward = distance_start_to_source - distance_end_to_source
            done = True

        return self.getObservation(), torch.tensor(reward), torch.tensor(done), {}

    def createGif(self, model=None, file_name='../output/gif/test.gif', sensor_gif_name='../output/gif/sensor.gif',
                  episode_len=None, epsilon=0):
        def validPosition(positions, boundary):
            final_mask = np.ones(len(positions)).astype(np.bool)
            for i in range(len(boundary)):
                bound_min, bound_max = boundary[i]
                axis_mask = (positions[:, i] >= bound_min) & (positions[:, i] <= bound_max)
                final_mask = final_mask & axis_mask
            return final_mask

        def applyMask(mask):
            self.packets_position = self.packets_position[mask]
            self.packets_concentration = self.packets_concentration[mask]
            self.packets_time_step = self.packets_time_step[mask]
            self.n_packet = len(self.packets_position)

        def scatterProcess(source, boundary):
            packets_position, packets_concentration, packets_time_step = source.packets_position, source.packets_concentration, source.packets_time_step

            # boundary limit
            mask = validPosition(packets_position, boundary)
            packets_position = packets_position[mask]
            packets_concentration = packets_concentration[mask]
            packets_time_step = packets_time_step[mask]

            # return values
            n_packet = packets_position.shape[0]
            packets_color = np.concatenate((np.tile(source_color, (n_packet, 1)), packets_concentration), axis=-1)
            packets_size = 2.4 ** (source.size_scale ** packets_time_step)

            return packets_position, packets_concentration, packets_color, packets_size

        obs = self.reset()
        if model is not None:
            model.reset()
        if episode_len == None:
            episode_len = self.episode_limit

        fig = plt.figure(figsize=(12, 12))
        ax_simulation = plt.subplot2grid((8, 9), (0, 0), rowspan=5, colspan=9)
        ax_agent_view = plt.subplot2grid((8, 9), (6, 0), rowspan=3, colspan=3)
        ax_sensor_view = plt.subplot2grid((8, 9), (6, 3), rowspan=3, colspan=3)

        # set balack back ground
        ax_simulation.set_facecolor('black')
        ax_agent_view.set_facecolor('black')
        ax_sensor_view.set_facecolor('black')
        # fig.patch.set_facecolor('black')

        camera = Camera(fig)

        # load axis box
        ax_simulation.set_xlabel('X label')
        ax_simulation.set_ylabel('Y label')

        # set ax_simulation range
        boundary_x, boundary_y = self.boundary
        ax_simulation.set_xlim(boundary_x[0], boundary_x[1])
        ax_simulation.set_ylim(boundary_y[0], boundary_y[1])

        # set ax_agent_view range
        radius = self.agent.radius
        ax_agent_view.set_xlim(-radius, radius)
        ax_agent_view.set_ylim(-radius, radius)

        # sensor view initialize
        sensor_concentrations = []  # (time_len, 1)

        for i_timestep in tqdm(range(episode_len)):

            # process step
            action = self.agent.getAction()

            obs, reward, done, info = self.step(action)

            # show gif

            # show agent
            agent_position, agent_direction = self.agent.position, self.agent.direction
            agent_expected_position = self.agent.expectPosition(self.agent.direction)
            agent_radius = self.agent.radius

            x1, y1, x2, y2 = *agent_position, *agent_expected_position
            circle = plt.Circle((x1, y1), agent_radius, color='y', alpha=0.3)

            ax_simulation.add_artist(circle)
            ax_simulation.plot([x1, x2], [y1, y2], color='y')

            ax_simulation.text(0.1, 0.9, str(self.agent.position), color='white', ha='center', va='center',
                               transform=ax_simulation.transAxes)

            ax_simulation.text(0.1, 0.8, str(self.wind.water_flow_force), color='white', ha='center', va='center',
                               transform=ax_simulation.transAxes)

            # show source
            source_color = np.array([1, 0, 0])
            x_source, y_source = self.source.position
            ax_simulation.scatter(x_source, y_source, s=int(600 * self.source.concentration),
                                  color=[*source_color, self.source.concentration],
                                  marker='o')

            # show packets
            packets_position, packets_concentration, packets_color, packets_size = scatterProcess(self.source,
                                                                                                  self.boundary)
            ax_simulation.scatter(packets_position[:, 0], packets_position[:, 1], s=30 * packets_size,
                                  color=packets_color)

            # show agent view
            radius, agent_position = self.agent.radius, self.agent.position
            agent_x, agent_y = agent_position
            agent_view_boundary = [(agent_x - radius, agent_x + radius), (agent_y - radius, agent_y + radius)]

            packets_position, packets_concentration, packets_color, packets_size = scatterProcess(self.source,
                                                                                                  agent_view_boundary)

            packets_position = packets_position - agent_position

            ax_agent_view.scatter(packets_position[:, 0], packets_position[:, 1], s=30 * packets_size,
                                  color=packets_color)

            # show sensor view
            sensor_total_concentration = np.sum(packets_concentration)
            sensor_concentrations.append(sensor_total_concentration)
            sensor_color_base = np.array([[[100 / 255, 0, 0]]]).astype(np.float)
            sensor_color = np.minimum(sensor_color_base * sensor_total_concentration, 1)

            ax_sensor_view.imshow(sensor_color)

            # snap gif
            camera.snap()

        animation = camera.animate()
        animation.save(file_name, writer='Pillow', dpi=100, fps=5)

        video_name = "video4"

        # create sensor gif
        image_path = "../output/video/frame{i}.png"
        fig = plt.figure(figsize=(7, 7))
        ax_sensor_view = fig.add_subplot(111)
        fig.patch.set_facecolor('black')
        camera = Camera(fig)
        sensor_concentrations = [0] + sensor_concentrations
        filled_sensor_concentrations = np.linspace(sensor_concentrations[:-1], sensor_concentrations[1:], num=5,
                                                   endpoint=False).T.reshape(-1)
        # write concentration_log
        log_path = "../output/log/{name}.txt".format(name=video_name)
        write_file = open(log_path, "w")
        np.savetxt(write_file, filled_sensor_concentrations)
        write_file.close()

        # for (i, concentration) in enumerate(filled_sensor_concentrations):
        #     ax_sensor_view.clear()
        #
        #     # show concentration
        #     sensor_color = np.minimum(sensor_color_base * concentration, 1)
        #     ax_sensor_view.imshow(sensor_color)
        #     # show time step
        #     ax_sensor_view.text(0.1, 0.9, str(i), color='white', ha='center', va='center',
        #                         transform=ax_sensor_view.transAxes)
        #     fig.savefig(image_path.format(i=i))
        #     camera.snap()
        #
        # video_name = "../output/video/{name}.avi".format(name=video_name)
        #
        #
        # # write video
        # images = [image_path.format(i=i) for i in range(len(filled_sensor_concentrations))]
        # frame = cv2.imread(images[0])
        # height, width, layer = frame.shape
        #
        # video = cv2.VideoWriter(video_name, 0, 25, (width, height))
        #
        # for image in images:
        #     video.write(cv2.imread(image))
        #
        # cv2.destroyAllWindows()
        # video.release()

    def createSourceSensingGif(self, model=None, file_name='../output/gif/sourceSensing.gif', episode_len=100,
                               boundary_map=(10, 5), feature_dim=3, create_data_log=False):
        def validPosition(positions, boundary):
            final_mask = np.ones(len(positions)).astype(np.bool)
            for i in range(len(boundary)):
                bound_min, bound_max = boundary[i]
                axis_mask = (positions[:, i] >= bound_min) & (positions[:, i] <= bound_max)
                final_mask = final_mask & axis_mask
            return final_mask

        def applyMask(mask):
            self.packets_position = self.packets_position[mask]
            self.packets_concentration = self.packets_concentration[mask]
            self.packets_time_step = self.packets_time_step[mask]
            self.n_packet = len(self.packets_position)

        def scatterProcess(source, boundary):
            packets_position, packets_concentration, packets_time_step = source.packets_position, source.packets_concentration, source.packets_time_step

            # boundary limit
            mask = validPosition(packets_position, boundary)
            packets_position = packets_position[mask]
            packets_concentration = packets_concentration[mask]
            packets_time_step = packets_time_step[mask]

            # return values
            n_packet = packets_position.shape[0]
            packets_color = np.concatenate((np.tile(source_color, (n_packet, 1)), packets_concentration), axis=-1)
            packets_size = 2.4 ** (source.size_scale ** packets_time_step)

            return packets_position, packets_concentration, packets_color, packets_size

        obs = self.reset()
        map_x, map_y = boundary_map

        if episode_len == None:
            episode_len = self.episode_limit

        # process model_input, model_output
        model_input = np.zeros((1, episode_len, feature_dim))
        model_output = np.zeros((1, map_x * map_y))

        fig = plt.figure(figsize=(12, 12))
        ax_simulation = plt.subplot2grid((8, 9), (0, 0), rowspan=5, colspan=9)
        ax_agent_view = plt.subplot2grid((8, 9), (6, 0), rowspan=3, colspan=3)
        ax_sensor_view = plt.subplot2grid((8, 9), (6, 3), rowspan=3, colspan=3)
        ax_predict_view = plt.subplot2grid((8, 9), (6, 6), rowspan=3, colspan=3)

        # set balack back ground
        ax_simulation.set_facecolor('black')
        ax_agent_view.set_facecolor('black')
        ax_sensor_view.set_facecolor('black')
        ax_predict_view.set_facecolor('black')
        # fig.patch.set_facecolor('black')

        camera = Camera(fig)

        # load axis box
        ax_simulation.set_xlabel('X label')
        ax_simulation.set_ylabel('Y label')

        # set ax_simulation range
        boundary_x, boundary_y = self.boundary
        ax_simulation.set_xlim(boundary_x[0], boundary_x[1])
        ax_simulation.set_ylim(boundary_y[0], boundary_y[1])

        # set ax_agent_view range
        radius = self.agent.radius
        ax_agent_view.set_xlim(-radius, radius)
        ax_agent_view.set_ylim(-radius, radius)

        # sensor view initialize
        sensor_concentrations = []  # (time_len, 1)
        sensor_water_flow = []

        for i_timestep in tqdm(range(episode_len)):

            # process step
            if self.mode == 0:
                if model is None:
                    action = torch.randint(0, self.agent.action_dim, (1,)).type(torch.long)
                else:
                    action, belief_state = model.getAct(obs)
            else:
                action = self.agent.getUpDownAction()

            obs, reward, done, info = self.step(action)

            # show gif

            # show agent
            agent_position, agent_direction = self.agent.position, self.agent.direction
            agent_expected_position = self.agent.expectPosition()
            agent_radius = self.agent.radius

            x1, y1, x2, y2 = *agent_position, *agent_expected_position
            circle = plt.Circle((x1, y1), agent_radius, color='y', alpha=0.3)

            ax_simulation.add_artist(circle)
            ax_simulation.plot([x1, x2], [y1, y2], color='y')

            ax_simulation.text(0.1, 0.9, str(self.agent.position), color='white', ha='center', va='center',
                               transform=ax_simulation.transAxes)

            # show source
            source_color = np.array([1, 0, 0])
            x_source, y_source = self.source.position
            ax_simulation.scatter(x_source, y_source, s=int(600 * self.source.concentration),
                                  color=[*source_color, self.source.concentration],
                                  marker='o')

            # show packets
            packets_position, packets_concentration, packets_color, packets_size = scatterProcess(self.source,
                                                                                                  self.boundary)
            ax_simulation.scatter(packets_position[:, 0], packets_position[:, 1], s=30 * packets_size,
                                  color=packets_color)

            # show agent view
            radius, agent_position = self.agent.radius, self.agent.position
            agent_x, agent_y = agent_position
            agent_view_boundary = [(agent_x - radius, agent_x + radius), (agent_y - radius, agent_y + radius)]

            packets_position, packets_concentration, packets_color, packets_size = scatterProcess(self.source,
                                                                                                  agent_view_boundary)

            packets_position = packets_position - agent_position

            ax_agent_view.scatter(packets_position[:, 0], packets_position[:, 1], s=30 * packets_size,
                                  color=packets_color)

            # show sensor view
            sensor_total_concentration = np.sum(packets_concentration)
            sensor_total_concentration = np.array(obs[0, 0])
            # sensor_water_flow.append(self.detect_water_flow_force)

            sensor_concentrations.append(sensor_total_concentration)
            sensor_color_base = np.array([[[100 / 255, 0, 0]]]).astype(np.float)
            # print(sensor_color_base.shape, sensor_total_concentration.shape)
            sensor_color = np.minimum(sensor_color_base * sensor_total_concentration, 1)

            ax_sensor_view.imshow(sensor_color)

            # show predict heat map
            input = obs[:, :feature_dim]
            if feature_dim == 2:
                input = obs[:, :feature_dim - 1]
                input_timestep = np.ones(input.shape) * i_timestep
                input = np.concatenate((input, input_timestep), axis=-1)
            model_input[:, i_timestep, :] = input
            if model is not None:
                model_output = model(model_input).numpy()[:, i_timestep]
            predict_heat_map = model_output.reshape(map_x, map_y)
            ax_predict_view.imshow(predict_heat_map.T[::-1], cmap='hot')

            # snap gif
            camera.snap()

        animation = camera.animate()
        animation.save(file_name, writer='Pillow', dpi=100, fps=5)

        video_name = "video4"

        # create sensor gif
        # image_path = "../output/video/frame{i}.png"
        # fig = plt.figure(figsize=(7, 7))
        # ax_sensor_view = fig.add_subplot(111)
        # fig.patch.set_facecolor('black')
        # camera = Camera(fig)
        # sensor_concentrations = [0] + sensor_concentrations
        # filled_sensor_concentrations = np.linspace(sensor_concentrations[:-1], sensor_concentrations[1:], num=5,
        #                                            endpoint=False).T.reshape(-1)
        # write concentration_log
        if create_data_log:
            sensor_concentrations = np.array(sensor_concentrations)
            sensor_water_flow = np.array(sensor_water_flow).squeeze()

            # print(sensor_concentrations.shape, sensor_water_flow.shape)


            concentration_log_path = "../output/log/gif_concentration.txt"
            water_flow_log_path = "../output/log/gif_water_flow.txt"

            write_file = open(concentration_log_path, "w")
            np.savetxt(write_file, sensor_concentrations)
            write_file.close()

            write_file = open(water_flow_log_path, "w")
            np.savetxt(write_file, sensor_water_flow)
            write_file.close()

    def createGammaSensingGif(self, gammaMem, model = None, file_name='../output/gif/sourceSensing.gif', episode_len=100,
                               boundary_map=(10, 5), feature_dim=3, create_data_log=False):
        def validPosition(positions, boundary):
            final_mask = np.ones(len(positions)).astype(np.bool)
            for i in range(len(boundary)):
                bound_min, bound_max = boundary[i]
                axis_mask = (positions[:, i] >= bound_min) & (positions[:, i] <= bound_max)
                final_mask = final_mask & axis_mask
            return final_mask

        def applyMask(mask):
            self.packets_position = self.packets_position[mask]
            self.packets_concentration = self.packets_concentration[mask]
            self.packets_time_step = self.packets_time_step[mask]
            self.n_packet = len(self.packets_position)

        def scatterProcess(source, boundary):
            packets_position, packets_concentration, packets_time_step = source.packets_position, source.packets_concentration, source.packets_time_step

            # boundary limit
            mask = validPosition(packets_position, boundary)
            packets_position = packets_position[mask]
            packets_concentration = packets_concentration[mask]
            packets_time_step = packets_time_step[mask]

            # return values
            n_packet = packets_position.shape[0]
            packets_color = np.concatenate((np.tile(source_color, (n_packet, 1)), packets_concentration), axis=-1)
            packets_size = 2.4 ** (source.size_scale ** packets_time_step)

            return packets_position, packets_concentration, packets_color, packets_size

        obs = self.reset()
        hidden = gammaMem.hiddenInitialize()
        map_x, map_y = boundary_map

        if episode_len == None:
            episode_len = self.episode_limit

        # process model_input, model_output
        model_input = np.zeros((1, episode_len, feature_dim))
        model_output = np.zeros((1, map_x * map_y))

        fig = plt.figure(figsize=(12, 12))
        ax_simulation = plt.subplot2grid((8, 9), (0, 0), rowspan=5, colspan=9)
        ax_gamma_view = plt.subplot2grid((8, 9), (6, 0), rowspan=3, colspan=3)
        ax_predict_view = plt.subplot2grid((8, 9), (6, 3), rowspan=3, colspan=3)
        ax_predict_pause_view = plt.subplot2grid((8, 9), (6, 6), rowspan=3, colspan=3)

        # set balack back ground
        ax_simulation.set_facecolor('black')
        ax_gamma_view.set_facecolor('black')
        ax_predict_view.set_facecolor('black')
        ax_predict_pause_view.set_facecolor('black')

        camera = Camera(fig)

        # load axis box
        ax_simulation.set_xlabel('X label')
        ax_simulation.set_ylabel('Y label')

        # set ax_simulation range
        boundary_x, boundary_y = self.boundary
        ax_simulation.set_xlim(boundary_x[0], boundary_x[1])
        ax_simulation.set_ylim(boundary_y[0], boundary_y[1])

        # set ax_agent_view range
        radius = self.agent.radius


        # sensor view initialize
        sensor_concentrations = []  # (time_len, 1)
        sensor_water_flow = []

        action_list = []
        for i_timestep in tqdm(range(episode_len)):
            action = self.agent.getUpDownAction() #(bsize, 1)
            action_list.append(action)

            obs, reward, done, info = self.step(action)
            #obs = (bsize, feature_dim)

            # show gif

            # show agent
            agent_position, agent_direction = self.agent.position, self.agent.direction
            agent_expected_position = self.agent.expectPosition()
            agent_radius = self.agent.radius

            x1, y1, x2, y2 = *agent_position, *agent_expected_position
            circle = plt.Circle((x1, y1), agent_radius, color='y', alpha=0.3)

            ax_simulation.add_artist(circle)
            ax_simulation.plot([x1, x2], [y1, y2], color='y')

            ax_simulation.text(0.1, 0.9, str(self.agent.position), color='white', ha='center', va='center',
                               transform=ax_simulation.transAxes)

            # show source
            source_color = np.array([1, 0, 0])
            x_source, y_source = self.source.position
            ax_simulation.scatter(x_source, y_source, s=int(600 * self.source.concentration),
                                  color=[*source_color, self.source.concentration],
                                  marker='o')

            # show packets
            packets_position, packets_concentration, packets_color, packets_size = scatterProcess(self.source,
                                                                                                  self.boundary)
            ax_simulation.scatter(packets_position[:, 0], packets_position[:, 1], s=30 * packets_size,
                                  color=packets_color)

            # show gamma mem
            mem, hidden = gammaMem(obs[:,np.newaxis], action[:,np.newaxis], hidden)
            show_mem = mem[0,0] #(mem_size, mem_size)
            ax_gamma_view.imshow(show_mem, cmap='hot')

            # show predict heat map
            if model is not None:
                model_output = model(mem) #(bsize, map_x, map_y)
            predict_heat_map = model_output.reshape(map_x, map_y)
            ax_predict_view.imshow(predict_heat_map.T[::-1], cmap='hot')

            # snap gif
            camera.snap()

        animation = camera.animate()
        animation.save(file_name, writer='Pillow', dpi=100, fps=2)
        return list_action

    def getEpisodeData(self, save_list, model=None, episode_len=None):

        episode_data = {}
        for (instance, variable) in save_list:
            episode_data[variable] = []
        obs = self.reset()
        if model is not None:
            model.reset()
        if episode_len == None:
            episode_len = self.episode_limit
        self.agent_start = self.agent.position
        # sensor view initialize
        sensor_concentrations = []  # (time_len, 1)


        for i_timestep in range(episode_len):

            # process step
            action = self.agent.getAction()
            self.action = action



            obs, reward, done, info = self.step(action)
            self.agent_position = self.agent.position
            for (instance, variable) in save_list:
                episode_data[variable].append(np.array(getattr(eval(instance), variable)).squeeze().tolist())

        # for (instance,variable) in save_list:
        #     episode_data[variable] = np.array(episode_data[variable])
        return episode_data

