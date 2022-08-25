import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from celluloid import Camera
from simulation.odorSimulation.aquaEnv import AquaEnv
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AquaEnv2DCropPaste(AquaEnv):
    def __init__(self, env_arg, agent_arg, source_arg, wind_arg):
        super(AquaEnv2DCropPaste, self).__init__(env_arg, agent_arg, source_arg, wind_arg)

    def createGif(self, model=None, file_name='../output/gif/gammaCropPaste.gif', episode_len=None):
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

        def drawRect(ax, crop_range):
            # print(crop_range)
            # print(crop_range.shape)
            crop_shape = crop_range[0][:, 1] - crop_range[0][:, 0]
            rect = patches.Rectangle(crop_range[0][:, 0].numpy(), crop_shape[0], crop_shape[1], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        obs = self.reset()  # 1, 3

        if episode_len == None:
            episode_len = self.episode_limit

        fig = plt.figure(figsize=(15, 15))
        ax_simulation = plt.subplot2grid((10, 12), (0, 0), rowspan=5, colspan=8)
        ax_global = plt.subplot2grid((10, 12), (5, 0), rowspan=5, colspan=8)

        ax_agent = plt.subplot2grid((10, 12), (0, 8), rowspan=3, colspan=3)
        ax_gamma = plt.subplot2grid((10, 12), (3, 8), rowspan=3, colspan=3)
        ax_local = plt.subplot2grid((10, 12), (7, 8), rowspan=4, colspan=4)

        # set black background
        axes = [ax_simulation, ax_global, ax_agent, ax_gamma, ax_local]
        for ax in axes:
            ax.set_facecolor('black')

        camera = Camera(fig)

        # set ax_simulation range
        boundary_x, boundary_y = self.boundary
        ax_simulation.set_xlim(boundary_x[0], boundary_x[1])
        ax_simulation.set_ylim(boundary_y[0], boundary_y[1])

        # set ax_agent_view range
        radius = self.agent.radius
        ax_agent.set_xlim(-radius, radius)
        ax_agent.set_ylim(-radius, radius)

        # flip axis
        gamma_value_max = 0

        obses = []
        agent_actions = []
        for i_timestep in tqdm(range(episode_len)):
            action = self.agent.getAction()
            # action = self.agent.getRandomAction()
            obs, reward, done, info = self.step(action)

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

            ax_simulation.text(0.9, 0.1, str(i_timestep), color='white', ha='center', va='center',
                               transform=ax_simulation.transAxes)

            # show source
            source_color = np.array([1, 0, 0])
            x_source, y_source = self.source.position
            ax_simulation.scatter(x_source, y_source, s=int(600 * self.source.concentration),
                                  color=[*source_color, self.source.concentration], marker='o')

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

            # show sensor view
            sensor_total_concentration = np.array(obs[0, 0])
            sensor_color_base = np.array([[[100 / 255, 0, 0]]]).astype(np.float)
            sensor_color = np.minimum(sensor_color_base * sensor_total_concentration, 1)
            ax_agent.imshow(sensor_color)

            # show predict heat map
            obses.append(obs)
            agent_actions.append(torch.tensor([[action]]))
            torch_input = torch.stack(obses, dim=1)
            torch_action = torch.stack(agent_actions, dim=1).type(torch.long)

            global_map = model(torch_input, torch_action)[0][-1].to('cpu').detach().numpy().T  # 30, 10
            local_map = model.local_map[0].to('cpu').detach().numpy().T  # 30, 10
            gamma_map = model.gamma2D_map[0][0].T

            # normalize into sum = 1
            global_map /= np.sum(global_map)
            local_map /= np.sum(local_map)

            # draw crop rectangle
            gamma_value_max = max(gamma_value_max, torch.max(gamma_map).item())
            ax_gamma.imshow(gamma_map, cmap='hot', vmin=0, vmax=gamma_value_max)
            ax_local.imshow(local_map, cmap='hot', vmin=0, vmax=1,
                            extent=[0, self.local_map_size[0], 0, self.local_map_size[1]])
            ax_global.imshow(global_map, cmap='hot', vmin=0, vmax=1,
                             extent=[0, self.global_map_size[0], 0, self.global_map_size[1]])

            drawRect(ax_local, model.local_crop_range)
            drawRect(ax_global, model.global_crop_range)
            camera.snap()

        animation = camera.animate()
        animation.save(file_name, writer='Pillow', dpi=100, fps=5)


class AquaEnvStackT(AquaEnv):
    def __init__(self, env_arg, agent_arg, source_arg, wind_arg):
        super(AquaEnvStackT, self).__init__(env_arg, agent_arg, source_arg, wind_arg)

    def createGif(self, model=None, file_name='../output/gif/stackT.gif', episode_len=None):
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

        def drawRect(ax, crop_range):
            # print(crop_range)
            # print(crop_range.shape)
            crop_shape = crop_range[0][:, 1] - crop_range[0][:, 0]
            rect = patches.Rectangle(crop_range[0][:, 0].numpy(), crop_shape[0], crop_shape[1], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        obs = self.reset()  # 1, 3

        if episode_len == None:
            episode_len = self.episode_limit

        fig = plt.figure(figsize=(15, 15))
        ax_simulation = plt.subplot2grid((10, 12), (0, 0), rowspan=5, colspan=8)

        ax_stackT = plt.subplot2grid((10, 12), (5, 0), rowspan=5, colspan=5)
        ax_predict = plt.subplot2grid((10, 12), (5, 6), rowspan=5, colspan=5)
        ax_predict.invert_yaxis()

        # set black background
        axes = [ax_simulation, ax_stackT, ax_predict]
        for ax in axes:
            ax.set_facecolor('black')

        camera = Camera(fig)

        # set ax_simulation range
        boundary_x, boundary_y = self.boundary
        ax_simulation.set_xlim(boundary_x[0], boundary_x[1])
        ax_simulation.set_ylim(boundary_y[0], boundary_y[1])


        # flip axis
        stack_value_max = 0

        obses = []
        agent_actions = []
        for i_timestep in tqdm(range(episode_len)):
            action = self.agent.getAction()
            # action = self.agent.getRandomAction()
            obs, reward, done, info = self.step(action)

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
            # ax_simulation.text(0.1, 0.8, str(self.wind.water_flow_force), color='white', ha='center', va='center',
            #                    transform=ax_simulation.transAxes)

            ax_simulation.text(0.9, 0.1, str(i_timestep), color='white', ha='center', va='center',
                               transform=ax_simulation.transAxes)

            # show source
            source_color = np.array([1, 0, 0])
            x_source, y_source = self.source.position
            ax_simulation.scatter(x_source, y_source, s=int(600 * self.source.concentration),
                                  color=[*source_color, self.source.concentration], marker='o')

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

            # show predict heat map
            obses.append(obs)
            agent_actions.append(torch.tensor([[action]]))
            torch_input = torch.stack(obses, dim=1)
            torch_action = torch.stack(agent_actions, dim=1).type(torch.long)

            predict_map = model(torch_input, torch_action)[0][-1].to('cpu').detach().numpy()#.T  # 20, 20
            # print(model.curStackT.shape)
            # print(model.curStackT.numpy()[0][::3].shape)
            # try:
            #     stackT = model.curStackT.numpy()[0][::3].sum(axis=0).T  # 20, 20
            # except:
            #     stackT = model.concen_stack.numpy()[0].sum(axis = 0).T#.T # 20, 20

            # normalize into sum = 1
            # predict_map /= np.sum(predict_map)

            # draw crop rectangle
            # stack_value_max = max(stack_value_max, np.max(stackT))
            # ax_stackT.imshow(stackT, cmap='hot', vmin=0, vmax=stack_value_max)
            ax_predict.imshow(predict_map, cmap='hot', vmin=0, vmax=1)#,
                            #extent=[0, self.predict_map_shape[0], 0, self.predict_map_shape[1]])
            # ax_predict.invert_yaxis()


            # drawRect(ax_local, model.local_crop_range)
            # drawRect(ax_global, model.global_crop_range)
            camera.snap()

        animation = camera.animate()
        animation.save(file_name, writer='Pillow', dpi=100, fps=5)
