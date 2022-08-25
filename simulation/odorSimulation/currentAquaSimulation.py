import numpy as np
import torch
from simulation.odorSimulation.aquaEnv import AquaEnv

env_arg = {
    # env_arg
    "mode": 1,
    "boundary": np.array([(-100, 100), (-100, 100)]),  # boundary of environment
    "boundary_map": np.array([30, 20]),
    "time_step": 200,  # simulation time_length
    "total_map_x": 10,  # Heatmap resolution (10 x20)
    "total_map_y": 4,
    "map_x": 1,
    "map_y": 1,

    # water_flow
    "max_water_flow_force": 7,  # water_flow_force vector
    "min_water_flow_force": 3,  # water_flow_force vector
    "water_flow_force_shape": 2,
    "water_flow_force_scale": 2,
    "max_random_turbulence": 1,
    "random_water_flow": True,  # random_water_flow_force

    # training
    "episode_limit": 120,
    "feature_dim": 3,

    # reward design
    'success_reward': 100,

    "global_boundary": torch.tensor([[0, 300], [-100, 100]]),
    # "local_boundary": torch.tensor([[-100, 100], [-100, 100]]),
    "local_boundary": torch.tensor([[-100, 100], [-100, 100]]),

    "global_map_boundary": torch.tensor([[0, 30], [0, 20]]),
    "local_map_boundary": torch.tensor([[0, 20], [0, 20]]),

    "local_map_size": (20, 20),
    "global_map_size": (30, 20),

    "directions": torch.tensor([[0, -1], [0, 1]]),
    "dt": 5,
}

agent_arg = {
    "start_position": np.array([0, 0]),
    "dt": 8,
    "speed": 1,
    "radius": 10,

    # possible actions
    "directions": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),  # np.array([[0, -1], [0, 1]]), # up/down action
    "action_dim": 4,

    # initialize
    "random_start": False,
    "random_range": np.array([[0.4, 1], [0, 1]]),
    "random_direction": True,

    # action mode
    'action_mode': 2,  # 0: up_down, 1:cyclone

    # updown range
    "updown_range": np.array([0, 1]),

}

source_arg = {
    "position": np.array([0, 0]),
    "source_pos_shape": 8,
    "source_pos_scale": 12,
    "dt": 1,
    "decay_rate": 0.97,
    "spawn_scale": 0.2,  # source produces "concentration/spawn scale" odor packets per time_step
    "size_scale": 1.05,
    "concentration_threshold": 0.3,

    # initialize
    "random_start": True,
    "random_range": np.array([[0, 1], [0.3, 0.7]]),
    "random_direction": True,
    "random_concentration": True,
    "min_concentration": 0.6,
    "max_concentration": 1.0,
    "burn_in": True,
}

wind_arg = {
    # Turbulence
    "energy": 0.2,
    "length_scale": 10,
    "tau_f": 5,
    "sampling_interval": 1 / 2,
}

env_arg['action_dim'] = agent_arg['action_dim']
env_arg['directions'] = torch.tensor(agent_arg['directions'])

# generate environment
#
# env = AquaEnv(env_arg, agent_arg, source_arg, wind_arg)
