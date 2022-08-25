import numpy as np
from simulation.odorSimulation.aquaEnv import AquaEnv
from policy.QLearning import QLearning
from simulation.odorSimulation.currentAquaSimulation import *
from runner.loadLocalizeModel import getLocalizeModel


model_arg = {
    'file_name': "q_learning",
    # training parameter
    'er_size': 8000,
    'n_ep_training': 20000,
    'n_ep_evaluate': 200,

    #TODO change buffer ini to True
    'buffer_ini': True,

    #TODO change
    'print_while_learning': True,

    # #env parameter overwrite
    # 'episode_limit': 40,

    # RL parameter
    #TODO check decay
    'epsilon_decay': 150000,
    'bsize_update': 128,
    'gamma': 0.99,
    'grad_norm_clip': 10,
    'target_update_cycle': 5,

    # save parameter
    'save_interval': 300,
    'save_model': True,
    'save_time': True,
    'evaluate_model':True,
}

# localize Model
localize_model = getLocalizeModel()
model_arg['belief_state_model'] = localize_model
env_arg['state_dim'] = 25 #localize_model.hid_dim
env_arg['episode_limit'] = 40


env = AquaEnv(env_arg, agent_arg, source_arg, wind_arg)
model = QLearning(env, model_arg, env_arg)
# model.showEpsilon()
model.learn()