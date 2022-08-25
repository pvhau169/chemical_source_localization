import numpy as np
import torch


# Recurrent ExperienceReplay for MultiAgent
class ExperienceReplay:

    def __init__(self, env_arg):
        self.env_arg = env_arg

        for key, value in env_arg.items():
            setattr(self, key, value)

        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        # self.buffers = {'o': torch.empty([self.er_size, self.feature_dim]),  # observation
        #                 'a': torch.empty([self.er_size, 1]).type(torch.long),  # action
        #                 'last_a': torch.empty([self.er_size, 1]).type(torch.long),  # action
        #                 'r': torch.empty([self.er_size, 1]),  # reward
        #                 'o_next': torch.empty([self.er_size, self.feature_dim]),  # next_obs
        #                 'a_onehot': torch.empty([self.er_size, self.action_dim]),
        #                 'last_a_onehot': torch.empty([self.er_size, self.action_dim]),  # action one hot
        #                 'done': torch.empty([self.er_size, 1])
        #                 }

        self.buffers = {'s': torch.empty([self.er_size, self.state_dim]),  # observation
                        'a': torch.empty([self.er_size, 1]).type(torch.long),  # action
                        'last_a': torch.empty([self.er_size, 1]).type(torch.long),  # action
                        'r': torch.empty([self.er_size, 1]),  # reward
                        's_next': torch.empty([self.er_size, self.state_dim]),  # next_obs
                        'a_onehot': torch.empty([self.er_size, self.action_dim]),
                        'last_a_onehot': torch.empty([self.er_size, self.action_dim]),  # action one hot
                        'done': torch.empty([self.er_size, 1])
                        }

        # store the episode

    def storeEpisode(self, episode_batch=1):
        idxs = self.getStorageID()

        for key in episode_batch.keys():
            self.buffers[key][idxs] = episode_batch[key]

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def getStorageID(self, inc=1):
        if self.current_idx + inc <= self.er_size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.er_size:
            overflow = inc - (self.er_size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.er_size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.er_size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
