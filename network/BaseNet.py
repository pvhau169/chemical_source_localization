import torch.nn as nn
import torch.nn.functional as f

class FullyConnected(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, in_dim, out_dim):
        super(FullyConnected, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, out_dim),
        )

    def forward(self, input):
        return self.layers(input)