import operator as op
from functools import reduce
from GPUtil import showUtilization as gpu_usage
import numpy as np
import torch
from torch import nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMNet(nn.Module):
    def __init__(self, feature_dim, action_dim, hid_dim):
        super(LSTMNet, self).__init__()

        self.lstm_num_layer = 1
        self.hid_dim = hid_dim

        lstm_in = 20
        post_obs_dim, post_action_dim = 10, 10
        self.obs_layer = nn.Linear(feature_dim, post_obs_dim)
        self.action_layer = nn.Linear(action_dim, post_action_dim)
        self.lstm_layer = nn.LSTM(input_size=post_obs_dim+post_action_dim, hidden_size=self.hid_dim, batch_first=True, num_layers=1)
        self.bias = 0.1

    def forward(self, obs, action, hidden, get_all=False, heat_map=True):
        # obs = (bsize, time_len, feature_dim)
        # action = (bsize, time_len, action_dim)
        # hidden = (bsize, mem_size, mem_size, feature_dim)
        self.device = obs.device
        device = self.device

        bsize, time_len = obs.shape[:2]

        post_obs = self.obs_layer(obs)
        # print(action.shape)
        post_action = self.action_layer(action)
        lstm_input = torch.cat((post_obs, post_action), dim=-1)

        out, hidden = self.lstm_layer(lstm_input, hidden)
        return out, hidden

    def hiddenInitialize(self, bsize=1, device=DEVICE):
        return torch.zeros(self.lstm_num_layer, bsize, self.hid_dim).float().to(device), \
               torch.zeros(self.lstm_num_layer, bsize, self.hid_dim).float().to(device)


class Generator(nn.Module):
    def __init__(self, belief_dim, up):
        super(Generator, self).__init__()
        self.num_channels = 1
        self.bias = 0.1
        self.belief_dim = belief_dim
        self.ngf = 32
        self.up = up  # 1: 5x5, 2:10x10, 3:20x20
        channel_dims = [self.ngf * 2 ** (self.up - i - 1) for i in range(self.up)]

        prev_dim = belief_dim
        layers = []
        for i in range(self.up):
            if i == 0:
                ext_layers = [nn.ConvTranspose2d(prev_dim, channel_dims[i], 5, 1, 0, bias=False),
                              nn.BatchNorm2d(channel_dims[i]),
                              nn.Softplus(),
                              ]
            else:
                ext_layers = [nn.ConvTranspose2d(channel_dims[i-1], channel_dims[i], 4, 2, 1, bias=False),
                              nn.BatchNorm2d(channel_dims[i]),
                              nn.Softplus(),
                              ]
            layers += ext_layers

        layers += [nn.Conv2d(self.ngf, 1, 1, bias=False),
                   nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def normalizeHeatmap(self, out):
        # out = bsize, channel_dim, imgsize, imgsize
        out = out + self.bias
        out = out / torch.sum(out, dim=(-1, -2))[..., None, None]
        return out  # bsize, channel_dim, imgsize, imgsize

    def forward(self, input):
        out = self.net(input)

        out = self.normalizeHeatmap(out)
        return out


class GeneratorPredict(nn.Module):
    def __init__(self, env_arg, model_arg):
        super(GeneratorPredict, self).__init__()

        for arg in [env_arg, model_arg]:
            for key, value in arg.items():
                setattr(self, key, value)
        self.action_dim = len(self.directions)
        self.lstm_net = LSTMNet(self.feature_dim, len(self.directions), 64)
        self.generator_net = Generator(64, self.up)

    def forward(self, obs, action, get_all=True):
        # obs = (bsize, time_len, 3)
        # action = (bsize, time_len, 1)

        all_global_map = []
        bsize, time_len, _ = obs.shape
        self.time_len = time_len

        self.hidden = self.lstm_net.hiddenInitialize(bsize=bsize)  # , device = DEVICE) # bsize,  k_tap, k_tap, 3

        self.all_pred = torch.zeros(bsize, time_len, 25 * (2**((self.up-1)*2)))

        for i_time_step in range(time_len):
            # stack T
            current_obs = obs[:, i_time_step:i_time_step + 1]
            current_action = action[:, i_time_step:i_time_step + 1]  # (bsize, 1, 1)

            encoded_action = torch.eye(self.action_dim)[current_action.squeeze(-1)]
            # new_obs = torch.cat((current_obs, encoded_action), dim=-1)  # bsize, timelen, 3 + action_dim

            self.prediction, self.hidden = self.lstm_net(current_obs.to(DEVICE), encoded_action.to(DEVICE),
                                                         self.hidden)  # (bsize, 1,64)
            self.prediction = self.generator_net(self.prediction.squeeze(1)[..., None, None]).squeeze(1)
            # print(self.prediction.shape)
            # print(self.all_pred[:, i_time_step].shape)
            self.all_pred[:, i_time_step] = self.prediction.to('cpu').clone().reshape(bsize, -1)

        if get_all:
            map_return = self.all_pred
        else:
            map_return = self.all_pred[:, -1]

        return map_return  # bsize, timelen 30, 10
