import operator as op
from functools import reduce
from network.PatchUnet import UNet
import numpy as np
import torch
from torch import nn
from GPUtil import showUtilization as gpu_usage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMNet(nn.Module):
    def __init__(self, feature_dim, map_dim):
        super(LSTMNet, self).__init__()

        self.lstm_num_layer = 1
        self.map_dim = map_dim

        lstm_in, lstm_out = 20, 64
        self.hid_dim = lstm_out
        output_dim = np.product(map_dim)

        self.pre_layer = nn.Linear(feature_dim, lstm_in)
        self.lstm_layer = nn.LSTM(input_size=feature_dim, hidden_size=lstm_out, batch_first=True, num_layers=1)
        self.post_layer = nn.Linear(lstm_out, output_dim)
        self.bias = 0.1

    def forward(self, obs, hidden, get_all=False, heat_map=True):
        # obs = (bsize, time_len, feature_dim)
        # action = (bsize, time_len, 1)
        # hidden = (bsize, mem_size, mem_size, feature_dim)
        self.device = obs.device
        device = self.device

        bsize, time_len = obs.shape[:2]


        out = obs
        out, hidden = self.lstm_layer(out, hidden)
        out = nn.Softplus()(self.post_layer(out))  # bsize, time_len, 25
        out = out.reshape(bsize, *self.map_dim)  # bsize, 5, 5

        if heat_map:
            out = out + self.bias
            out = out / torch.sum(out, dim=(-1, -2))[..., None, None]

        return out, hidden

    def hiddenInitialize(self, bsize=1, device=DEVICE):
        return torch.zeros(self.lstm_num_layer, bsize, self.hid_dim).float().to(device), \
               torch.zeros(self.lstm_num_layer, bsize, self.hid_dim).float().to(device)


class LSTMPredict(nn.Module):
    def __init__(self, env_arg, model_arg):
        super(LSTMPredict, self).__init__()

        for arg in [env_arg, model_arg]:
            for key, value in arg.items():
                setattr(self, key, value)
        self.action_dim = len(self.directions)
        self.lstm_net = LSTMNet(self.feature_dim + len(self.directions), self.predict_map_shape)

    def forward(self, obs, action, get_all=True):
        # obs = (bsize, time_len, 3)
        # action = (bsize, time_len, 1)

        all_global_map = []
        bsize, time_len, feature_dim = obs.shape
        self.time_len = time_len

        self.hidden = self.lstm_net.hiddenInitialize(bsize=bsize)  # , device = DEVICE) # bsize,  k_tap, k_tap, 3

        self.all_pred = torch.zeros(bsize, time_len, *self.predict_map_shape)

        for i_time_step in range(time_len):
            # stack T
            current_obs = obs[:, i_time_step:i_time_step + 1]
            current_action = action[:, i_time_step:i_time_step + 1]  # (bsize, 1, 1)

            encoded_action = torch.eye(self.action_dim)[current_action.squeeze(-1)]
            new_obs = torch.cat((current_obs, encoded_action), dim=-1)  # bsize, timelen, 3 + action_dim

            self.prediction, self.hidden = self.lstm_net(new_obs.to(DEVICE), self.hidden)  # (bsize, 5, 5)
            self.all_pred[:, i_time_step] = self.prediction.clone()

        if get_all:
            map_return = self.all_pred
        else:
            map_return = self.all_pred[:, -1]

        return map_return  # bsize, timelen 30, 10
