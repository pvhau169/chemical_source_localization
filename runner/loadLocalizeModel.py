import torch
from network.mapGeneratorNet import GeneratorPredict
from common.readFiles import getModel
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def getLocalizeModel(model_path):
def getLocalizeModel(model_path):
    env_arg = {
        "predict_map_shape": (5, 5),

        "directions": torch.tensor([[0, -1], [1, 0], [0, 1], [-1, 0]]),
        "dt": 5,

        "feature_dim": 3,
    }

    model_arg = {
        "T": 5,
        "k_tap": 7,
        "T_range": 5,
        "T_threshold": 0,
        "up": 1,
        "skip_connection": False,
        "mu": 1,
    }

    map_x, new_loss, regularizer_constant = 5, False, 0
    model_arg["up"] = min(map_x // 5, 3)
    # path = "../output/ReportModel/stochastic/evaluate/modelWeight/model_{}_{}_{}.pth".format(map_x, new_loss,
    #                                                                                          regularizer_constant)
    #

    # model_path = "../output/ReportModel/stochastic/evaluate/modelWeight/model_{}_{}_{}.pth".format(map_x, new_loss,
    #                                                                                          regularizer_constant)

    model = GeneratorPredict(env_arg, model_arg)
    model = getModel(model, model_path)
    return model.to(DEVICE)
    return model.lstm_net.to(DEVICE)
