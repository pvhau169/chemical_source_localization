import torch
import pickle

def readFile(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return torch.tensor(data).type(torch.float)


def getData(path, variables):
    # X_input, agent_action, distance_map, source_map_position, source_map_position_index
    # path = "../output/data/processedData/voroni/stochastic/old_data/{}/".format(map_x)
    # path = "../output/data/processedData/voroni/stochastic/{}/".format(map_x)


    data = []
    for variable in variables:
        variable_path = path + variable + ".pkl"
        # globals()[variable] = readFile(variable_path)
        data.append(readFile(variable_path))

    return data
    # return data

def getModel(model, model_path):
    model.load_state_dict(torch.load(model_path)['model'])
    return model

